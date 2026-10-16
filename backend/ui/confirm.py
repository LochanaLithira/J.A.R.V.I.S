"""On-screen confirmation: a native popup that only a real keypress or click can approve.

How it stays safe:
- It runs in a SEPARATE short-lived process. A crash or hang in the window can never freeze
  or corrupt the agent, and a stuck window is killed after a timeout (answer: no).
- FAIL CLOSED: no tkinter, no screen, a crash, a timeout, a closed window, Esc, a wrong
  answer from the window process, or any error at all means DENIED.
- The window cannot be answered by voice, by the AI, or by text inside a tool argument.
- Approval needs the Y key or a click on Allow. Both are ignored for the first second, so a
  keystroke that was already on its way (you typing something else) cannot approve by accident.
- The window shows the tool name and arguments as escaped, clipped plain text, so text inside
  an argument cannot fake extra lines, hide parts of the message, or imitate buttons.

Honest limit: any program running as you can synthesize key presses. This protects against
the AI, prompt injection and misheard speech, not against malware already running as you.

Try it:  python -m backend.ui.confirm --demo
"""
from __future__ import annotations

import json
import re
import secrets
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, TextIO

TIMEOUT_SECONDS = 30.0
ARM_DELAY_MS = 1000
MAX_VALUE_CHARS = 120
MAX_NAME_CHARS = 40
MAX_ARGS_SHOWN = 8
MAX_REQUEST_CHARS = 10_000

_TOKEN_RE = re.compile(r"[0-9a-f]{32}")
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# What the window says
# ---------------------------------------------------------------------------

def _printable(text: str) -> str:
    """Escape anything that is not plainly printable (newlines, control and invisible
    characters, right-to-left overrides, odd spaces) so it cannot fake or hide text."""
    out = []
    for ch in text:
        if ch.isprintable():
            out.append(ch)
        elif ord(ch) <= 0xFFFF:
            out.append(f"\\u{ord(ch):04x}")
        else:
            out.append(f"\\U{ord(ch):08x}")
    return "".join(out)


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + f"... (+{len(text) - limit} more characters)"


def describe_action(tool_name: Any, args: Any) -> str:
    """Plain text for the window: the tool and its arguments, escaped and shortened."""
    lines = [f"Tool: {_clip(_printable(str(tool_name)), MAX_NAME_CHARS)}"]
    if not isinstance(args, dict):
        lines.append("  (arguments are not in the expected format)")
    elif not args:
        lines.append("  (no arguments)")
    else:
        items = list(args.items())
        for key, value in items[:MAX_ARGS_SHOWN]:
            try:
                shown = json.dumps(value, ensure_ascii=False)
            except (TypeError, ValueError):
                shown = "<unprintable value>"
            name = _clip(_printable(str(key)), MAX_NAME_CHARS)
            lines.append(f"  {name} = {_clip(_printable(shown), MAX_VALUE_CHARS)}")
        if len(items) > MAX_ARGS_SHOWN:
            lines.append(f"  ... and {len(items) - MAX_ARGS_SHOWN} more arguments")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# The window itself (runs only in the separate popup process)
# ---------------------------------------------------------------------------

def show_dialog(text: str, timeout_seconds: float = TIMEOUT_SECONDS, arm_delay_ms: int = ARM_DELAY_MS,
                on_ready: Callable[[SimpleNamespace], None] | None = None) -> bool:
    """Show the popup and return True only if the user explicitly allowed it."""
    import tkinter as tk  # imported here, so only the popup process ever needs a screen

    root = tk.Tk()
    state = {"allowed": False, "armed": False, "done": False}
    deadline = time.monotonic() + timeout_seconds

    def finish(allowed: bool) -> None:
        if state["done"]:
            return
        state["done"] = True
        state["allowed"] = allowed
        root.destroy()

    def allow() -> None:
        if state["armed"]:
            finish(True)

    def deny() -> None:
        finish(False)

    def on_key(event) -> None:
        if event.keysym == "Escape":
            deny()
        elif state["armed"] and event.char in ("y", "Y"):
            allow()
        # every other key, including Enter and Space, does nothing

    root.title("J.A.R.V.I.S needs your OK")
    root.resizable(False, False)
    root.attributes("-topmost", True)
    root.protocol("WM_DELETE_WINDOW", deny)

    frame = tk.Frame(root, padx=22, pady=18)
    frame.pack()
    tk.Label(frame, text="J.A.R.V.I.S wants to run this action:",
             font=("Segoe UI", 11, "bold")).pack(anchor="w")
    box = tk.Label(frame, text=text, justify="left", anchor="w", font=("Consolas", 10),
                   relief="groove", padx=10, pady=8, wraplength=520)
    box.pack(fill="x", pady=(10, 8))
    status = tk.Label(frame, text="", anchor="w", fg="#555555")
    status.pack(fill="x")
    row = tk.Frame(frame)
    row.pack(pady=(10, 0))
    deny_button = tk.Button(row, text="Deny  (Esc)", width=16, command=deny)
    allow_button = tk.Button(row, text="Allow  (press Y)", width=16, state="disabled", command=allow)
    deny_button.pack(side="left", padx=6)
    allow_button.pack(side="left", padx=6)

    def arm() -> None:
        state["armed"] = True
        allow_button.config(state="normal")

    def tick() -> None:
        if state["done"]:
            return
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            deny()
            return
        hint = "Press Y to allow, Esc to deny." if state["armed"] else "Please read first..."
        status.config(text=f"{hint}  Auto-deny in {int(remaining) + 1} s.")
        root.after(100, tick)

    root.bind("<Key>", on_key)
    root.after(arm_delay_ms, arm)
    root.after(0, tick)

    root.update_idletasks()
    x = (root.winfo_screenwidth() - root.winfo_reqwidth()) // 2
    y = (root.winfo_screenheight() - root.winfo_reqheight()) // 3
    root.geometry(f"+{max(x, 0)}+{max(y, 0)}")
    root.lift()
    root.focus_force()
    deny_button.focus_set()          # the safe choice has the focus

    if on_ready is not None:         # a hook for tests
        root.after(0, lambda: on_ready(SimpleNamespace(root=root, allow=allow_button, deny=deny_button, box=box)))
    root.mainloop()
    return state["allowed"]


# ---------------------------------------------------------------------------
# The popup process: reads one request on stdin, prints the secret token ONLY if allowed
# ---------------------------------------------------------------------------

def _child(stdin_text: str, show: Callable[[str, float], bool] = show_dialog,
           out: TextIO | None = None) -> int:
    """Exit codes: 0 allowed, 1 denied, 2 bad request, 3 the window failed."""
    out = out if out is not None else sys.stdout
    try:
        request = json.loads(stdin_text)
        text, token, timeout = request["text"], request["token"], float(request["timeout"])
        if not (isinstance(text, str) and len(text) <= 2000 and isinstance(token, str)
                and _TOKEN_RE.fullmatch(token) and 1 <= timeout <= 120):
            return 2
    except Exception:
        return 2
    try:
        allowed = show(text, timeout) is True
    except Exception:
        return 3
    if not allowed:
        return 1
    out.write(token)
    out.flush()
    return 0


# ---------------------------------------------------------------------------
# The agent side
# ---------------------------------------------------------------------------

Runner = Callable[[str, float], "tuple[int, str]"]


def _run_window_process(payload: str, timeout: float) -> tuple[int, str]:
    done = subprocess.run(
        [sys.executable, "-m", "backend.ui.confirm", "--window"],
        input=payload, capture_output=True, text=True, timeout=timeout + 5, cwd=_PROJECT_ROOT,
    )
    return done.returncode, done.stdout


def ask_user(text: str, runner: Runner | None = None, timeout: float = TIMEOUT_SECONDS) -> bool:
    """Show the popup in its own process. True only if that process returns our one-time secret.

    The secret token means an exit code of 0 alone is not enough: a crash, a missing module,
    or the wrong program answering can never count as a yes."""
    runner = runner or _run_window_process
    token = secrets.token_hex(16)
    payload = json.dumps({"text": text, "token": token, "timeout": timeout})   # ASCII-only JSON
    try:
        code, output = runner(payload, timeout)
    except Exception:                    # timeout, missing screen, crash: all mean no
        return False
    return code == 0 and isinstance(output, str) and output.strip() == token


def popup_confirm(tool_name: str, args: dict[str, Any]) -> bool:
    """The confirmation function the gate uses for high-risk tools."""
    try:
        return ask_user(describe_action(tool_name, args)) is True
    except Exception:
        return False


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv == ["--window"]:
        return _child(sys.stdin.read(MAX_REQUEST_CHARS))
    if argv == ["--demo"]:
        allowed = popup_confirm("demo_delete_file", {"name": "report", "note": "line one\nline two (only a demo)"})
        print("ALLOWED" if allowed else "DENIED")
        return 0
    print("usage: python -m backend.ui.confirm --demo")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())