"""The key box: a small native window where you paste an API key, shown as dots.

How it stays safe:
- It runs in its OWN short-lived process, like the confirmation popup. The key never goes near
  the web screen: the box hands it to J.A.R.V.I.S over the private pipe the two processes share,
  and J.A.R.V.I.S stores it in the Windows Credential Manager. It is never written to a file,
  logged, or shown.
- The text is masked. Nothing is copied to or cleared from your clipboard.
- Cancel, Esc, closing the window, a crash or 5 minutes without an answer all mean "no key".
- J.A.R.V.I.S asks for a key by a short fixed name; the window's words come from the list below,
  so nothing sent to it can change what it says.

    python -m frontend.keybox --key gemini     (started by J.A.R.V.I.S; prints the key on its pipe)
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, TextIO

TIMEOUT_SECONDS = 300
MAX_KEY_CHARS = 256
_PROJECT_ROOT = Path(__file__).resolve().parents[1]

# The keys the box can ask for: what to call them, and where you get one.
SERVICES = {
    "gemini": ("Gemini (the AI)", "Google AI Studio"),
    "groq": ("Groq (hearing your commands)", "the Groq console"),
    "fish": ("Fish Audio (the voice)", "Fish Audio"),
}


def show_box(service: str, timeout_seconds: float = TIMEOUT_SECONDS,
             on_ready: Callable[[SimpleNamespace], None] | None = None) -> str | None:
    """Show the box and return what was pasted, or None if you cancelled."""
    import tkinter as tk  # imported here, so only the box's process ever needs a screen

    title, source = SERVICES[service]
    root = tk.Tk()
    result: dict = {"key": None}

    def save() -> None:
        value = entry.get().strip()
        if not value:
            return
        result["key"] = value
        root.destroy()

    def cancel() -> None:
        result["key"] = None
        root.destroy()

    root.title("J.A.R.V.I.S - add a key")
    root.resizable(False, False)
    root.attributes("-topmost", True)
    root.protocol("WM_DELETE_WINDOW", cancel)
    frame = tk.Frame(root, padx=22, pady=18)
    frame.pack()
    tk.Label(frame, text=f"Paste your key for {title}", font=("Segoe UI", 11, "bold")).pack(anchor="w")
    tk.Label(frame, text=f"Copy it from {source}, then press Ctrl+V here. It is saved in the Windows\n"
                         "Credential Manager and never shown again.", justify="left", fg="#555555").pack(anchor="w", pady=(4, 10))
    entry = tk.Entry(frame, show="•", width=52, font=("Consolas", 10))
    entry.pack(fill="x")
    row = tk.Frame(frame)
    row.pack(pady=(14, 0))
    tk.Button(row, text="Cancel  (Esc)", width=14, command=cancel).pack(side="left", padx=6)
    tk.Button(row, text="Save  (Enter)", width=14, command=save).pack(side="left", padx=6)
    root.bind("<Escape>", lambda e: cancel())
    entry.bind("<Return>", lambda e: save())
    root.after(int(timeout_seconds * 1000), cancel)

    root.update_idletasks()
    x = (root.winfo_screenwidth() - root.winfo_reqwidth()) // 2
    y = (root.winfo_screenheight() - root.winfo_reqheight()) // 3
    root.geometry(f"+{max(x, 0)}+{max(y, 0)}")
    root.lift()
    root.focus_force()
    entry.focus_set()
    if on_ready is not None:         # a hook for tests
        root.after(0, lambda: on_ready(SimpleNamespace(root=root, entry=entry, save=save, cancel=cancel)))
    root.mainloop()
    return result["key"]


def _child(service: str, show: Callable[[str], str | None] = show_box, out: TextIO | None = None) -> int:
    """Exit codes: 0 a key was given (on stdout), 1 cancelled, 2 bad request, 3 the window failed."""
    out = out if out is not None else sys.stdout
    if service not in SERVICES:
        return 2
    try:
        key = show(service)
    except Exception:
        return 3
    if not isinstance(key, str) or not key or len(key) > MAX_KEY_CHARS or "\n" in key or "\r" in key:
        return 1
    out.write(key)
    out.flush()
    return 0


def ask_key(service: str, runner: Callable[[str], "tuple[int, str]"] | None = None) -> str | None:
    """Open the box in its own process and return the key you pasted, or None."""
    if service not in SERVICES:
        return None

    def run(name: str) -> tuple[int, str]:
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        done = subprocess.run([sys.executable, "-m", "frontend.keybox", "--key", name], capture_output=True,
                              text=True, timeout=TIMEOUT_SECONDS + 10, cwd=_PROJECT_ROOT, creationflags=flags)
        return done.returncode, done.stdout

    try:
        code, output = (runner or run)(service)
    except Exception:
        return None
    if code != 0 or not isinstance(output, str):
        return None
    key = output.strip()
    return key if 0 < len(key) <= MAX_KEY_CHARS else None


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) == 2 and argv[0] == "--key":
        return _child(argv[1])
    print("usage: python -m frontend.keybox --key [gemini|groq|fish]")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
