"""The futuristic screen: a separate window process, opened from the tray icon.

How it stays safe:
- It is its OWN process. If it crashes or freezes, J.A.R.V.I.S keeps listening; closing it frees
  all its memory.
- No network port at all. The page is loaded from our own files as a file:// address (pywebview
  would otherwise start a local web server), and the window talks to J.A.R.V.I.S only through
  the stdin/stdout pipes the two processes share. No other program can reach those pipes.
- J.A.R.V.I.S -> window: only a fixed list of screen updates (HUD calls) is accepted, and every
  value is passed as JSON, never pasted into code. The page itself shows text with textContent,
  never as HTML, and its Content-Security-Policy blocks outside scripts and connections.
- window -> J.A.R.V.I.S: exactly one possible command, "toggle_pause". The window cannot run tools.
- Private mode (nothing stored), no downloads, no right-click menu, no developer tools.

    python -m backend.ui.window      (started by the app; reads updates on stdin)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable, Iterable, TextIO

PAGE = Path(__file__).resolve().parent / "screen" / "index.html"
_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# The only screen calls J.A.R.V.I.S may make, with how many arguments each takes.
ALLOWED_CALLS = {"setState": 2, "setLevel": 1, "addMessage": 3, "setTimers": 1, "addActivity": 1, "setSystems": 1}
MAX_LINE_CHARS = 200_000
COMMANDS = {"toggle_pause"}


def to_js(message: Any) -> str | None:
    """Turn one update from J.A.R.V.I.S into a call on the page, or None if it isn't allowed."""
    if not isinstance(message, dict):
        return None
    name, args = message.get("call"), message.get("args", [])
    if name not in ALLOWED_CALLS or not isinstance(args, list) or len(args) > ALLOWED_CALLS[name]:
        return None
    try:
        payload = json.dumps(args, ensure_ascii=True, allow_nan=False)
    except (TypeError, ValueError):
        return None
    return f"window.HUD && window.HUD.{name}(...{payload})"


# ---------------------------------------------------------------------------
# The window process
# ---------------------------------------------------------------------------

class _Api:
    """What the page may call. Exactly one thing: ask to pause or resume listening."""

    def __init__(self, out: TextIO, lock: threading.Lock):
        self._out, self._lock = out, lock

    def toggle_pause(self) -> None:
        _write_line(self._out, self._lock, {"cmd": "toggle_pause"})


def _write_line(out: TextIO, lock: threading.Lock, message: dict) -> None:
    with lock:
        try:
            out.write(json.dumps(message) + "\n")
            out.flush()
        except (OSError, ValueError):
            pass                                     # J.A.R.V.I.S has gone; nothing to tell


def pump(lines: Iterable[str], window: Any, ready: threading.Event) -> None:
    """Apply updates from J.A.R.V.I.S to the page. When the pipe closes, close the window."""
    for raw in lines:
        if len(raw) > MAX_LINE_CHARS:
            continue
        try:
            message = json.loads(raw)
        except ValueError:
            continue
        if isinstance(message, dict) and message.get("call") == "focus":
            try:
                window.restore()
                window.show()
            except Exception:
                pass
            continue
        js = to_js(message)
        if js is None:
            continue
        ready.wait(15)
        try:
            window.evaluate_js(js)
        except Exception:
            pass                                     # the page is closing
    try:
        window.destroy()
    except Exception:
        pass


def run_window() -> int:
    import webview

    inp = sys.stdin if sys.stdin is not None else open(0, "r", encoding="utf-8", closefd=False)
    out = sys.stdout if sys.stdout is not None else open(1, "w", encoding="utf-8", closefd=False)
    lock = threading.Lock()
    webview.settings["ALLOW_DOWNLOADS"] = False
    webview.settings["SHOW_DEFAULT_MENUS"] = False        # no right-click menu, so no "inspect"
    webview.settings["OPEN_DEVTOOLS_IN_DEBUG"] = False
    window = webview.create_window("J.A.R.V.I.S", PAGE.as_uri(), js_api=_Api(out, lock), width=1280, height=800,
                                   min_size=(900, 600), background_color="#02070d")
    ready = threading.Event()

    def on_loaded() -> None:
        try:
            window.evaluate_js("window.HUD && window.HUD.attach()")
        finally:
            ready.set()
            _write_line(out, lock, {"event": "ready"})

    window.events.loaded += on_loaded
    threading.Thread(target=pump, args=(inp, window, ready), name="screen-pump", daemon=True).start()
    webview.start(private_mode=True, debug=False)        # a file:// page: no local web server
    return 0


# ---------------------------------------------------------------------------
# The app's side: starting the window and keeping it up to date
# ---------------------------------------------------------------------------

def _launch() -> Any:
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    return subprocess.Popen([sys.executable, "-m", "backend.ui.window"], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, encoding="utf-8",
                            bufsize=1, cwd=_PROJECT_ROOT, creationflags=flags)


class ScreenLink:
    """Owns the window process. Everything here is safe to call from any thread, and does nothing
    (cheaply) while the window is closed. It remembers the recent conversation and activity in
    memory only, so a newly opened window can show them."""

    MAX_MESSAGES = 60
    MAX_ACTIVITY = 40
    LEVEL_INTERVAL = 1 / 12                     # at most 12 sound-level updates a second
    REFRESH_SECONDS = 2.0

    def __init__(self, on_command: Callable[[str], None] | None = None,
                 get_systems: Callable[[], list[dict]] | None = None,
                 get_timers: Callable[[], list[dict]] | None = None,
                 launcher: Callable[[], Any] = _launch):
        self._on_command, self._get_systems, self._get_timers = on_command, get_systems, get_timers
        self._launcher = launcher
        self._proc: Any = None
        self._lock = threading.Lock()
        self._ready = False
        self._state: tuple[str, str] = ("starting", "")
        self._messages: deque = deque(maxlen=self.MAX_MESSAGES)
        self._activity: deque = deque(maxlen=self.MAX_ACTIVITY)
        self._last_level = 0.0

    # ---- window lifetime ------------------------------------------------------
    @property
    def is_open(self) -> bool:
        proc = self._proc
        return proc is not None and proc.poll() is None

    def open(self) -> bool:
        """Open the window, or bring it forward if it is already open."""
        with self._lock:
            if self.is_open:
                self._write({"call": "focus"})
                return True
            try:
                self._proc = self._launcher()
            except Exception:
                self._proc = None
                return False
            self._ready = False
        threading.Thread(target=self._read, args=(self._proc,), name="screen-read", daemon=True).start()
        threading.Thread(target=self._refresh, args=(self._proc,), name="screen-refresh", daemon=True).start()
        return True

    def close(self) -> None:
        proc = self._proc
        if proc is not None:
            try:
                proc.stdin.close()                   # the window closes itself when its pipe ends
                proc.wait(3)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass

    def _read(self, proc: Any) -> None:
        for raw in proc.stdout:
            try:
                message = json.loads(raw)
            except ValueError:
                continue
            if not isinstance(message, dict):
                continue
            if message.get("event") == "ready":
                self._ready = True
                self._snapshot()
            elif message.get("cmd") in COMMANDS and self._on_command is not None:
                try:
                    self._on_command(message["cmd"])
                except Exception:
                    pass

    def _refresh(self, proc: Any) -> None:
        while proc.poll() is None:
            time.sleep(self.REFRESH_SECONDS)
            if self._ready:
                self._send_timers()

    def _write(self, message: dict) -> None:
        proc = self._proc
        if proc is None or proc.poll() is not None:
            return
        try:
            proc.stdin.write(json.dumps(message) + "\n")
            proc.stdin.flush()
        except (OSError, ValueError):
            pass

    def _call(self, name: str, *args: Any) -> None:
        if self._ready:
            with self._lock:
                self._write({"call": name, "args": list(args)})

    def _snapshot(self) -> None:
        state, detail = self._state
        self._call("setState", state, detail)
        if self._get_systems is not None:
            try:
                self._call("setSystems", self._get_systems())
            except Exception:
                pass
        self._send_timers()
        for who, text, private in list(self._messages):
            self._call("addMessage", who, text, {"private": private})
        for item in reversed(list(self._activity)):
            self._call("addActivity", item)

    def _send_timers(self) -> None:
        if self._get_timers is not None:
            try:
                self._call("setTimers", self._get_timers())
            except Exception:
                pass

    # ---- updates from the app ---------------------------------------------------------
    def set_state(self, state: str, detail: str = "") -> None:
        self._state = (state, detail)
        self._call("setState", state, detail)

    def add_message(self, who: str, text: str, private: bool = False) -> None:
        self._messages.append((who, text, private))
        self._call("addMessage", who, text, {"private": private})

    def add_activity(self, text: str, kind: str = "ok") -> None:
        item = {"time": time.strftime("%H:%M:%S"), "text": text, "kind": kind}
        self._activity.appendleft(item)
        self._call("addActivity", item)

    def set_level(self, level: float) -> None:
        now = time.monotonic()
        if self._ready and now - self._last_level >= self.LEVEL_INTERVAL:
            self._last_level = now
            self._call("setLevel", round(max(0.0, min(1.0, level)), 3))

    def refresh_timers(self) -> None:
        self._send_timers()


if __name__ == "__main__":
    raise SystemExit(run_window())
