"""Switch to an open app, minimize everything, or close an app politely.

- Open windows are matched by the app's name: its Start menu name, its program name, and, only
  when those don't match, the window's title. Titles can hold private things (an email subject,
  a file name), so they are only compared on this PC: they never go to the AI, the voice service
  or the screen, and replies use the app's name instead.
- Closing is polite: the window gets the same message as clicking its X, so the app can still
  ask "Save changes?". Nothing is ever force-killed, only one window closes at a time, and
  J.A.R.V.I.S itself, Explorer and Windows' own system windows are never closed. Closing is
  MEDIUM risk; switching and minimizing are LOW.
- The tools talk to a small `desktop` object, so the tests use a fake one.
"""
from __future__ import annotations

import ctypes
import os
import time
from ctypes import wintypes
from dataclasses import dataclass
from pathlib import PureWindowsPath
from typing import Callable

from ..gate.models import Param, Risk, Tool, ToolError
from .apps import NAME_PATTERN, AppCatalog, match_level

# Windows that are part of Windows itself, or of J.A.R.V.I.S (which runs in python/pythonw).
PROTECTED_PROGRAMS = frozenset({
    "explorer", "python", "pythonw", "py", "pyw", "dwm", "csrss", "winlogon", "lsass", "services",
    "svchost", "sihost", "searchhost", "searchapp", "startmenuexperiencehost", "shellexperiencehost",
    "textinputhost", "lockapp", "logonui",
})
# Hosts the windows of Store apps (Calculator, Settings...): its program name says nothing, but the
# window's title is the app's name.
STORE_APP_HOST = "applicationframehost"

_GWL_EXSTYLE, _WS_EX_TOOLWINDOW, _GW_OWNER = -20, 0x80, 4
_DWMWA_CLOAKED, _SW_RESTORE, _WM_CLOSE = 14, 9, 0x0010
_VK_MENU, _KEYEVENTF_KEYUP = 0x12, 0x2
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000


@dataclass(frozen=True)
class Window:
    handle: int
    pid: int
    program: str                # full path of the program that owns it
    title: str                  # compared on this PC only; never shown, spoken or sent


def _program_path(pid: int) -> str:
    k32 = ctypes.windll.kernel32
    handle = k32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return ""
    try:
        size = wintypes.DWORD(1024)
        buf = ctypes.create_unicode_buffer(size.value)
        ok = k32.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size))
        return buf.value if ok else ""
    finally:
        k32.CloseHandle(handle)


class WindowsDesktop:
    """The real desktop: lists, focuses, minimizes and closes top-level windows."""

    def windows(self) -> list[Window]:
        """Visible app windows, front to back (the order you see them stacked)."""
        u32, dwm = ctypes.windll.user32, ctypes.windll.dwmapi
        found: list[Window] = []

        @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
        def visit(hwnd, _):
            if not u32.IsWindowVisible(hwnd) or u32.GetWindow(hwnd, _GW_OWNER):
                return True
            if u32.GetWindowLongW(hwnd, _GWL_EXSTYLE) & _WS_EX_TOOLWINDOW:
                return True
            cloaked = wintypes.DWORD()
            dwm.DwmGetWindowAttribute(hwnd, _DWMWA_CLOAKED, ctypes.byref(cloaked), ctypes.sizeof(cloaked))
            if cloaked.value:
                return True                     # on another virtual desktop, or a suspended Store app
            length = u32.GetWindowTextLengthW(hwnd)
            if length == 0:
                return True
            title = ctypes.create_unicode_buffer(length + 1)
            u32.GetWindowTextW(hwnd, title, length + 1)
            pid = wintypes.DWORD()
            u32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            found.append(Window(int(hwnd), pid.value, _program_path(pid.value), title.value))
            return True

        u32.EnumWindows(visit, 0)
        return found

    def focus(self, window: Window) -> bool:
        u32 = ctypes.windll.user32
        if u32.IsIconic(window.handle):
            u32.ShowWindow(window.handle, _SW_RESTORE)
        # Windows only lets the app you're using move another window to the front. A tap of the Alt
        # key counts as you using the keyboard, which lifts that rule for this one switch.
        u32.keybd_event(_VK_MENU, 0, 0, 0)
        u32.keybd_event(_VK_MENU, 0, _KEYEVENTF_KEYUP, 0)
        u32.SetForegroundWindow(window.handle)
        for _ in range(10):                     # the switch can take a moment to show
            front = u32.GetForegroundWindow()
            pid = wintypes.DWORD()
            u32.GetWindowThreadProcessId(front, ctypes.byref(pid))
            if front == window.handle or pid.value == window.pid:
                return True
            time.sleep(0.05)
        return False

    def minimize_all(self) -> None:
        from .wincom import shell
        shell().MinimizeAll()

    def close(self, window: Window) -> None:
        if not ctypes.windll.user32.PostMessageW(window.handle, _WM_CLOSE, 0, 0):
            raise ToolError("I couldn't close that app.")


def _stem(path: str) -> str:
    return PureWindowsPath(path).stem.lower() if path else ""


def make_window_tools(catalog: AppCatalog, desktop=None, own_pid: int | None = None) -> list[Tool]:
    desktop = desktop or WindowsDesktop()
    own_pid = os.getpid() if own_pid is None else own_pid

    def app_names(w: Window) -> tuple[list[str], list[str]]:
        """(Start menu names, program name) for a window."""
        program = _stem(w.program)
        if not program or program == STORE_APP_HOST:
            return [], []
        try:
            names = catalog.names_for_program(w.program)
        except ToolError:
            names = []
        return names, [program]

    def find(spoken: str) -> tuple[Window, str]:
        """The front-most window of the one app you meant, and the name to call it by: its Start
        menu name, or else your own words (never the window's title)."""
        scored = []
        for w in desktop.windows():
            menu_names, program = app_names(w)
            by_name = max((match_level(spoken, n) for n in menu_names + program), default=0)
            score = by_name * 2 if by_name else match_level(spoken, w.title)    # names beat titles
            if score:
                choice = (menu_names + program + [""])[0]             # to tell apps apart when asking
                scored.append((score, w, menu_names[0] if menu_names else spoken.strip(), choice))
        if not scored:
            raise ToolError("I can't see that app open.")
        top = max(row[0] for row in scored)
        best = [row[1:] for row in scored if row[0] == top]
        # One app = one program; each Store app window counts as its own app.
        apps = {w.handle if _stem(w.program) == STORE_APP_HOST else _stem(w.program) for w, _, _ in best}
        if len(apps) > 1:
            choices = sorted({choice for _, _, choice in best if choice})[:3]
            if len(choices) > 1:
                raise ToolError("Did you mean " + " or ".join(choices) + "?")
            raise ToolError("More than one app matches. Say its full name.")
        window, label, _ = best[0]                      # windows come front to back
        return window, label

    def switch_to_app(name: str) -> str:
        window, label = find(name)
        if not desktop.focus(window):
            raise ToolError(f"Windows wouldn't let me bring {label} to the front.")
        return f"Switching to {label}."

    def minimize_all() -> str:
        desktop.minimize_all()
        return "Everything is minimized."

    def close_app(name: str) -> str:
        window, label = find(name)
        if window.pid == own_pid or _stem(window.program) in PROTECTED_PROGRAMS or not window.program:
            raise ToolError("I'm not allowed to close that one.")
        desktop.close(window)
        return f"Asking {label} to close."

    name_p = Param(str, max_len=64, pattern=NAME_PATTERN)
    return [
        Tool("switch_to_app", "Bring an app that is already open to the front, by the name the user said.",
             Risk.LOW, {"name": name_p}, switch_to_app),
        Tool("minimize_all", "Minimize every window to show the desktop.", Risk.LOW, {}, minimize_all),
        Tool("close_app", "Close one open app politely, like clicking its X, so it can still ask to save. "
             "Use the name the user said.", Risk.MEDIUM, {"name": name_p}, close_app),
    ]
