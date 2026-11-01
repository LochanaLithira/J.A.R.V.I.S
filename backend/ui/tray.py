"""The tray icon next to the clock, built directly on Windows' own functions (no extra library).

- The icon's colour shows what J.A.R.V.I.S is doing: green listening, blue working, grey paused
  (microphone off), red problem, amber starting. Hovering shows the same in words.
- Left-click opens the J.A.R.V.I.S screen. Right-click for the menu: Open J.A.R.V.I.S,
  Pause / Resume listening, and Quit.
- It owns a hidden window that receives Windows' own events: the screen being locked or
  unlocked, and the laptop going to sleep or waking. J.A.R.V.I.S uses these to close the
  microphone while you're away.
- set_status() and notify() are safe to call from any thread: they only post a message to the
  window, and the window's own thread updates the icon.
"""
from __future__ import annotations

import ctypes
import logging
import struct
import threading
import time
from collections import deque
from typing import Any, Callable

import numpy as np

log = logging.getLogger("jarvis")

GREY = (127, 140, 141)
STATUS_LOOK: dict[str, tuple[tuple[int, int, int], str]] = {
    "starting": ((243, 156, 18), "Starting"),
    "listening": ((46, 204, 113), "Listening"),
    "working": ((52, 152, 219), "Working on a command"),
    "paused": (GREY, "Paused (microphone off)"),
    "locked": (GREY, "Paused while Windows is locked (microphone off)"),
    "asleep": (GREY, "Paused while asleep (microphone off)"),
    "problem": ((231, 76, 60), "Problem"),
    "stopped": (GREY, "Stopped"),
}
NOTICE_REPEAT_SECONDS = 300           # the same notice is not shown again within 5 minutes


def tooltip_text(status: Any, problem: str = "") -> str:
    key = getattr(status, "value", status)
    label = STATUS_LOOK.get(key, (GREY, str(key)))[1]
    text = f"J.A.R.V.I.S: {label}" + (f": {problem}" if problem else "")
    return text if len(text) <= 127 else text[:124] + "..."


def icon_image(size: int, rgb: tuple[int, int, int]) -> bytes:
    """A round "reactor" icon (a ring with a bright core), as icon-resource bytes for Windows."""
    y, x = np.mgrid[0:size, 0:size] + 0.5
    r = np.hypot(x - size / 2, y - size / 2)
    edge = lambda d: np.clip(d + 0.5, 0.0, 1.0)            # soft, anti-aliased edges
    ring = edge(size * 0.47 - r) * edge(r - size * 0.32)
    core = edge(size * 0.20 - r)
    alpha = np.maximum(ring, core)
    base = np.array(rgb, dtype=np.float64)
    light = base + (255 - base) * 0.55
    colour = np.where((core >= ring)[..., None], light, base)
    bgra = np.zeros((size, size, 4), dtype=np.uint8)
    bgra[..., 0], bgra[..., 1], bgra[..., 2] = colour[..., 2], colour[..., 1], colour[..., 0]
    bgra[..., 3] = (alpha * 255).round().astype(np.uint8)
    pixels = bgra[::-1].tobytes()                          # bottom row first, as Windows expects
    mask = b"\x00" * (((size + 31) // 32) * 4 * size)      # transparency comes from the alpha channel
    header = struct.pack("<IiiHHIIiiII", 40, size, size * 2, 1, 32, 0, len(pixels), 0, 0, 0, 0)
    return header + pixels + mask


# ---------------------------------------------------------------------------
# Windows functions (loaded only when a tray is actually created)
# ---------------------------------------------------------------------------

WM_DESTROY, WM_CLOSE, WM_NULL = 0x0002, 0x0010, 0x0000
WM_CONTEXTMENU, WM_LBUTTONUP, WM_RBUTTONUP = 0x007B, 0x0202, 0x0205
WM_POWERBROADCAST, WM_WTSSESSION_CHANGE = 0x0218, 0x02B1
WM_APP = 0x8000
PBT_APMSUSPEND, PBT_APMRESUMESUSPEND, PBT_APMRESUMEAUTOMATIC = 0x4, 0x7, 0x12
WTS_SESSION_LOCK, WTS_SESSION_UNLOCK = 0x7, 0x8
NIM_ADD, NIM_MODIFY, NIM_DELETE = 0, 1, 2
NIF_MESSAGE, NIF_ICON, NIF_TIP, NIF_INFO = 0x1, 0x2, 0x4, 0x10
NIIF_INFO = 0x1
MF_STRING, MF_SEPARATOR = 0x0, 0x800
TPM_RIGHTBUTTON, TPM_NONOTIFY, TPM_RETURNCMD = 0x2, 0x80, 0x100
SM_CXSMICON = 49

_api: Any = None


def _load_api() -> Any:
    """Declare every Windows function with exact argument types (handles are 64-bit)."""
    global _api
    if _api is not None:
        return _api
    from ctypes import wintypes as w

    LRESULT = ctypes.c_ssize_t
    WNDPROC = ctypes.WINFUNCTYPE(LRESULT, w.HWND, w.UINT, w.WPARAM, w.LPARAM)
    CTRL_HANDLER = ctypes.WINFUNCTYPE(w.BOOL, w.DWORD)

    class WNDCLASSEXW(ctypes.Structure):
        _fields_ = [("cbSize", w.UINT), ("style", w.UINT), ("lpfnWndProc", WNDPROC), ("cbClsExtra", ctypes.c_int),
                    ("cbWndExtra", ctypes.c_int), ("hInstance", w.HINSTANCE), ("hIcon", w.HICON),
                    ("hCursor", w.HANDLE), ("hbrBackground", w.HBRUSH), ("lpszMenuName", w.LPCWSTR),
                    ("lpszClassName", w.LPCWSTR), ("hIconSm", w.HICON)]

    class GUID(ctypes.Structure):
        _fields_ = [("Data1", ctypes.c_ulong), ("Data2", ctypes.c_ushort), ("Data3", ctypes.c_ushort),
                    ("Data4", ctypes.c_ubyte * 8)]

    class NOTIFYICONDATAW(ctypes.Structure):
        _fields_ = [("cbSize", w.DWORD), ("hWnd", w.HWND), ("uID", w.UINT), ("uFlags", w.UINT),
                    ("uCallbackMessage", w.UINT), ("hIcon", w.HICON), ("szTip", w.WCHAR * 128),
                    ("dwState", w.DWORD), ("dwStateMask", w.DWORD), ("szInfo", w.WCHAR * 256),
                    ("uVersion", w.UINT), ("szInfoTitle", w.WCHAR * 64), ("dwInfoFlags", w.DWORD),
                    ("guidItem", GUID), ("hBalloonIcon", w.HICON)]

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    shell32 = ctypes.WinDLL("shell32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    wtsapi32 = ctypes.WinDLL("wtsapi32", use_last_error=True)

    def sig(fn, args, res):
        fn.argtypes, fn.restype = args, res
        return fn

    P = ctypes.POINTER
    api = type("WinApi", (), {})()
    api.WNDPROC, api.CTRL_HANDLER = WNDPROC, CTRL_HANDLER
    api.WNDCLASSEXW, api.NOTIFYICONDATAW, api.MSG, api.POINT = WNDCLASSEXW, NOTIFYICONDATAW, w.MSG, w.POINT
    api.RegisterClassExW = sig(user32.RegisterClassExW, [P(WNDCLASSEXW)], w.ATOM)
    api.UnregisterClassW = sig(user32.UnregisterClassW, [w.LPCWSTR, w.HINSTANCE], w.BOOL)
    api.CreateWindowExW = sig(user32.CreateWindowExW, [w.DWORD, w.LPCWSTR, w.LPCWSTR, w.DWORD, ctypes.c_int,
                                                       ctypes.c_int, ctypes.c_int, ctypes.c_int, w.HWND, w.HMENU,
                                                       w.HINSTANCE, w.LPVOID], w.HWND)
    api.DestroyWindow = sig(user32.DestroyWindow, [w.HWND], w.BOOL)
    api.DefWindowProcW = sig(user32.DefWindowProcW, [w.HWND, w.UINT, w.WPARAM, w.LPARAM], LRESULT)
    api.PostMessageW = sig(user32.PostMessageW, [w.HWND, w.UINT, w.WPARAM, w.LPARAM], w.BOOL)
    api.SendMessageW = sig(user32.SendMessageW, [w.HWND, w.UINT, w.WPARAM, w.LPARAM], LRESULT)
    api.GetMessageW = sig(user32.GetMessageW, [P(w.MSG), w.HWND, w.UINT, w.UINT], ctypes.c_int)
    api.TranslateMessage = sig(user32.TranslateMessage, [P(w.MSG)], w.BOOL)
    api.DispatchMessageW = sig(user32.DispatchMessageW, [P(w.MSG)], LRESULT)
    api.PostQuitMessage = sig(user32.PostQuitMessage, [ctypes.c_int], None)
    api.RegisterWindowMessageW = sig(user32.RegisterWindowMessageW, [w.LPCWSTR], w.UINT)
    api.CreatePopupMenu = sig(user32.CreatePopupMenu, [], w.HMENU)
    api.AppendMenuW = sig(user32.AppendMenuW, [w.HMENU, w.UINT, ctypes.c_size_t, w.LPCWSTR], w.BOOL)
    api.TrackPopupMenu = sig(user32.TrackPopupMenu, [w.HMENU, w.UINT, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                                     w.HWND, ctypes.c_void_p], ctypes.c_int)
    api.DestroyMenu = sig(user32.DestroyMenu, [w.HMENU], w.BOOL)
    api.GetCursorPos = sig(user32.GetCursorPos, [P(w.POINT)], w.BOOL)
    api.SetForegroundWindow = sig(user32.SetForegroundWindow, [w.HWND], w.BOOL)
    api.CreateIconFromResourceEx = sig(user32.CreateIconFromResourceEx,
                                       [ctypes.c_char_p, w.DWORD, w.BOOL, w.DWORD, ctypes.c_int, ctypes.c_int,
                                        w.UINT], w.HICON)
    api.DestroyIcon = sig(user32.DestroyIcon, [w.HICON], w.BOOL)
    api.GetSystemMetrics = sig(user32.GetSystemMetrics, [ctypes.c_int], ctypes.c_int)
    api.Shell_NotifyIconW = sig(shell32.Shell_NotifyIconW, [w.DWORD, P(NOTIFYICONDATAW)], w.BOOL)
    api.GetModuleHandleW = sig(kernel32.GetModuleHandleW, [w.LPCWSTR], w.HMODULE)
    api.GetConsoleWindow = sig(kernel32.GetConsoleWindow, [], w.HWND)
    api.SetConsoleCtrlHandler = sig(kernel32.SetConsoleCtrlHandler, [CTRL_HANDLER, w.BOOL], w.BOOL)
    api.WTSRegisterSessionNotification = sig(wtsapi32.WTSRegisterSessionNotification, [w.HWND, w.DWORD], w.BOOL)
    api.WTSUnRegisterSessionNotification = sig(wtsapi32.WTSUnRegisterSessionNotification, [w.HWND], w.BOOL)
    _api = api
    return api


class Tray:
    WM_TRAY = WM_APP + 1               # mouse events on the icon
    WM_REFRESH = WM_APP + 2            # "status or notice changed" from another thread
    CMD_PAUSE, CMD_QUIT, CMD_OPEN = 1, 2, 3

    def __init__(self, on_toggle_pause: Callable[[], None] | None = None,
                 is_paused: Callable[[], bool] | None = None,
                 on_lock: Callable[[bool], None] | None = None,
                 on_sleep: Callable[[bool], None] | None = None,
                 on_quit: Callable[[], None] | None = None, show_icon: bool = True,
                 on_open: Callable[[], Any] | None = None):
        self._on_toggle_pause, self._is_paused, self._on_open = on_toggle_pause, is_paused, on_open
        self._on_lock, self._on_sleep, self._on_quit = on_lock, on_sleep, on_quit
        self._show_icon = show_icon
        self._lock = threading.Lock()
        self._status: tuple[str, str] = ("starting", "")
        self._notes: deque = deque(maxlen=5)
        self._shown_notes: dict[tuple[str, str], float] = {}
        self._hwnd = None
        self._icons: dict[str, Any] = {}
        self._icon_added = False
        self._class_name = f"JarvisTrayWindow{id(self)}"
        self.ready = threading.Event()   # set once the window exists (for tests)

    # ---- safe from any thread -------------------------------------------------------
    def set_status(self, status: Any, problem: str = "") -> None:
        with self._lock:
            self._status = (getattr(status, "value", status), problem)
        self._post(self.WM_REFRESH)

    def notify(self, title: str, text: str) -> None:
        key, now = (title, text), time.monotonic()
        with self._lock:
            if now - self._shown_notes.get(key, -1e9) < NOTICE_REPEAT_SECONDS:
                return
            self._shown_notes[key] = now
            self._notes.append(key)
        self._post(self.WM_REFRESH)

    def quit(self) -> None:
        self._post(WM_CLOSE)

    def _post(self, msg: int) -> None:
        hwnd = self._hwnd
        if hwnd:
            _load_api().PostMessageW(hwnd, msg, 0, 0)

    # ---- the window's own thread ------------------------------------------------------
    def run(self) -> None:
        """Create the hidden window and the icon, then handle Windows messages until Quit."""
        api = _load_api()
        hinst = api.GetModuleHandleW(None)
        self._wndproc_ref = api.WNDPROC(self._wndproc)        # keep a reference, or Windows calls freed memory
        wc = api.WNDCLASSEXW()
        wc.cbSize = ctypes.sizeof(api.WNDCLASSEXW)
        wc.lpfnWndProc = self._wndproc_ref
        wc.hInstance = hinst
        wc.lpszClassName = self._class_name
        if not api.RegisterClassExW(ctypes.byref(wc)):
            raise OSError("Could not register the tray window.")
        # A normal (never shown) top-level window: message-only windows miss lock and power events.
        self._hwnd = api.CreateWindowExW(0, self._class_name, "J.A.R.V.I.S", 0, 0, 0, 0, 0,
                                         None, None, hinst, None)
        if not self._hwnd:
            api.UnregisterClassW(self._class_name, hinst)
            raise OSError("Could not create the tray window.")
        self._taskbar_created = api.RegisterWindowMessageW("TaskbarCreated")
        api.WTSRegisterSessionNotification(self._hwnd, 0)      # 0 = this session only
        if self._show_icon:
            self._add_icon()
        self._ctrl_ref = None
        if api.GetConsoleWindow():                             # Ctrl+C in a console means Quit
            self._ctrl_ref = api.CTRL_HANDLER(lambda kind: bool(self.quit()) or True)
            api.SetConsoleCtrlHandler(self._ctrl_ref, True)
        self.ready.set()
        msg = api.MSG()
        try:
            while api.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
                api.TranslateMessage(ctypes.byref(msg))
                api.DispatchMessageW(ctypes.byref(msg))
        finally:
            if self._ctrl_ref is not None:
                api.SetConsoleCtrlHandler(self._ctrl_ref, False)
            for icon in self._icons.values():
                api.DestroyIcon(icon)
            self._icons.clear()
            self._hwnd = None
            api.UnregisterClassW(self._class_name, hinst)

    def _wndproc(self, hwnd, msg, wparam, lparam):
        api = _load_api()
        try:
            if msg == self.WM_TRAY:
                if lparam == WM_LBUTTONUP and self._on_open is not None:
                    self._on_open()                          # left-click: open the screen
                elif lparam in (WM_RBUTTONUP, WM_LBUTTONUP, WM_CONTEXTMENU):
                    self._show_menu()
                return 0
            if msg == self.WM_REFRESH:
                self._apply()
                return 0
            if msg == WM_WTSSESSION_CHANGE:
                if wparam in (WTS_SESSION_LOCK, WTS_SESSION_UNLOCK) and self._on_lock is not None:
                    self._on_lock(wparam == WTS_SESSION_LOCK)
                return 0
            if msg == WM_POWERBROADCAST:
                if self._on_sleep is not None:
                    if wparam == PBT_APMSUSPEND:
                        self._on_sleep(True)
                    elif wparam in (PBT_APMRESUMESUSPEND, PBT_APMRESUMEAUTOMATIC):
                        self._on_sleep(False)
                return 1
            if msg == getattr(self, "_taskbar_created", None) and msg:
                self._icon_added = False                     # Explorer restarted: put the icon back
                if self._show_icon:
                    self._add_icon()
                return 0
            if msg == WM_CLOSE:
                api.DestroyWindow(hwnd)
                return 0
            if msg == WM_DESTROY:
                self._remove_icon()
                api.WTSUnRegisterSessionNotification(hwnd)
                api.PostQuitMessage(0)
                return 0
        except Exception as exc:                             # never let an error escape into Windows
            log.error("tray event failed: %s", type(exc).__name__)
        return api.DefWindowProcW(hwnd, msg, wparam, lparam)

    def _icon_for(self, key: str):
        if key not in self._icons:
            api = _load_api()
            size = api.GetSystemMetrics(SM_CXSMICON) or 16
            data = icon_image(size, STATUS_LOOK.get(key, (GREY, ""))[0])
            self._icons[key] = api.CreateIconFromResourceEx(data, len(data), True, 0x00030000, size, size, 0)
        return self._icons[key]

    def _data(self, flags: int):
        api = _load_api()
        nid = api.NOTIFYICONDATAW()
        nid.cbSize = ctypes.sizeof(api.NOTIFYICONDATAW)
        nid.hWnd, nid.uID, nid.uFlags = self._hwnd, 1, flags
        return nid

    def _add_icon(self) -> None:
        api = _load_api()
        with self._lock:
            key, problem = self._status
        nid = self._data(NIF_MESSAGE | NIF_ICON | NIF_TIP)
        nid.uCallbackMessage = self.WM_TRAY
        nid.hIcon = self._icon_for(key)
        nid.szTip = tooltip_text(key, problem)
        self._icon_added = bool(api.Shell_NotifyIconW(NIM_ADD, ctypes.byref(nid)))

    def _remove_icon(self) -> None:
        if self._icon_added:
            _load_api().Shell_NotifyIconW(NIM_DELETE, ctypes.byref(self._data(0)))
            self._icon_added = False

    def _apply(self) -> None:
        api = _load_api()
        with self._lock:
            key, problem = self._status
            notes = list(self._notes)
            self._notes.clear()
        if not self._icon_added:
            return
        nid = self._data(NIF_ICON | NIF_TIP)
        nid.hIcon = self._icon_for(key)
        nid.szTip = tooltip_text(key, problem)
        api.Shell_NotifyIconW(NIM_MODIFY, ctypes.byref(nid))
        for title, text in notes:
            nid = self._data(NIF_INFO)
            nid.szInfoTitle = title[:63]
            nid.szInfo = text if len(text) <= 255 else text[:252] + "..."
            nid.dwInfoFlags = NIIF_INFO
            api.Shell_NotifyIconW(NIM_MODIFY, ctypes.byref(nid))

    def menu_items(self) -> list[tuple[int, str]]:
        """The menu, top to bottom, as (command, label). 0 is a separator."""
        paused = bool(self._is_paused()) if self._is_paused is not None else False
        items = [(self.CMD_OPEN, "Open J.A.R.V.I.S")] if self._on_open is not None else []
        items += [(self.CMD_PAUSE, "Resume listening" if paused else "Pause listening"), (0, ""),
                  (self.CMD_QUIT, "Quit J.A.R.V.I.S")]
        return items

    def menu_labels(self) -> list[str]:
        return [label for cmd, label in self.menu_items() if cmd]

    def _show_menu(self) -> None:
        api = _load_api()
        menu = api.CreatePopupMenu()
        try:
            for cmd, label in self.menu_items():
                if cmd:
                    api.AppendMenuW(menu, MF_STRING, cmd, label)
                else:
                    api.AppendMenuW(menu, MF_SEPARATOR, 0, None)
            pt = api.POINT()
            api.GetCursorPos(ctypes.byref(pt))
            api.SetForegroundWindow(self._hwnd)               # otherwise the menu won't close properly
            cmd = api.TrackPopupMenu(menu, TPM_RIGHTBUTTON | TPM_RETURNCMD | TPM_NONOTIFY,
                                     pt.x, pt.y, 0, self._hwnd, None)
            api.PostMessageW(self._hwnd, WM_NULL, 0, 0)
        finally:
            api.DestroyMenu(menu)
        self.handle_command(cmd)

    def handle_command(self, cmd: int) -> None:
        if cmd == self.CMD_OPEN and self._on_open is not None:
            self._on_open()
        elif cmd == self.CMD_PAUSE and self._on_toggle_pause is not None:
            self._on_toggle_pause()
        elif cmd == self.CMD_QUIT:
            if self._on_quit is not None:
                self._on_quit()
            self.quit()
