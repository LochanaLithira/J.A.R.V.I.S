"""PC controls: sound, media keys, screen brightness, battery, CPU, and locking the PC.

- Only Windows' own interfaces are used (Core Audio, the media keys, WMI, kernel32 and
  LockWorkStation). No program is started and nothing is written to disk.
- Nothing here can lose your work, so every tool is LOW risk.
- The tools talk to a small `pc` object, so the tests use a fake one and never touch your real
  volume or screen.
"""
from __future__ import annotations

import ctypes
import time
from ctypes import wintypes

from ..gate.models import Param, Risk, Tool, ToolError

STEP = 10                   # "volume up" / "brighter" move this many percent
MIN_BRIGHTNESS = 5          # never fully black: you could not see the screen to fix it

MEDIA_KEYS = {"play_pause": 0xB3, "next": 0xB0, "previous": 0xB1, "stop": 0xB2}
_KEYEVENTF_EXTENDEDKEY, _KEYEVENTF_KEYUP = 0x1, 0x2


class _PowerStatus(ctypes.Structure):
    _fields_ = [("ac_line", ctypes.c_ubyte), ("flag", ctypes.c_ubyte), ("percent", ctypes.c_ubyte),
                ("saver", ctypes.c_ubyte), ("life", wintypes.DWORD), ("full_life", wintypes.DWORD)]


def _ticks(t: wintypes.FILETIME) -> int:
    return (t.dwHighDateTime << 32) | t.dwLowDateTime


class WindowsPC:
    """The real controls. Every method is one small Windows call."""

    # ---- sound ----------------------------------------------------------------
    def volume(self) -> int:
        from .wincom import speakers
        return round(speakers().GetMasterVolumeLevelScalar() * 100)

    def set_volume(self, percent: int) -> None:
        from .wincom import speakers
        speakers().SetMasterVolumeLevelScalar(percent / 100, None)

    def muted(self) -> bool:
        from .wincom import speakers
        return bool(speakers().GetMute())

    def set_mute(self, mute: bool) -> None:
        from .wincom import speakers
        speakers().SetMute(mute, None)

    def press(self, key: str) -> None:
        vk = MEDIA_KEYS[key]
        user32 = ctypes.windll.user32
        user32.keybd_event(vk, 0, _KEYEVENTF_EXTENDEDKEY, 0)
        user32.keybd_event(vk, 0, _KEYEVENTF_EXTENDEDKEY | _KEYEVENTF_KEYUP, 0)

    # ---- screen ---------------------------------------------------------------
    def brightness(self) -> int | None:
        """None when the screen can't be dimmed by Windows (most external monitors)."""
        # WMI's names are not in a type library, so they are reached through Properties_ and
        # ExecMethod_ rather than as plain attributes.
        from .wincom import wmi
        try:
            for screen in wmi().ExecQuery("SELECT CurrentBrightness FROM WmiMonitorBrightness WHERE Active=TRUE"):
                return int(screen.Properties_.Item("CurrentBrightness").Value)
        except Exception:
            return None
        return None

    def set_brightness(self, percent: int) -> None:
        from .wincom import wmi
        try:
            for screen in wmi().ExecQuery("SELECT * FROM WmiMonitorBrightnessMethods WHERE Active=TRUE"):
                args = screen.Methods_.Item("WmiSetBrightness").InParameters.SpawnInstance_()
                args.Properties_.Item("Timeout").Value = 1
                args.Properties_.Item("Brightness").Value = percent
                screen.ExecMethod_("WmiSetBrightness", args)
                return
        except Exception:
            pass
        raise ToolError("I can't change the brightness of this screen.")

    # ---- status ---------------------------------------------------------------
    def battery(self) -> tuple[int | None, bool]:
        """(percent or None if there is no battery, plugged in)."""
        s = _PowerStatus()
        if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(s)):
            return None, True
        no_battery = s.flag == 128 or s.percent == 255
        return (None if no_battery else s.percent), s.ac_line == 1

    def cpu(self) -> int:
        def sample():
            idle, kernel, user = wintypes.FILETIME(), wintypes.FILETIME(), wintypes.FILETIME()
            ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user))
            return _ticks(idle), _ticks(kernel) + _ticks(user)     # kernel time includes idle time

        idle1, total1 = sample()
        time.sleep(0.3)
        idle2, total2 = sample()
        total = total2 - total1
        return 0 if total <= 0 else round(100 * (total - (idle2 - idle1)) / total)

    # ---- lock -----------------------------------------------------------------
    def lock(self) -> None:
        if not ctypes.windll.user32.LockWorkStation():
            raise ToolError("I couldn't lock the PC.")


def _clamp(value: int, low: int = 0, high: int = 100) -> int:
    return max(low, min(high, value))


def make_pc_tools(pc=None) -> list[Tool]:
    pc = pc or WindowsPC()

    def set_volume(percent: int) -> str:
        pc.set_volume(percent)
        if percent and pc.muted():
            pc.set_mute(False)
        return f"Volume at {percent} percent."

    def volume(action: str) -> str:
        if action == "mute":
            pc.set_mute(True)
            return "Muted."
        if action == "unmute":
            pc.set_mute(False)
            return "Sound is back on."
        level = _clamp(pc.volume() + (STEP if action == "up" else -STEP))
        pc.set_volume(level)
        if action == "up":
            pc.set_mute(False)          # like the keyboard's volume key: turning it up unmutes
        return f"Volume {action} to {level} percent."

    def media(action: str) -> str:
        pc.press(action)
        return {"play_pause": "Done.", "next": "Next track.", "previous": "Previous track.",
                "stop": "Stopped."}[action]

    def set_brightness(percent: int) -> str:
        level = _clamp(percent, MIN_BRIGHTNESS)
        pc.set_brightness(level)
        return f"Brightness at {level} percent."

    def brightness(action: str) -> str:
        now = pc.brightness()
        if now is None:
            raise ToolError("I can't change the brightness of this screen.")
        level = _clamp(now + (STEP if action == "up" else -STEP), MIN_BRIGHTNESS)
        pc.set_brightness(level)
        return f"Brightness {action} to {level} percent."

    def pc_status(what: str) -> str:
        parts = []
        if what in ("battery", "all"):
            percent, plugged = pc.battery()
            if percent is None:
                parts.append("There's no battery; you're on mains power.")
            else:
                parts.append(f"Battery at {percent} percent, " + ("plugged in." if plugged else "on battery."))
        if what in ("cpu", "all"):
            parts.append(f"The processor is at {pc.cpu()} percent.")
        if what in ("volume", "all"):
            parts.append("Sound is muted." if pc.muted() else f"Volume is at {pc.volume()} percent.")
        if what in ("brightness", "all"):
            level = pc.brightness()
            if level is not None:
                parts.append(f"Brightness is at {level} percent.")
            elif what == "brightness":
                parts.append("I can't read this screen's brightness.")
        return " ".join(parts)

    def lock_pc() -> str:
        pc.lock()
        return "Locking the PC."

    percent = Param(int, min_value=0, max_value=100)
    up_down = Param(str, max_len=8, choices=("up", "down"))
    return [
        Tool("set_volume", "Set the speaker volume to a percentage, 0 to 100.", Risk.LOW,
             {"percent": percent}, set_volume),
        Tool("volume", "Turn the volume up or down a step, or mute or unmute the sound.", Risk.LOW,
             {"action": Param(str, max_len=8, choices=("up", "down", "mute", "unmute"))}, volume),
        Tool("media", "Press a media key: play_pause (play or pause the music or video), next, previous "
             "or stop.", Risk.LOW,
             {"action": Param(str, max_len=10, choices=tuple(MEDIA_KEYS))}, media),
        Tool("set_brightness", "Set the screen brightness to a percentage, 0 to 100.", Risk.LOW,
             {"percent": percent}, set_brightness),
        Tool("brightness", "Make the screen brighter (up) or dimmer (down) by a step.", Risk.LOW,
             {"action": up_down}, brightness),
        Tool("pc_status", "Tell the battery level, CPU load, volume or screen brightness, or all of them.",
             Risk.LOW, {"what": Param(str, max_len=10, choices=("battery", "cpu", "volume", "brightness", "all"))},
             pc_status),
        Tool("lock_pc", "Lock the PC (like pressing Windows+L).", Risk.LOW, {}, lock_pc),
    ]
