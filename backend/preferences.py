"""Choices made on the screen that are not secrets: which microphone, and whether setup is done.

Kept in %LOCALAPPDATA%\\jarvis\\preferences.json, outside the workspace, so no tool can reach it.
Nothing secret is ever written here (keys stay in the Credential Manager). The file is checked
when read: anything unexpected is ignored and the defaults from config.py are used instead, so
a damaged or edited file can only ever choose a microphone or skip the setup guide.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from . import config

log = logging.getLogger("jarvis")

MAX_FILE_BYTES = 4096
MAX_MIC_CHARS = 200
_UNSET = object()


def _clean_mic(value: Any) -> Any:
    """A microphone name ("" = the Windows default), or _UNSET if the value is not usable."""
    if isinstance(value, str) and len(value) <= MAX_MIC_CHARS and value.isprintable():
        return value
    return _UNSET


def load(path: Path | None = None) -> dict:
    """{"mic": name or "" (only if chosen), "setup_done": bool}. Never raises."""
    path = Path(path or config.PREFERENCES)
    out: dict = {"setup_done": False}
    try:
        if not path.is_file() or path.stat().st_size > MAX_FILE_BYTES:
            return out
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        log.warning("preferences file unreadable, using defaults")
        return out
    if not isinstance(raw, dict):
        return out
    if "mic" in raw and _clean_mic(raw["mic"]) is not _UNSET:
        out["mic"] = raw["mic"]
    out["setup_done"] = raw.get("setup_done") is True
    return out


def save(prefs: dict, path: Path | None = None) -> None:
    """Write the preferences all at once (never a half-written file)."""
    path = Path(path or config.PREFERENCES)
    data: dict = {"setup_done": prefs.get("setup_done") is True}
    if "mic" in prefs:
        if _clean_mic(prefs["mic"]) is _UNSET:
            raise ValueError("That microphone name can't be saved.")
        data["mic"] = prefs["mic"]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def update(path: Path | None = None, **changes: Any) -> dict:
    prefs = load(path)
    prefs.update(changes)
    save(prefs, path)
    return prefs


def microphone(path: Path | None = None) -> str | None:
    """The microphone to listen with: the one picked on the screen, else MIC_DEVICE in config.py.
    None means the Windows default."""
    prefs = load(path)
    if "mic" in prefs:
        return prefs["mic"] or None
    return config.MIC_DEVICE
