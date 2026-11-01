"""File tools. Every action stays inside a short list of named folders.

Rules this file enforces:
- You name a folder from a fixed list ("documents", "downloads", ...) and a file NAME. Never a
  path: names cannot contain slashes, colons or other path syntax.
- Only files directly inside that folder are reachable. Hidden files, system files and links or
  junctions (which could point anywhere) are invisible, and the real resolved location must still
  be inside the folder.
- Opening is limited to documents, pictures, audio and video. Programs, scripts and shortcuts are
  never opened, whatever they are called.
- Nothing is ever overwritten. Renaming keeps the file's type (a .txt can never become a .exe).
- "Delete" moves the file into the J.A.R.V.I.S trash folder in the workspace, so it can always be
  brought back, and it needs your keypress first.
- Results contain file names, so they are private: shown on screen, never sent to the voice service.
- Searching is capped (depth, number of entries looked at, number of results), so it stays quick.
"""
from __future__ import annotations

import os
import re
import shutil
import time
from pathlib import Path
from typing import Callable, Iterable, Mapping

from ..gate.models import Param, Risk, Tool, ToolError
from ..gate.validate import RESERVED_NAMES

FILE_NAME_PATTERN = r'[^\x00-\x1f<>:"/\\|?*]{1,120}'
_NAME_RE = re.compile(FILE_NAME_PATTERN)

MAX_LIST = 30
MAX_FOUND = 15
MAX_SEARCH_DEPTH = 3
MAX_SEARCH_ENTRIES = 5000

OPENABLE = {
    ".txt", ".md", ".rtf", ".pdf", ".csv", ".doc", ".docx", ".odt", ".xls", ".xlsx", ".ods",
    ".ppt", ".pptx", ".odp", ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp",
    ".mp3", ".wav", ".m4a", ".flac", ".mp4", ".mkv", ".avi", ".mov", ".webm",
}

_HIDDEN_ATTRS = 0x2 | 0x4 | 0x400            # hidden, system, reparse point (links and junctions)
_SKIP_NAMES = {"desktop.ini", "thumbs.db"}

# Windows' own names for the standard folders (they may be moved, for example into OneDrive).
_KNOWN_FOLDER_IDS = {
    "documents": "FDD39AD0-238F-46AF-ADB4-6C85480369C7",
    "downloads": "374DE290-123F-4565-9164-39C4925E467B",
    "desktop": "B4BFCC3A-DB2C-424C-B029-7FE99A87C641",
    "pictures": "33E28130-4E1E-4676-835A-98395C3BC3BB",
    "music": "4BD8D571-6D19-48D3-BE97-422220080E43",
    "videos": "18989B1D-99B5-455B-841C-AB7C74E4DDFC",
}


def known_folder(name: str) -> Path | None:
    """Where Windows really keeps a standard folder, or None if it can't be found."""
    folder_id = _KNOWN_FOLDER_IDS.get(name)
    if folder_id is None:
        return None
    try:
        import ctypes
        import uuid
        from ctypes import wintypes

        class GUID(ctypes.Structure):
            _fields_ = [("Data1", ctypes.c_uint32), ("Data2", ctypes.c_uint16), ("Data3", ctypes.c_uint16),
                        ("Data4", ctypes.c_ubyte * 8)]

        u = uuid.UUID(folder_id)
        guid = GUID(u.fields[0], u.fields[1], u.fields[2], (ctypes.c_ubyte * 8)(*u.bytes[8:]))
        shell32, ole32 = ctypes.WinDLL("shell32"), ctypes.WinDLL("ole32")
        shell32.SHGetKnownFolderPath.argtypes = [ctypes.POINTER(GUID), wintypes.DWORD, wintypes.HANDLE,
                                                 ctypes.POINTER(ctypes.c_wchar_p)]
        shell32.SHGetKnownFolderPath.restype = ctypes.c_long
        out = ctypes.c_wchar_p()
        if shell32.SHGetKnownFolderPath(ctypes.byref(guid), 0, None, ctypes.byref(out)) != 0:
            return None
        try:
            return Path(out.value)
        finally:
            ole32.CoTaskMemFree(out)
    except Exception:
        fallback = Path.home() / name.capitalize()
        return fallback if fallback.is_dir() else None


def file_folders(workspace: Path, names: Iterable[str]) -> dict[str, Path]:
    """The folders the file tools may use: the workspace plus the standard folders that exist."""
    folders: dict[str, Path] = {}
    for name in names:
        path = Path(workspace) if name == "workspace" else known_folder(name)
        if path is not None and path.is_dir():
            folders[name] = path
    return folders


# ---------------------------------------------------------------------------
# Safe lookups
# ---------------------------------------------------------------------------

def _hidden(entry: os.DirEntry) -> bool:
    try:
        attrs = getattr(entry.stat(follow_symlinks=False), "st_file_attributes", 0)
    except OSError:
        return True
    return bool(attrs & _HIDDEN_ATTRS) or entry.is_symlink() or entry.name.lower() in _SKIP_NAMES \
        or entry.name.startswith("~$")


def _check_name(name: str) -> str:
    """Refuse anything that is not a plain file name."""
    if not isinstance(name, str) or _NAME_RE.fullmatch(name) is None or name != name.strip() \
            or name.endswith(".") or set(name) <= {"."}:
        raise ToolError("That file name isn't allowed.")
    if name.split(".")[0].lower() in RESERVED_NAMES:
        raise ToolError("That file name isn't allowed.")
    return name


def _visible_files(folder: Path) -> list[os.DirEntry]:
    try:
        with os.scandir(folder) as it:
            return [e for e in it if not _hidden(e) and e.is_file(follow_symlinks=False)]
    except OSError:
        raise ToolError("I couldn't read that folder.") from None


def _inside(path: Path, folder: Path) -> Path:
    """The real location of `path`, which must be directly inside `folder`."""
    try:
        real = path.resolve(strict=True)
    except OSError:
        raise ToolError("That file isn't there any more.") from None
    if real.parent != folder.resolve() or path.is_symlink():
        raise ToolError("That file isn't allowed.")
    return real


def _find(folder: Path, folder_label: str, name: str) -> Path:
    """Match a spoken name to one file: the full name first, then the name without its ending."""
    _check_name(name)
    entries = _visible_files(folder)
    exact = [e for e in entries if e.name.lower() == name.lower()]
    matches = exact or [e for e in entries if Path(e.name).stem.lower() == name.lower()]
    if not matches:
        raise ToolError(f"There's no file called {name} in {folder_label}.")
    if len(matches) > 1:
        names = ", ".join(sorted(e.name for e in matches)[:5])
        raise ToolError(f"Several files are called that: {names}. Say the full name with its ending.")
    return _inside(Path(matches[0].path), folder)


def _free_target(folder: Path, name: str) -> Path:
    target = folder / name
    if os.path.lexists(target):
        raise ToolError(f"There's already a file called {name} there, and I never overwrite files.")
    return target


# ---------------------------------------------------------------------------
# The tools
# ---------------------------------------------------------------------------

def make_file_tools(folders: Mapping[str, Path], trash: Path,
                    opener: Callable[[str], None] | None = None) -> list[Tool]:
    folders = {k: Path(v).resolve() for k, v in folders.items()}
    trash = Path(trash)
    opener = opener or getattr(os, "startfile", None)
    if not folders:
        return []

    def folder_of(label: str) -> Path:
        path = folders.get(label)
        if path is None or not path.is_dir():
            raise ToolError("That folder isn't available.")
        return path

    def list_files(folder: str) -> str:
        entries = _visible_files(folder_of(folder))
        if not entries:
            return f"{folder.capitalize()} has no files."
        newest = sorted(entries, key=lambda e: e.stat(follow_symlinks=False).st_mtime, reverse=True)
        shown = ", ".join(e.name for e in newest[:MAX_LIST])
        more = f" (and {len(entries) - MAX_LIST} more)" if len(entries) > MAX_LIST else ""
        return f"{folder.capitalize()} has {len(entries)} files. Newest first: {shown}{more}."

    def find_files(name: str) -> str:
        if not isinstance(name, str) or len(name.strip()) < 2:
            raise ToolError("Tell me at least two letters of the name.")
        needle, found, looked = name.strip().lower(), [], 0
        skip = {trash.resolve()} if trash.exists() else set()
        for label, root in folders.items():
            stack = [(root, 0)]
            while stack and len(found) < MAX_FOUND and looked < MAX_SEARCH_ENTRIES:
                current, depth = stack.pop()
                try:
                    with os.scandir(current) as it:
                        entries = list(it)
                except OSError:
                    continue
                for e in entries:
                    looked += 1
                    if _hidden(e):
                        continue
                    if e.is_dir(follow_symlinks=False):
                        if depth + 1 < MAX_SEARCH_DEPTH and Path(e.path).resolve() not in skip:
                            stack.append((Path(e.path), depth + 1))
                    elif needle in e.name.lower():
                        where = Path(e.path).parent.relative_to(root)
                        found.append(f"{e.name} in {label}" + (f"\\{where}" if str(where) != "." else ""))
                        if len(found) >= MAX_FOUND:
                            break
        if not found:
            return f"I couldn't find a file with {name.strip()} in its name."
        return "Found: " + "; ".join(found) + "."

    def open_file(folder: str, name: str) -> str:
        path = _find(folder_of(folder), folder, name)
        if path.suffix.lower() not in OPENABLE:
            raise ToolError("I only open documents, pictures, audio and video, never programs or scripts.")
        if opener is None:
            raise ToolError("Opening files isn't available here.")
        try:
            opener(str(path))
        except OSError:
            raise ToolError("Windows couldn't open that file.") from None
        return f"Opening {path.name}."

    def open_folder(folder: str) -> str:
        path = folder_of(folder)
        if opener is None:
            raise ToolError("Opening folders isn't available here.")
        try:
            opener(str(path))
        except OSError:
            raise ToolError("Windows couldn't open that folder.") from None
        return f"Opening {folder}."

    def move_file(folder: str, name: str, to_folder: str) -> str:
        source_dir, target_dir = folder_of(folder), folder_of(to_folder)
        if source_dir == target_dir:
            raise ToolError("That file is already there.")
        path = _find(source_dir, folder, name)
        target = _free_target(target_dir, path.name)
        try:
            shutil.move(str(path), str(target))
        except OSError:
            raise ToolError("I couldn't move that file. It may be open in another program.") from None
        return f"Moved {path.name} to {to_folder}."

    def rename_file(folder: str, name: str, new_name: str) -> str:
        directory = folder_of(folder)
        path = _find(directory, folder, name)
        new_name = _check_name(new_name)
        if Path(new_name).suffix.lower() == "" and path.suffix:
            new_name += path.suffix                       # "budget" keeps the old ending
        if Path(new_name).suffix.lower() != path.suffix.lower():
            raise ToolError("I can't change what type of file it is, only its name.")
        target = _free_target(directory, new_name)
        try:
            path.rename(target)
        except OSError:
            raise ToolError("I couldn't rename that file. It may be open in another program.") from None
        return f"Renamed {path.name} to {new_name}."

    def delete_file(folder: str, name: str) -> str:
        path = _find(folder_of(folder), folder, name)
        trash.mkdir(parents=True, exist_ok=True)
        target = _free_target(trash, f"{time.strftime('%Y%m%d-%H%M%S')}-{path.name}")
        try:
            shutil.move(str(path), str(target))
        except OSError:
            raise ToolError("I couldn't move that file to the trash. It may be open in another program.") from None
        return f"Moved {path.name} to the J.A.R.V.I.S trash ({trash}). You can move it back from there."

    choice = tuple(sorted(folders))
    folder_p = Param(str, max_len=20, choices=choice)
    name_p = Param(str, max_len=120, pattern=FILE_NAME_PATTERN)
    return [
        Tool("list_files", "List the files in one of your folders, newest first.", Risk.LOW,
             {"folder": folder_p}, list_files, private_output=True),
        Tool("find_files", "Find files whose name contains some text, in all of your folders.", Risk.LOW,
             {"name": Param(str, max_len=60, pattern=FILE_NAME_PATTERN)}, find_files, private_output=True),
        Tool("open_file", "Open a document, picture, song or video from one of your folders.", Risk.MEDIUM,
             {"folder": folder_p, "name": name_p}, open_file, private_output=True),
        Tool("open_folder", "Open one of your folders in File Explorer.", Risk.LOW,
             {"folder": folder_p}, open_folder),
        Tool("move_file", "Move a file from one of your folders to another (never overwrites).", Risk.MEDIUM,
             {"folder": folder_p, "name": name_p, "to_folder": folder_p}, move_file, private_output=True),
        Tool("rename_file", "Rename a file in one of your folders, keeping its type.", Risk.MEDIUM,
             {"folder": folder_p, "name": name_p, "new_name": name_p}, rename_file, private_output=True),
        Tool("delete_file", "Move a file to the J.A.R.V.I.S trash (can be undone). Needs your keypress.",
             Risk.HIGH, {"folder": folder_p, "name": name_p}, delete_file, private_output=True),
    ]
