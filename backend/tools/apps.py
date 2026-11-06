"""Open any installed app by name, the way the Start menu does, except the dangerous ones.

- The list comes from Windows' own Apps folder (shell:AppsFolder): everything in the Start menu,
  Store apps included. It is read when first needed and refreshed every 10 minutes.
- The AI only passes the name you said. The tool picks the entry from that list and opens it by
  its Windows ID: no path, argument or command from the AI is ever used.
- Deny by default. An entry is offered only if it is a Store app, or if the real program behind
  it (not just its label) is a windowed .exe that is not on the blocked list below. So command
  lines, PowerShell, Terminal, script hosts, interpreters, the Registry Editor, admin tools and
  uninstallers can never be opened, even through a renamed shortcut.
- Documents, web links and control panels in the Start menu are not apps and are skipped.
"""
from __future__ import annotations

import ctypes
import re
import struct
import time
from dataclasses import dataclass
from pathlib import PureWindowsPath
from typing import Callable, Iterable

from ..gate.models import Param, Risk, Tool, ToolError

# Programs that run commands, scripts or code, change the system, or remove software.
BLOCKED_PROGRAMS = frozenset({
    # command lines and shells
    "cmd", "powershell", "powershell_ise", "pwsh", "wt", "windowsterminal", "openconsole", "conhost",
    "bash", "sh", "wsl", "wslhost", "wslg", "ubuntu", "debian", "kali", "git-bash", "git-cmd", "mintty",
    # script hosts and other programs attackers misuse to run code
    "wscript", "cscript", "mshta", "rundll32", "regsvr32", "msiexec", "certutil", "bitsadmin",
    "installutil", "msbuild", "regasm", "regsvcs", "cmstp", "forfiles", "pcalua", "scriptrunner",
    # interpreters
    "python", "pythonw", "py", "pyw", "node", "java", "javaw", "ruby", "perl", "php",
    # system and admin tools
    "regedit", "regedt32", "reg", "mmc", "taskmgr", "msconfig", "perfmon", "resmon", "eventvwr",
    "control", "schtasks", "sc", "netsh", "gpedit", "secpol", "lusrmgr", "compmgmt", "diskmgmt",
    "devmgmt", "services", "useraccountcontrolsettings", "systempropertiesadvanced", "iexpress",
    "odbcad32", "recoverydrive", "dism", "bcdedit", "vssadmin", "wmic", "takeown", "icacls",
    "dfrgui", "cleanmgr", "mdsched", "iscsicpl", "stackbuilder",
    # letting someone else control the PC (a favourite of phone scammers)
    "quickassist", "msra",
})
# Store apps that are command lines, Linux shells or remote control, by package family.
BLOCKED_PACKAGES = ("microsoft.windowsterminal", "microsoft.powershell", "canonicalgrouplimited.",
                    "thedebianproject.", "kalilinux.", "microsoftcorporationii.windowssubsystemforlinux",
                    "microsoft.windbg", "microsoftcorporationii.quickassist")
# Words that mark an entry as dangerous whatever program it points at.
BLOCKED_WORDS = ("uninstall", "command prompt", "powershell", "terminal", "registry", "bash", "shell")
# The one shell entry that is an ordinary app rather than a file or a control panel.
SHELL_APPS = frozenset({"microsoft.windows.explorer"})

REFRESH_SECONDS = 600
NAME_PATTERN = r"[A-Za-z0-9][A-Za-z0-9 .+&'()\-]{0,63}"
_GUI_SUBSYSTEM = 2


@dataclass(frozen=True)
class AppEntry:
    """One entry of the Apps folder, as Windows describes it."""
    name: str
    launch_id: str              # what follows shell:AppsFolder\ : a Store app ID, an app ID, or a path
    target: str | None = None   # the program behind a shortcut, when Windows knows it


def is_windowed_exe(path: str) -> bool:
    """True only for a real .exe that opens a window, not a console (command-line) program."""
    try:
        with open(path, "rb") as f:
            head = f.read(4096)
        if head[:2] != b"MZ":
            return False
        pe = struct.unpack_from("<I", head, 0x3C)[0]
        if head[pe:pe + 4] != b"PE\0\0":
            return False
        return struct.unpack_from("<H", head, pe + 24 + 68)[0] == _GUI_SUBSYSTEM
    except (OSError, struct.error):
        return False


def _program(path: str) -> str:
    return PureWindowsPath(path).stem.lower()


def is_allowed(entry: AppEntry, windowed: Callable[[str], bool] = is_windowed_exe) -> bool:
    """Deny by default: True only for a Store app or a windowed .exe that is not blocked."""
    name, launch_id = entry.name.lower(), entry.launch_id
    if any(word in name for word in BLOCKED_WORDS):
        return False
    if "://" in launch_id:
        return False                                            # a web link or a file:// document
    if "!" in launch_id and "\\" not in launch_id:              # a Store app: Family_hash!App
        return not launch_id.lower().startswith(BLOCKED_PACKAGES)
    if launch_id.lower() in SHELL_APPS:
        return True
    target = entry.target or (launch_id if "\\" in launch_id else None)
    if not target or target.startswith("::") or PureWindowsPath(target).suffix.lower() != ".exe":
        return False                                            # documents, control panels, unknown
    program = _program(target)
    if program in BLOCKED_PROGRAMS or program.startswith(("unins", "setup", "install")):
        return False
    return windowed(target)


def simplify(text: str) -> str:
    """'Microsoft Word (classic)' -> 'microsoft word classic'."""
    return " ".join(re.sub(r"[^a-z0-9+]+", " ", text.lower()).split())


def _spoken(text: str) -> str:
    s = simplify(text)
    s = re.sub(r"^(the|my) ", "", s)
    return re.sub(r" (app|application|program)$", "", s)


def match_level(spoken: str, label: str) -> int:
    """2: the same name, 1: every word you said is in the name, 0: no match."""
    s, l = _spoken(spoken), simplify(label)
    if not s or not l:
        return 0
    if s == l:
        return 2
    return 1 if set(s.split()) <= set(l.split()) else 0


def best_matches(spoken: str, labelled: Iterable[tuple[str, object]]) -> list[tuple[str, object]]:
    """The (label, item) pairs that match best; an exact name beats a partial one."""
    scored = [(match_level(spoken, label), label, item) for label, item in labelled]
    top = max((score for score, _, _ in scored), default=0)
    return [] if top == 0 else [(label, item) for score, label, item in scored if score == top]


def list_start_menu() -> list[AppEntry]:
    from .wincom import shell

    items = shell().NameSpace("shell:AppsFolder").Items()
    entries = []
    for i in range(items.Count):
        item = items.Item(i)
        try:
            target = item.ExtendedProperty("System.Link.TargetParsingPath")
        except Exception:
            target = None
        entries.append(AppEntry(str(item.Name), str(item.Path), str(target) if target else None))
    return entries


class AppCatalog:
    """The safe part of the Start menu, cached for a few minutes."""

    def __init__(self, lister: Callable[[], list[AppEntry]] = list_start_menu,
                 allowed: Callable[[AppEntry], bool] = is_allowed,
                 clock: Callable[[], float] = time.monotonic, refresh_seconds: float = REFRESH_SECONDS):
        self._lister, self._allowed, self._clock = lister, allowed, clock
        self._refresh = refresh_seconds
        self._all: list[AppEntry] = []
        self._loaded_at: float | None = None

    def _entries(self) -> list[AppEntry]:
        now = self._clock()
        if self._loaded_at is None or now - self._loaded_at > self._refresh:
            try:
                self._all = list(self._lister())
            except Exception:
                if self._loaded_at is None:
                    raise ToolError("I couldn't read the list of apps.")
            self._loaded_at = now
        return self._all

    def apps(self) -> list[AppEntry]:
        return [e for e in self._entries() if self._allowed(e)]

    def names_for_program(self, path: str) -> list[str]:
        """Start menu names of the program at path, e.g. WINWORD.EXE -> ['Word']. Any entry, blocked
        or not: used only to recognise windows that are already open."""
        program = _program(path)
        return [e.name for e in self._entries() if e.target and _program(e.target) == program]

    def find(self, spoken: str) -> AppEntry:
        matches = best_matches(spoken, ((e.name, e) for e in self.apps()))
        unique = list({e.launch_id: (label, e) for label, e in matches}.values())
        if not unique:
            raise ToolError("I couldn't find an app by that name that I'm allowed to open.")
        if len(unique) > 1:
            names = sorted({label for label, _ in unique})[:3]
            raise ToolError("Did you mean " + " or ".join(names) + "?")
        return unique[0][1]


def shell_open(launch_id: str) -> None:
    """Open an Apps folder entry exactly as the Start menu would (no extra arguments)."""
    result = ctypes.windll.shell32.ShellExecuteW(None, "open", "shell:AppsFolder\\" + launch_id, None, None, 1)
    if result <= 32:
        raise OSError("ShellExecute failed")


def make_app_tools(catalog: AppCatalog | None = None, launcher: Callable[[str], None] = shell_open) -> list[Tool]:
    catalog = catalog or AppCatalog()

    def open_app(name: str) -> str:
        app = catalog.find(name)
        try:
            launcher(app.launch_id)
        except OSError:
            raise ToolError("I couldn't open that app.")
        return f"Opening {app.name}."

    return [
        Tool("open_app", "Open an installed app by the name the user said, for example 'spotify', 'word' or "
             "'calculator'. Command lines, PowerShell, Terminal, the Registry Editor and admin tools are "
             "never opened.", Risk.LOW, {"name": Param(str, max_len=64, pattern=NAME_PATTERN)}, open_app),
    ]
