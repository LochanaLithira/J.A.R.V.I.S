"""Builds the gate with the approved tool set. Add tools here, and only here."""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

from . import config
from .gate.audit import AuditLog
from .gate.gate import ConfirmFn, Gate, deny_all
from .tools.apps import make_app_tools
from .tools.files import file_folders, make_file_tools
from .tools.notes import make_note_tools
from .tools.timers import TimerService, make_timer_tools


def build_gate(workspace: Path | None = None, log_path: Path | None = None,
               confirm: ConfirmFn = deny_all, timers: TimerService | None = None,
               folders: Mapping[str, Path] | None = None) -> Gate:
    """`timers` adds the timer tools (the app passes the service that rings them). `folders`
    overrides which folders the file tools may use (normally the list in config.py)."""
    workspace = Path(workspace or config.WORKSPACE)
    workspace.mkdir(parents=True, exist_ok=True)
    if folders is None:
        folders = file_folders(workspace, config.FILE_FOLDERS)

    tools = [*make_note_tools(workspace), *make_app_tools(config.ALLOWED_APPS),
             *make_file_tools(folders, trash=workspace / "trash")]
    if timers is not None:
        tools += make_timer_tools(timers)

    # A duplicate name would silently replace a tool, so refuse to start instead.
    names = [t.name for t in tools]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate tool name in registry.")

    return Gate(tools, AuditLog(log_path or config.AUDIT_LOG), confirm,
                config.RATE_LIMIT_CALLS, config.RATE_LIMIT_WINDOW_SECONDS)
