"""Builds the gate with the approved tool set. Add tools here, and only here."""
from __future__ import annotations

from pathlib import Path

from . import config
from .gate.audit import AuditLog
from .gate.gate import ConfirmFn, Gate, deny_all
from .tools.apps import make_app_tools
from .tools.notes import make_note_tools


def build_gate(workspace: Path | None = None, log_path: Path | None = None,
               confirm: ConfirmFn = deny_all) -> Gate:
    workspace = Path(workspace or config.WORKSPACE)
    workspace.mkdir(parents=True, exist_ok=True)

    tools = [*make_note_tools(workspace), *make_app_tools(config.ALLOWED_APPS)]

    # A duplicate name would silently replace a tool, so refuse to start instead.
    names = [t.name for t in tools]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate tool name in registry.")

    return Gate(tools, AuditLog(log_path or config.AUDIT_LOG), confirm,
                config.RATE_LIMIT_CALLS, config.RATE_LIMIT_WINDOW_SECONDS)