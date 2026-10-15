"""Central settings. Edit paths for your machine."""
from __future__ import annotations

import os
from pathlib import Path

# The only folder the agent is allowed to write to.
WORKSPACE = Path.home() / "jarvis_workspace"

# The audit log lives OUTSIDE the workspace, so no tool can ever reach it.
_base = Path(os.environ.get("LOCALAPPDATA", str(Path.home())))
AUDIT_LOG = _base / "jarvis" / "audit.jsonl"

# Short name -> full path of the program. Add apps one at a time, with full paths.
ALLOWED_APPS = {
    "notepad": r"C:\Windows\System32\notepad.exe",
    "calculator": r"C:\Windows\System32\calc.exe",
}

# At most this many proposals per time window, enforced by the gate.
RATE_LIMIT_CALLS = 10
RATE_LIMIT_WINDOW_SECONDS = 60.0