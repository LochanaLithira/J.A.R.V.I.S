"""Append-only, hash-chained audit log (JSON Lines).

Each entry stores the SHA-256 of the previous entry, so edits, deletions or
reordering of lines are detectable with verify(). This is tamper-EVIDENT, not
tamper-PROOF: code running as the same OS user could rewrite the whole file.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Any

GENESIS = "0" * 64
MAX_FIELD_CHARS = 200


def _clip(value: Any) -> Any:
    """Shorten long values so the log stays small and never stores whole files."""
    if isinstance(value, str):
        if len(value) > MAX_FIELD_CHARS:
            return value[:MAX_FIELD_CHARS] + f"...(+{len(value) - MAX_FIELD_CHARS} chars)"
        return value
    if isinstance(value, dict):
        return {str(k)[:MAX_FIELD_CHARS]: _clip(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clip(v) for v in value[:20]]
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return f"<{type(value).__name__}>"


def _digest(entry: dict[str, Any]) -> str:
    body = json.dumps(entry, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


class AuditLog:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._last = self._read_last_hash()

    def _read_last_hash(self) -> str:
        if not self.path.exists():
            return GENESIS
        last = GENESIS
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        last = json.loads(line).get("hash", last)
                    except json.JSONDecodeError:
                        pass
        return last

    def record(self, event: str, **fields: Any) -> None:
        with self._lock:
            entry = {"ts": round(time.time(), 3), "event": event}
            entry.update({k: _clip(v) for k, v in fields.items()})
            entry["prev"] = self._last
            h = _digest(entry)
            line = json.dumps({**entry, "hash": h}, sort_keys=True, separators=(",", ":"))
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
                f.flush()
                os.fsync(f.fileno())
            self._last = h

    def verify(self) -> tuple[bool, int | None]:
        """Return (True, None) if the chain is intact, else (False, bad_line_number)."""
        prev = GENESIS
        if not self.path.exists():
            return True, None
        with open(self.path, "r", encoding="utf-8") as f:
            for n, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    claimed = entry.pop("hash")
                except (json.JSONDecodeError, KeyError, AttributeError):
                    return False, n
                if entry.get("prev") != prev or _digest(entry) != claimed:
                    return False, n
                prev = claimed
        return True, None