"""The tool gate: the ONLY path from an LLM proposal to code that runs.

The LLM proposes (tool_name, args). The gate decides. Deny by default: unknown
tool, bad arguments, rate limit, missing confirmation, or any error => refuse.
"""
from __future__ import annotations

import time
from collections import deque
from typing import Any, Callable, Iterable

from .audit import AuditLog
from .models import Result, Risk, Tool, ToolError, ValidationError
from .validate import validate_args

ConfirmFn = Callable[[str, dict[str, Any]], bool]


def deny_all(tool_name: str, args: dict[str, Any]) -> bool:
    """Default confirmation: refuse. The real keypress popup is added in a later feature."""
    return False


class Gate:
    def __init__(
        self,
        tools: Iterable[Tool],
        audit: AuditLog,
        confirm: ConfirmFn = deny_all,
        max_calls: int = 10,
        window_seconds: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._tools = {t.name: t for t in tools}
        self._audit = audit
        self._confirm = confirm
        self._max_calls = max_calls
        self._window = window_seconds
        self._clock = clock
        self._calls: deque[float] = deque()

    @property
    def audit(self) -> AuditLog:
        """The one audit log. Everything that records events must share this object,
        because two writers on the same file would break the tamper-evident chain."""
        return self._audit

    def tool_specs(self) -> list[dict[str, Any]]:
        """Descriptions of the tools that are safe to show the LLM. Never exposes the functions."""
        specs = []
        for t in self._tools.values():
            specs.append({
                "name": t.name,
                "description": t.description,
                "risk": t.risk.value,
                "params": {
                    k: {
                        "type": p.kind.__name__,
                        "max_len": p.max_len if p.kind is str else None,
                        "choices": list(p.choices) if p.choices else None,
                    }
                    for k, p in t.params.items()
                },
            })
        return specs

    def _rate_limited(self) -> bool:
        now = self._clock()
        while self._calls and now - self._calls[0] > self._window:
            self._calls.popleft()
        if len(self._calls) >= self._max_calls:
            return True
        self._calls.append(now)
        return False

    def _deny(self, tool_name: Any, reason: str, message: str) -> Result:
        self._audit.record("denied", tool=tool_name, reason=reason)
        return Result(False, message)

    def execute(self, tool_name: Any, args: Any) -> Result:
        self._audit.record("proposed", tool=tool_name, args=args)

        if self._rate_limited():
            return self._deny(tool_name, "rate_limit", "Too many requests. Please wait a moment.")

        tool = self._tools.get(tool_name) if isinstance(tool_name, str) else None
        if tool is None:
            return self._deny(tool_name, "unknown_tool", "That action is not allowed.")

        try:
            clean = validate_args(tool.params, args)
        except ValidationError as exc:
            return self._deny(tool_name, f"invalid_args: {exc}", "I couldn't use those details.")

        if tool.risk is Risk.HIGH:
            try:
                approved = self._confirm(tool.name, clean) is True
            except Exception:
                approved = False
            self._audit.record("confirmation", tool=tool.name, approved=approved)
            if not approved:
                return self._deny(tool.name, "not_confirmed", "Cancelled.")

        try:
            output = tool.fn(**clean)
        except ToolError as exc:
            self._audit.record("failed", tool=tool.name, reason=str(exc))
            return Result(False, str(exc))
        except Exception as exc:  # unexpected: log the type only, show nothing internal
            self._audit.record("error", tool=tool.name, error_type=type(exc).__name__)
            return Result(False, "Something went wrong.")

        self._audit.record("ran", tool=tool.name, risk=tool.risk.value)
        return Result(True, output, private=tool.private_output)

