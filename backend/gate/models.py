"""Core types for the tool gate. Pure data, no side effects."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Mapping


class Risk(str, Enum):
    LOW = "low"        # runs automatically
    MEDIUM = "medium"  # runs automatically, always logged, reversible
    HIGH = "high"      # needs an on-screen keypress confirmation


class ValidationError(ValueError):
    """Raised when arguments do not match a tool's schema."""


class ToolError(Exception):
    """A tool failed in an expected way. The message is safe to show the user."""


@dataclass(frozen=True)
class Param:
    kind: type                      # str or int only
    max_len: int = 200              # str only
    pattern: str | None = None      # str only, must fully match
    choices: tuple | None = None    # str only, exact match
    min_value: int | None = None    # int only
    max_value: int | None = None    # int only


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    risk: Risk
    params: Mapping[str, Param]
    fn: Callable[..., str]
    private_output: bool = False    # True if the output can contain the user's own data (note text, file names...)


@dataclass(frozen=True)
class Result:
    ok: bool
    message: str
    private: bool = False           # True: show on screen only, never send to a cloud service (e.g. text-to-speech)