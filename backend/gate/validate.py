"""Strict argument and filename validation. Deny by default."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping

from .models import Param, ValidationError

# Windows reserved device names (invalid as file names, with any extension).
RESERVED_NAMES = {"con", "prn", "aux", "nul"} | {f"com{i}" for i in range(1, 10)} | {
    f"lpt{i}" for i in range(1, 10)
}

# Letters, digits, space, underscore, hyphen. No dots, slashes, colons or other
# characters, so path tricks and alternate data streams cannot be expressed.
NAME_PATTERN = r"[A-Za-z0-9][A-Za-z0-9 _-]{0,63}"
_NAME_RE = re.compile(NAME_PATTERN)


def validate_args(params: Mapping[str, Param], args: Any) -> dict[str, Any]:
    """Return a clean copy of args, or raise ValidationError.

    Rules: args must be a dict with exactly the declared keys, exact types
    (bool is not an int), length/range limits, no NUL bytes.
    Error messages name parameters only and never echo the supplied values.
    """
    if not isinstance(args, dict):
        raise ValidationError("Arguments must be an object.")
    extra = set(args) - set(params)
    missing = set(params) - set(args)
    if extra:
        raise ValidationError("Unexpected argument(s): " + ", ".join(sorted(map(str, extra))[:5]))
    if missing:
        raise ValidationError("Missing argument(s): " + ", ".join(sorted(missing)))

    clean: dict[str, Any] = {}
    for name, spec in params.items():
        value = args[name]
        if spec.kind is int:
            if type(value) is not int:
                raise ValidationError(f"Argument '{name}' must be an integer.")
            if spec.min_value is not None and value < spec.min_value:
                raise ValidationError(f"Argument '{name}' is too small.")
            if spec.max_value is not None and value > spec.max_value:
                raise ValidationError(f"Argument '{name}' is too large.")
        elif spec.kind is str:
            if type(value) is not str:
                raise ValidationError(f"Argument '{name}' must be text.")
            if len(value) > spec.max_len:
                raise ValidationError(f"Argument '{name}' is too long.")
            if "\x00" in value:
                raise ValidationError(f"Argument '{name}' contains an invalid character.")
            if spec.pattern is not None and re.fullmatch(spec.pattern, value) is None:
                raise ValidationError(f"Argument '{name}' has an invalid format.")
            if spec.choices is not None and value not in spec.choices:
                raise ValidationError(f"Argument '{name}' is not an allowed value.")
        else:
            raise ValidationError("Unsupported parameter type in tool schema.")
        clean[name] = value
    return clean


def safe_file_in(base: Path, name: str, suffix: str) -> Path:
    """Map a simple name to a file directly inside base, or raise ValidationError.

    The name is restricted to a tiny character set (see NAME_PATTERN), then the
    resolved result is checked to still be inside base (defeats symlinks).
    """
    if not isinstance(name, str) or _NAME_RE.fullmatch(name) is None:
        raise ValidationError("Invalid name.")
    if name != name.rstrip(" "):          # Windows silently strips trailing spaces
        raise ValidationError("Invalid name.")
    if name.lower() in RESERVED_NAMES:
        raise ValidationError("Invalid name.")
    base_resolved = base.resolve()
    candidate = (base_resolved / f"{name}{suffix}").resolve()
    if candidate.parent != base_resolved:
        raise ValidationError("Invalid name.")
    return candidate