"""Open allowlisted apps. Fixed executable paths, no shell, no arguments."""
from __future__ import annotations

import subprocess
from typing import Callable, Mapping

from ..gate.models import Param, Risk, Tool, ToolError


def make_app_tools(allowed: Mapping[str, str], launcher: Callable = subprocess.Popen) -> list[Tool]:
    def open_app(name: str) -> str:
        exe = allowed.get(name)
        if exe is None:
            raise ToolError("That app is not allowed.")
        try:
            launcher([exe], shell=False, stdin=subprocess.DEVNULL,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except OSError:
            raise ToolError("I couldn't open that app.")
        return f"Opening {name}."

    return [
        Tool("open_app", "Open an allowlisted application.", Risk.LOW,
             {"name": Param(str, max_len=32, choices=tuple(sorted(allowed)))}, open_app),
    ]