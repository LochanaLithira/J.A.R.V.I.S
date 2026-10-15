"""Tests for the configuration settings."""
import re
import sys
from pathlib import Path, PureWindowsPath

import pytest

from backend import config


def test_app_names_are_simple():
    for name in config.ALLOWED_APPS:
        assert re.fullmatch(r"[a-z0-9_]{1,32}", name)


def test_app_paths_are_absolute_exe_files():
    for path in config.ALLOWED_APPS.values():
        p = PureWindowsPath(path)
        assert p.is_absolute() and p.suffix.lower() == ".exe"


@pytest.mark.skipif(sys.platform != "win32", reason="Windows paths only")
def test_app_files_exist_on_this_machine():
    missing = [n for n, p in config.ALLOWED_APPS.items() if not Path(p).is_file()]
    assert not missing, f"Fix these paths in config.py: {missing}"


def test_audit_log_is_outside_the_workspace():
    assert config.AUDIT_LOG != config.WORKSPACE
    assert config.WORKSPACE not in config.AUDIT_LOG.parents


def test_limits_are_sane():
    assert 1 <= config.RATE_LIMIT_CALLS <= 100
    assert 1 <= config.RATE_LIMIT_WINDOW_SECONDS <= 3600