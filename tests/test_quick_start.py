"""setup.ps1 and start.bat: getting J.A.R.V.I.S running after cloning the project."""
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SETUP = ROOT / "setup.ps1"
START = ROOT / "start.bat"


def test_files_are_plain_ascii():
    # Windows PowerShell 5.1 reads a file without a byte-order mark as the old code page.
    for path in (SETUP, START):
        assert all(b < 128 for b in path.read_bytes()), path.name


def test_setup_installs_only_fingerprinted_packages_into_the_project():
    text = SETUP.read_text(encoding="ascii")
    assert "--require-hashes -r $Requirements" in text
    assert "$Venv = Join-Path $Root '.venv'" in text
    assert "-m', 'backend.main'" in text and "pythonw.exe" in text
    for risky in ("Invoke-WebRequest", "Invoke-Expression", "iex ", "Set-ExecutionPolicy", "RunAs", "HKLM:"):
        assert risky not in text, risky


def test_setup_asks_for_the_normal_python_314():
    text = SETUP.read_text(encoding="ascii")
    assert "'-V:3.14'" in text and "'3.14 64 gil'" in text     # not the free-threaded 3.14t
    probe = text.split("$Probe = '", 1)[1].split("'\n", 1)[0]
    assert '"' not in probe                                    # PowerShell 5.1 would mangle double quotes


def test_start_sets_up_first_then_starts_in_the_background():
    text = START.read_text(encoding="ascii")
    assert 'if not exist ".venv\\Scripts\\pythonw.exe"' in text
    assert 'setup.ps1' in text
    assert 'start "" ".venv\\Scripts\\pythonw.exe" -m backend.main' in text


@pytest.mark.skipif(sys.platform != "win32" or shutil.which("powershell") is None, reason="needs Windows PowerShell")
def test_check_only_finds_this_python():
    done = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(SETUP), "-CheckOnly"],
                          capture_output=True, text=True, timeout=120)
    if "was not found" in done.stdout:
        pytest.skip("no normal 64-bit Python 3.14 on this machine")
    assert done.returncode == 0 and "Python is ready." in done.stdout
