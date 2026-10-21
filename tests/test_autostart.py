"""The autostart script, run against a temporary folder (never your real Startup folder)."""
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "install_autostart.ps1"

pytestmark = pytest.mark.skipif(sys.platform != "win32" or shutil.which("powershell") is None,
                                reason="needs Windows PowerShell")


def run(startup, *flags):
    done = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(SCRIPT),
                           "-StartupDir", str(startup), *flags], capture_output=True, text=True, timeout=60)
    return done.returncode, done.stdout + done.stderr


def shortcut(path):
    ps = (f"$s = (New-Object -ComObject WScript.Shell).CreateShortcut('{path}'); "
          "Write-Output $s.TargetPath; Write-Output $s.Arguments; Write-Output $s.WorkingDirectory")
    out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True).stdout
    return out.strip().splitlines()


def make_foreign(path):
    ps = (f"$s = (New-Object -ComObject WScript.Shell).CreateShortcut('{path}'); "
          "$s.TargetPath = 'C:\\Windows\\notepad.exe'; $s.Save()")
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True, capture_output=True)


def test_on_status_off(tmp_path):
    link = tmp_path / "J.A.R.V.I.S.lnk"
    code, out = run(tmp_path)
    assert code == 0 and "ON" in out and link.is_file()
    assert shortcut(link) == [str(ROOT / ".venv" / "Scripts" / "pythonw.exe"), "-m backend.main", str(ROOT)]
    assert "Autostart is ON" in run(tmp_path, "-Status")[1]
    assert run(tmp_path)[0] == 0                                   # turning it on twice is fine
    code, out = run(tmp_path, "-Remove")
    assert code == 0 and not link.exists()
    assert "Autostart is OFF" in run(tmp_path, "-Status")[1]
    assert "already OFF" in run(tmp_path, "-Remove")[1]


def test_only_its_own_shortcut_is_touched(tmp_path):
    link = tmp_path / "J.A.R.V.I.S.lnk"
    make_foreign(link)
    assert run(tmp_path, "-Remove")[0] != 0 and link.exists()      # someone else's: never removed
    assert run(tmp_path)[0] != 0                                   # and never overwritten
    assert shortcut(link)[0].lower().endswith("notepad.exe")
    assert "does not point at this project" in run(tmp_path, "-Status")[1]


def test_nothing_else_is_written(tmp_path):
    run(tmp_path)
    assert [p.name for p in tmp_path.iterdir()] == ["J.A.R.V.I.S.lnk"]


def test_missing_startup_folder_is_an_error(tmp_path):
    code, out = run(tmp_path / "missing")
    assert code != 0 and "not found" in out
