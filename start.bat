@echo off
rem Start J.A.R.V.I.S in the background. The first time, this sets it up (setup.ps1) instead.
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
    echo J.A.R.V.I.S isn't set up yet, so setting it up now...
    powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1"
    pause
    exit /b
)
start "" ".venv\Scripts\pythonw.exe" -m backend.main
