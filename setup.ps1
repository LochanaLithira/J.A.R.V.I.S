<#
.SYNOPSIS
    Get J.A.R.V.I.S ready to run after cloning (or downloading) the project, then start it.

.DESCRIPTION
    1. Finds Python 3.14 (64-bit, the normal build). If it is missing, says where to get it.
    2. Creates this project's own Python environment in .venv (only inside this folder).
    3. Installs the exact packages listed in requirements.txt. Every file is checked against the
       fingerprint written there, so a changed or swapped package is refused.
    4. Starts J.A.R.V.I.S in the background. The first time, its screen opens on the setup guide
       (keys, models, microphone, your voice, starting with Windows).

    No administrator rights, nothing outside this folder (Python's own package cache aside), no
    system setting is changed. Run it again after "git pull" to bring the packages up to date.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File setup.ps1
    Set up and start J.A.R.V.I.S.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File setup.ps1 -NoStart
    Set up only.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File setup.ps1 -CheckOnly
    Only check that a suitable Python is installed.

.NOTES
    "-ExecutionPolicy Bypass" applies to this one run only; it does not change any setting.
#>
[CmdletBinding()]
param(
    [switch]$NoStart,
    [switch]$CheckOnly
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$Root = $PSScriptRoot
$Venv = Join-Path $Root '.venv'
$VenvPython = Join-Path $Venv 'Scripts\python.exe'
$VenvPythonW = Join-Path $Venv 'Scripts\pythonw.exe'
$Requirements = Join-Path $Root 'requirements.txt'
$DownloadPage = 'https://www.python.org/downloads/windows/'

# Prints "3.14 64 gil" for a suitable Python: version, bits, and whether it is the normal build.
# (No double quotes inside: Windows PowerShell 5.1 would mangle them on the way to Python.)
$Probe = 'import struct,sys,sysconfig;print(''%d.%d'' % sys.version_info[:2],struct.calcsize(''P'')*8,''nogil'' if sysconfig.get_config_var(''Py_GIL_DISABLED'') else ''gil'')'

function Step([string]$Text) { Write-Host "`n== $Text" -ForegroundColor Cyan }
function Fail([string]$Text) { Write-Host "`n$Text" -ForegroundColor Red; exit 1 }

function Invoke-Quiet([string]$Exe, [string[]]$Arguments) {
    # Run a program, hiding its error output, and return (exit code, output). Error output from a
    # program never stops this script here (Windows PowerShell 5.1 would otherwise treat it as fatal).
    $saved = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $out = & $Exe @Arguments 2>$null
        return @($LASTEXITCODE, "$out".Trim())
    } catch {
        return @(1, '')
    } finally {
        $ErrorActionPreference = $saved
    }
}

function Test-Python([string]$Exe, [string[]]$Pre) {
    # True if this Python is 3.14, 64-bit and the normal (not free-threaded) build.
    $code, $out = Invoke-Quiet $Exe (@($Pre) + @('-c', $Probe))
    return ($code -eq 0) -and ($out -eq '3.14 64 gil')
}

function Find-Python {
    # The Python launcher first (asking for the normal 3.14 by name), then "python" on PATH.
    $candidates = @()
    if (Get-Command py -ErrorAction SilentlyContinue) { $candidates += , @('py', @('-V:3.14')) }
    if (Get-Command python -ErrorAction SilentlyContinue) { $candidates += , @('python', @()) }
    foreach ($c in $candidates) {
        if (Test-Python $c[0] $c[1]) { return , $c }
    }
    return $null
}

Write-Host 'J.A.R.V.I.S setup' -ForegroundColor Cyan

# ---- 1. Python ----------------------------------------------------------------------------
Step 'Looking for Python 3.14 (64-bit)'
$python = Find-Python
if ($null -eq $python) {
    Fail ("Python 3.14 (64-bit) was not found.`n" +
          "Install it from $DownloadPage (the 'Windows installer (64-bit)'),`n" +
          "keep 'tcl/tk' ticked (J.A.R.V.I.S uses it for its popups), then run this again.")
}
$pyExe, $pyPre = $python[0], $python[1]
Write-Host "Found: $pyExe $($pyPre -join ' ')"
$code, $null = Invoke-Quiet $pyExe (@($pyPre) + @('-c', 'import tkinter'))
if ($code -ne 0) {
    Fail "This Python has no tkinter. Run the Python installer again, choose Modify, tick 'tcl/tk', then run this again."
}
if ($CheckOnly) {
    Write-Host 'Python is ready.' -ForegroundColor Green
    exit 0
}
if (-not (Test-Path -LiteralPath $Requirements -PathType Leaf)) {
    Fail "requirements.txt is missing. Run this from the J.A.R.V.I.S folder."
}

# ---- 2. The project's own environment ---------------------------------------------------------
Step 'Creating the project environment (.venv)'
if ((Test-Path -LiteralPath $VenvPython -PathType Leaf) -and -not (Test-Python $VenvPython @())) {
    Fail ".venv exists but was made with a different Python. Delete the .venv folder and run this again."
}
if (Test-Path -LiteralPath $VenvPython -PathType Leaf) {
    Write-Host 'Already there.'
} else {
    & $pyExe @pyPre -m venv $Venv
    if ($LASTEXITCODE -ne 0) { Fail 'Could not create .venv.' }
    Write-Host 'Created.'
}

# ---- 3. Packages ----------------------------------------------------------------------------------
Step 'Installing packages (each one checked against its fingerprint; a few minutes the first time)'
& $VenvPython -m pip install --disable-pip-version-check --require-hashes -r $Requirements
if ($LASTEXITCODE -ne 0) {
    Fail 'Installing the packages failed. Check the internet connection and run this again.'
}

# ---- 4. Start ----------------------------------------------------------------------------------------
Write-Host "`nJ.A.R.V.I.S is set up." -ForegroundColor Green
if ($NoStart) {
    Write-Host 'Start it any time by double-clicking start.bat.'
    exit 0
}
Step 'Starting J.A.R.V.I.S'
Start-Process -FilePath $VenvPythonW -ArgumentList '-m', 'backend.main' -WorkingDirectory $Root
Write-Host 'It runs in the background: look for its icon next to the clock. The first time, its screen'
Write-Host 'opens on the setup guide. Next time, double-click start.bat (or turn on "Start with Windows").'
exit 0
