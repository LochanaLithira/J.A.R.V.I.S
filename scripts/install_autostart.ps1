<#
.SYNOPSIS
    Start J.A.R.V.I.S automatically when you sign in to Windows, or stop doing so.

.DESCRIPTION
    Creates (or removes) exactly one shortcut, "J.A.R.V.I.S.lnk", in YOUR Startup folder.
    - No administrator rights, no scheduled task, no registry change, no system-wide setting.
      It only affects your own Windows account.
    - The shortcut runs this project's own Python (.venv) with no console window:
          .venv\Scripts\pythonw.exe -m backend.main
    - You can also turn it off any time in Task Manager > Startup apps.
    - -Remove deletes only that one shortcut, and only if it really points at this project.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\install_autostart.ps1
    Turn autostart on.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\install_autostart.ps1 -Status
    Show whether autostart is on.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\install_autostart.ps1 -Remove
    Turn autostart off.

.NOTES
    "-ExecutionPolicy Bypass" applies to this one run only; it does not change any setting.
#>
[CmdletBinding()]
param(
    [switch]$Remove,
    [switch]$Status,
    # For testing only: use another folder instead of your real Startup folder.
    [string]$StartupDir = [Environment]::GetFolderPath('Startup')
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$LinkName = 'J.A.R.V.I.S.lnk'
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$PythonW = Join-Path $Root '.venv\Scripts\pythonw.exe'
$Arguments = '-m backend.main'
$Link = Join-Path $StartupDir $LinkName

function Read-Link([string]$Path) {
    $shell = New-Object -ComObject WScript.Shell
    return $shell.CreateShortcut($Path)
}

function Test-OurLink([string]$Path) {
    # Only a shortcut that runs this project's pythonw with "-m backend.main" counts as ours.
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { return $false }
    $sc = Read-Link $Path
    return ($sc.TargetPath -ieq $PythonW) -and ($sc.Arguments -eq $Arguments) -and ($sc.WorkingDirectory -ieq $Root)
}

if (-not (Test-Path -LiteralPath $StartupDir -PathType Container)) {
    throw "The Startup folder was not found: $StartupDir"
}

if ($Status) {
    if (Test-OurLink $Link) {
        Write-Output "Autostart is ON: J.A.R.V.I.S starts when you sign in."
        Write-Output "  Shortcut: $Link"
    } elseif (Test-Path -LiteralPath $Link) {
        Write-Output "A shortcut called $LinkName exists but does not point at this project: $Link"
    } else {
        Write-Output "Autostart is OFF."
    }
    exit 0
}

if ($Remove) {
    if (Test-OurLink $Link) {
        Remove-Item -LiteralPath $Link
        Write-Output "Autostart is now OFF. (J.A.R.V.I.S keeps running until you quit it from the tray.)"
    } elseif (Test-Path -LiteralPath $Link) {
        throw "Refusing to remove $Link because it does not point at this project. Remove it yourself if you're sure."
    } else {
        Write-Output "Autostart was already OFF."
    }
    exit 0
}

# ---- turn autostart on ----------------------------------------------------------------------
if (-not (Test-Path -LiteralPath $PythonW -PathType Leaf)) {
    throw "This project's Python was not found at $PythonW. Create the virtual environment first (see README)."
}
if (-not (Test-Path -LiteralPath (Join-Path $Root 'backend\main.py') -PathType Leaf)) {
    throw "backend\main.py was not found under $Root."
}
if ((Test-Path -LiteralPath $Link) -and -not (Test-OurLink $Link)) {
    throw "A different shortcut called $LinkName is already in your Startup folder: $Link. Not replacing it."
}

$sc = Read-Link $Link
$sc.TargetPath = $PythonW
$sc.Arguments = $Arguments
$sc.WorkingDirectory = $Root
$sc.Description = 'J.A.R.V.I.S voice assistant (runs in the background, tray icon by the clock)'
$sc.WindowStyle = 7          # minimised; pythonw opens no console window anyway
$sc.Save()

Write-Output "Autostart is now ON: J.A.R.V.I.S will start the next time you sign in."
Write-Output "  Shortcut: $Link"
Write-Output "To start it right now:  .venv\Scripts\pythonw.exe -m backend.main"
Write-Output "To turn it off:         powershell -ExecutionPolicy Bypass -File scripts\install_autostart.ps1 -Remove"
