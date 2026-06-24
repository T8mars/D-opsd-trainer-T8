[CmdletBinding()]
param(
  [string]$Distro = "Ubuntu-22.04",
  [string]$ProjectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path,
  [switch]$SkipPipInstall,
  [switch]$SkipProbe
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function ConvertTo-BashSingleQuoted {
  param([Parameter(Mandatory = $true)][string]$Value)
  $singleQuote = [string][char]39
  $doubleQuote = [string][char]34
  $escapedSingleQuote = $singleQuote + $doubleQuote + $singleQuote + $doubleQuote + $singleQuote
  return $singleQuote + $Value.Replace($singleQuote, $escapedSingleQuote) + $singleQuote
}

function Invoke-Checked {
  param(
    [Parameter(Mandatory = $true)][string]$FilePath,
    [Parameter(Mandatory = $true)][string[]]$ArgumentList,
    [Parameter(Mandatory = $true)][string]$Name
  )

  $output = & $FilePath @ArgumentList 2>&1
  if ($LASTEXITCODE -ne 0) {
    $detail = ($output | Out-String).Trim()
    throw "$Name failed with exit code $LASTEXITCODE. $detail"
  }
  return $output
}

function ConvertTo-WslPath {
  param(
    [Parameter(Mandatory = $true)][string]$WindowsPath,
    [Parameter(Mandatory = $true)][string]$Name
  )

  $resolved = (Resolve-Path -LiteralPath $WindowsPath).Path
  if ($resolved -match "^([A-Za-z]):\\(.*)$") {
    $drive = $Matches[1].ToLowerInvariant()
    $rest = $Matches[2] -replace "\\", "/"
    return "/mnt/$drive/$rest"
  }

  return (Invoke-Checked -FilePath "wsl.exe" -ArgumentList @("-d", $Distro, "--", "wslpath", "-a", $resolved) -Name $Name | Select-Object -First 1).Trim()
}

if (-not (Get-Command "wsl.exe" -ErrorAction SilentlyContinue)) {
  throw "wsl.exe was not found. Install WSL2 and the $Distro distro first."
}

$ProjectRoot = (Resolve-Path -LiteralPath $ProjectRoot).Path
$requirementsPath = Join-Path $ProjectRoot "requirements-trainer.txt"
$envPath = Join-Path $ProjectRoot "scripts\dopsd_wsl_env.sh"

if (-not (Test-Path -LiteralPath $requirementsPath)) {
  throw "Missing requirements-trainer.txt at $requirementsPath"
}
if (-not (Test-Path -LiteralPath $envPath)) {
  throw "Missing dopsd_wsl_env.sh at $envPath"
}

Invoke-Checked -FilePath "wsl.exe" -ArgumentList @("-d", $Distro, "--", "bash", "-lc", "printf ready") -Name "WSL distro probe" | Out-Null
$wslProjectRoot = ConvertTo-WslPath -WindowsPath $ProjectRoot -Name "wslpath project root"
$quotedProjectRoot = ConvertTo-BashSingleQuoted $wslProjectRoot

$setupLines = @(
  "set -euo pipefail",
  "cd $quotedProjectRoot",
  "export PROJECT_ROOT=$quotedProjectRoot",
  "source scripts/dopsd_wsl_env.sh",
  "if ! command -v python3 >/dev/null 2>&1; then echo 'python3 is required inside WSL.' >&2; exit 1; fi",
  "python3 -m venv ""`$VIRTUAL_ENV""",
  "source ""`$VIRTUAL_ENV/bin/activate"""
)

if ($SkipPipInstall) {
  $setupLines += "echo 'Skipping pip install by request.'"
} else {
  $setupLines += @(
    "python -m pip install --upgrade pip wheel setuptools",
    "python -m pip install -r requirements-trainer.txt",
    "python -m pip check"
  )
}

if (-not $SkipProbe) {
  $setupLines += @(
    "python scripts/check_runtime.py probe --project-root ""`$PROJECT_ROOT"" >/tmp/dopsd_trainer_probe.json",
    "python scripts/check_runtime.py settings --project-root ""`$PROJECT_ROOT"" >/tmp/dopsd_trainer_settings.json"
  )
}

$setupLines += "echo setup_wsl_trainer_ok"

$setupDir = Join-Path $ProjectRoot "trainer-data\setup"
New-Item -ItemType Directory -Force -Path $setupDir | Out-Null
$setupScript = Join-Path $setupDir "setup_wsl_trainer.sh"
$utf8NoBom = [System.Text.UTF8Encoding]::new($false)
[System.IO.File]::WriteAllText($setupScript, (($setupLines -join "`n") + "`n"), $utf8NoBom)
$wslSetupScript = ConvertTo-WslPath -WindowsPath $setupScript -Name "wslpath setup script"

Invoke-Checked -FilePath "wsl.exe" -ArgumentList @("-d", $Distro, "--", "bash", $wslSetupScript) -Name "WSL trainer dependency setup" | ForEach-Object {
  Write-Host $_
}

[pscustomobject]@{
  Ok = $true
  Distro = $Distro
  ProjectRoot = $ProjectRoot
  WslProjectRoot = $wslProjectRoot
  SetupScript = $setupScript
  Requirements = $requirementsPath
} | ConvertTo-Json -Depth 4
