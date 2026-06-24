[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$ProjectRoot,
    [Parameter(Mandatory = $true)]
    [string]$UiRoot,
    [int]$Port = 8675
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Resolve-NpmCommand {
    $npmCommand = Get-Command npm.cmd -ErrorAction SilentlyContinue
    if (-not $npmCommand) {
        $npmCommand = Get-Command npm -ErrorAction SilentlyContinue
    }

    if (-not $npmCommand) {
        throw "npm was not found. Install Node.js, then rerun D-OPSD-Trainer.cmd."
    }

    return $npmCommand.Source
}

if (-not (Test-Path -LiteralPath $ProjectRoot)) {
    throw "Project root was not found: $ProjectRoot"
}

if (-not (Test-Path -LiteralPath (Join-Path $UiRoot "package.json"))) {
    throw "UI package.json was not found: $UiRoot"
}

$npm = Resolve-NpmCommand
$env:NEXT_TELEMETRY_DISABLED = "1"
Set-Location -LiteralPath $ProjectRoot

& $npm run dev --prefix $UiRoot -- --hostname "127.0.0.1" --port "$Port"
exit $LASTEXITCODE
