[CmdletBinding()]
param(
    [int]$Port = 8675,
    [switch]$NoBrowser,
    [switch]$SmokeTest,
    [switch]$Wait,
    [int]$TimeoutSeconds = 90
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ProjectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$UiRoot = Join-Path $ProjectRoot "trainer-ui"
$ServerScript = Join-Path $PSScriptRoot "run_ui_server.ps1"
$LauncherRoot = Join-Path $ProjectRoot "trainer-data\launcher"
$StdoutLog = Join-Path $LauncherRoot "launcher.stdout.log"
$StderrLog = Join-Path $LauncherRoot "launcher.stderr.log"
$PidFile = Join-Path $LauncherRoot "launcher.pid"
$Url = "http://127.0.0.1:$Port"
$HealthUrl = "$Url/api/project"
$process = $null

function Write-LauncherMessage {
    param([string]$Message)
    Write-Host "[D-OPSD Trainer] $Message"
}

function Get-LogTail {
    param(
        [string]$Path,
        [int]$Lines = 40
    )

    if (-not (Test-Path -LiteralPath $Path)) {
        return ""
    }

    return (Get-Content -LiteralPath $Path -Tail $Lines -ErrorAction SilentlyContinue) -join [Environment]::NewLine
}

function Test-DopsdProjectResponse {
    param([object]$Project)

    if ($null -eq $Project -or $null -eq $Project.meta) {
        return $false
    }

    if ($Project.meta.name -ne "D-OPSD Trainer") {
        return $false
    }

    if ($Project.meta.slug -ne "d-opsd-trainer") {
        return $false
    }

    return $true
}

function Test-UiReady {
    param([string]$TargetUrl)

    try {
        $response = Invoke-WebRequest -UseBasicParsing -Uri $TargetUrl -TimeoutSec 2
        if ([int]$response.StatusCode -lt 200 -or [int]$response.StatusCode -ge 300) {
            return $false
        }

        $project = $response.Content | ConvertFrom-Json -ErrorAction Stop
        return (Test-DopsdProjectResponse -Project $project)
    } catch {
        return $false
    }
}

function Test-LocalPortInUse {
    param([int]$TargetPort)

    try {
        $connections = @(Get-NetTCPConnection -LocalPort $TargetPort -State Listen -ErrorAction Stop)
        return ($connections.Count -gt 0)
    } catch {
        try {
            $lines = @(& netstat.exe -ano -p tcp 2>$null)
            $matches = @($lines | Where-Object { $_ -match "^\s*TCP\s+\S+:$TargetPort\s+\S+\s+LISTENING\s+\d+\s*$" })
            return ($matches.Count -gt 0)
        } catch {
            return $false
        }
    }
}

function Test-WslReady {
    try {
        $wslCommand = Get-Command wsl.exe -ErrorAction SilentlyContinue
        if (-not $wslCommand) {
            return $false
        }

        $null = & $wslCommand.Source --status 2>$null
        return ($LASTEXITCODE -eq 0)
    } catch {
        return $false
    }
}

function Stop-LauncherProcessTree {
    param([int]$TargetPid)

    $children = @(Get-CimInstance Win32_Process -Filter "ParentProcessId=$TargetPid" -ErrorAction SilentlyContinue)
    foreach ($child in $children) {
        Stop-LauncherProcessTree -TargetPid ([int]$child.ProcessId)
    }

    $target = Get-Process -Id $TargetPid -ErrorAction SilentlyContinue
    if ($target) {
        Stop-Process -Id $TargetPid -Force -ErrorAction SilentlyContinue
    }
}

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

function Resolve-PowerShellHost {
    $currentHost = [System.Diagnostics.Process]::GetCurrentProcess().Path
    if ($currentHost -and (Test-Path -LiteralPath $currentHost)) {
        return $currentHost
    }

    $windowsPowerShell = Get-Command powershell.exe -ErrorAction SilentlyContinue
    if ($windowsPowerShell) {
        return $windowsPowerShell.Source
    }

    $pwsh = Get-Command pwsh.exe -ErrorAction SilentlyContinue
    if ($pwsh) {
        return $pwsh.Source
    }

    throw "PowerShell was not found."
}

try {
    if (-not (Test-Path -LiteralPath $UiRoot)) {
        throw "UI folder was not found: $UiRoot"
    }

    if (-not (Test-Path -LiteralPath (Join-Path $UiRoot "package.json"))) {
        throw "UI package.json was not found under trainer-ui."
    }

    if (-not (Test-Path -LiteralPath $ServerScript)) {
        throw "UI server wrapper was not found: $ServerScript"
    }

    New-Item -ItemType Directory -Force -Path $LauncherRoot | Out-Null

    Write-LauncherMessage "Project: $ProjectRoot"
    Write-LauncherMessage "UI URL: $Url"

    if (-not (Test-WslReady)) {
        Write-LauncherMessage "Warning: WSL is not responding. The UI can open, but training jobs need WSL2 Ubuntu."
    }

    $npm = Resolve-NpmCommand
    $nodeModules = Join-Path $UiRoot "node_modules"
    if (-not (Test-Path -LiteralPath $nodeModules)) {
        Write-LauncherMessage "Installing UI dependencies because trainer-ui\node_modules is missing."
        & $npm install --prefix $UiRoot
        if ($LASTEXITCODE -ne 0) {
            throw "npm install failed. Check the console output above."
        }
    }

    if (Test-UiReady -TargetUrl $HealthUrl) {
        Write-LauncherMessage "UI is already running."
        if (-not $NoBrowser -and -not $SmokeTest) {
            Start-Process $Url
        }
        if ($Wait) {
            Write-LauncherMessage "Existing UI detected; launcher does not own that process."
        }
        exit 0
    }

    if (Test-LocalPortInUse -TargetPort $Port) {
        throw "Port $Port is already in use by a different service. Stop it or launch with -Port <free port>."
    }

    Write-LauncherMessage "Starting Next.js UI server..."
    $powerShellHost = Resolve-PowerShellHost
    $serverArgs = @(
        "-NoLogo",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        $ServerScript,
        "-ProjectRoot",
        $ProjectRoot,
        "-UiRoot",
        $UiRoot,
        "-Port",
        "$Port"
    )
    $process = Start-Process `
        -FilePath $powerShellHost `
        -ArgumentList $serverArgs `
        -WorkingDirectory $ProjectRoot `
        -RedirectStandardOutput $StdoutLog `
        -RedirectStandardError $StderrLog `
        -PassThru `
        -NoNewWindow

    Set-Content -LiteralPath $PidFile -Value ([string]$process.Id) -Encoding UTF8

    $ready = $false
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        if (Test-UiReady -TargetUrl $HealthUrl) {
            $ready = $true
            break
        }

        $process.Refresh()
        if ($process.HasExited) {
            $stderrTail = Get-LogTail -Path $StderrLog
            throw "UI server exited before it became ready. $stderrTail"
        }

        Start-Sleep -Milliseconds 750
    }

    if (-not $ready) {
        $stderrTail = Get-LogTail -Path $StderrLog
        throw "Timed out waiting for $Url. $stderrTail"
    }

    Write-LauncherMessage "UI is ready: $Url"
    Write-LauncherMessage "Logs: $StdoutLog"

    if ($SmokeTest) {
        Write-LauncherMessage "Smoke test passed; stopping the temporary UI server."
        Stop-LauncherProcessTree -TargetPid $process.Id
        Remove-Item -LiteralPath $PidFile -Force -ErrorAction SilentlyContinue
        exit 0
    }

    if (-not $NoBrowser) {
        Start-Process $Url
    }

    if ($Wait) {
        Write-LauncherMessage "Launcher is keeping the UI server alive. Press Ctrl+C to stop this session."
        Wait-Process -Id $process.Id
        exit 0
    }

    exit 0
} catch {
    if ($SmokeTest -and $null -ne $process -and -not $process.HasExited) {
        Stop-LauncherProcessTree -TargetPid $process.Id
    }

    Write-Error $_.Exception.Message
    Write-Host ""
    Write-Host "Launcher logs:"
    Write-Host "  $StdoutLog"
    Write-Host "  $StderrLog"
    exit 1
}
