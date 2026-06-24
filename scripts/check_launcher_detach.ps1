[CmdletBinding()]
param(
    [int]$Port = 18861,
    [int]$TimeoutSeconds = 120,
    [int]$HoldSeconds = 5
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ProjectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$StartScript = Join-Path $PSScriptRoot "start_trainer.ps1"
$LauncherRoot = Join-Path $ProjectRoot "trainer-data\launcher"
$PidFile = Join-Path $LauncherRoot "launcher.pid"
$StdoutLog = Join-Path $LauncherRoot "launcher-detach.stdout.log"
$StderrLog = Join-Path $LauncherRoot "launcher-detach.stderr.log"
$BaseUrl = "http://127.0.0.1:$Port"
$HealthUrl = "$BaseUrl/api/project"

function Resolve-PowerShellHost {
    $pwsh = Get-Command pwsh.exe -ErrorAction SilentlyContinue
    if ($pwsh) {
        return $pwsh.Source
    }

    $windowsPowerShell = Get-Command powershell.exe -ErrorAction SilentlyContinue
    if ($windowsPowerShell) {
        return $windowsPowerShell.Source
    }

    throw "PowerShell was not found."
}

function Invoke-ProjectHealth {
    try {
        $project = Invoke-RestMethod -Uri $HealthUrl -TimeoutSec 3
        return ($null -ne $project.meta -and $project.meta.name -eq "D-OPSD Trainer" -and $project.meta.slug -eq "d-opsd-trainer")
    } catch {
        return $false
    }
}

function Get-PortOwnerPids {
    try {
        return @(
            Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
                Select-Object -ExpandProperty OwningProcess -Unique |
                Where-Object { $_ -gt 0 }
        )
    } catch {
        return @()
    }
}

function Get-ProjectPortProcessPids {
    $escapedPort = [regex]::Escape("$Port")
    return @(
        Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
            Where-Object {
                $_.ProcessId -ne $PID -and
                $_.CommandLine -and
                $_.CommandLine -like "*$ProjectRoot*" -and
                $_.CommandLine -match $escapedPort -and
                (
                    $_.CommandLine -match "start_trainer\.ps1" -or
                    $_.CommandLine -match "run_ui_server\.ps1" -or
                    $_.CommandLine -match "next"
                )
            } |
            Select-Object -ExpandProperty ProcessId -Unique |
            Where-Object { $_ -gt 0 }
    )
}

function Get-ExistingProjectUiServerPids {
    return @(
        Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
            Where-Object {
                $_.ProcessId -ne $PID -and
                $_.CommandLine -and
                $_.CommandLine -like "*$ProjectRoot*" -and
                (
                    $_.CommandLine -match "run_ui_server\.ps1" -or
                    $_.CommandLine -match "next\W+dev"
                )
            } |
            Select-Object -ExpandProperty ProcessId -Unique |
            Where-Object { $_ -gt 0 }
    )
}

function Assert-NoExistingProjectUiServer {
    $existingPids = @(Get-ExistingProjectUiServerPids)
    if ($existingPids.Count -gt 0) {
        throw "Existing project UI process detected: $($existingPids -join ', '). Stop the running UI before this detach check; concurrent Next dev servers share .next and can invalidate routes."
    }
}

function Stop-ProcessTree {
    param([int]$TargetPid)

    if ($TargetPid -le 0 -or $TargetPid -eq $PID) {
        return
    }

    $targetInfo = Get-CimInstance Win32_Process -Filter "ProcessId=$TargetPid" -ErrorAction SilentlyContinue
    if ($null -eq $targetInfo -or -not $targetInfo.CommandLine -or $targetInfo.CommandLine -notlike "*$ProjectRoot*") {
        return
    }

    $children = @(Get-CimInstance Win32_Process -Filter "ParentProcessId=$TargetPid" -ErrorAction SilentlyContinue)
    foreach ($child in $children) {
        Stop-ProcessTree -TargetPid ([int]$child.ProcessId)
    }

    Stop-Process -Id $TargetPid -Force -ErrorAction SilentlyContinue
}

function Stop-TemporaryUi {
    $ownerPids = @(Get-PortOwnerPids)
    $projectPids = @(Get-ProjectPortProcessPids)
    $targets = @($ownerPids + $projectPids | Select-Object -Unique)

    foreach ($targetPid in $targets) {
        Stop-ProcessTree -TargetPid ([int]$targetPid)
    }
}

function Wait-ForLauncherExit {
    param([System.Diagnostics.Process]$Process)

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        $Process.Refresh()
        if ($Process.HasExited) {
            return
        }
        Start-Sleep -Milliseconds 500
    }

    throw "Launcher process did not exit in non-Wait mode."
}

function Wait-ForHealth {
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        if (Invoke-ProjectHealth) {
            return
        }
        Start-Sleep -Milliseconds 750
    }

    throw "Timed out waiting for $HealthUrl to return D-OPSD Trainer metadata."
}

if (-not (Test-Path -LiteralPath $StartScript)) {
    throw "Launcher script not found: $StartScript"
}

if (Invoke-ProjectHealth -or @(Get-PortOwnerPids).Count -gt 0) {
    throw "Port $Port is already serving or listening. Use a free temporary port."
}

Assert-NoExistingProjectUiServer

New-Item -ItemType Directory -Force -Path $LauncherRoot | Out-Null
$previousPidFile = $null
if (Test-Path -LiteralPath $PidFile) {
    $previousPidFile = Get-Content -Raw -LiteralPath $PidFile
}

$process = $null
try {
    Remove-Item -LiteralPath $StdoutLog, $StderrLog -Force -ErrorAction SilentlyContinue

    $launcherArgs = @(
        "-NoLogo",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        $StartScript,
        "-NoBrowser",
        "-Port",
        "$Port",
        "-TimeoutSeconds",
        "$TimeoutSeconds"
    )

    $process = Start-Process `
        -FilePath (Resolve-PowerShellHost) `
        -ArgumentList $launcherArgs `
        -WorkingDirectory $ProjectRoot `
        -RedirectStandardOutput $StdoutLog `
        -RedirectStandardError $StderrLog `
        -PassThru `
        -WindowStyle Hidden

    Wait-ForHealth
    Wait-ForLauncherExit -Process $process
    Start-Sleep -Seconds $HoldSeconds

    $healthyAfterLauncherExit = Invoke-ProjectHealth
    if (-not $healthyAfterLauncherExit) {
        throw "UI did not stay healthy after the non-Wait launcher exited."
    }

    [pscustomobject]@{
        Ok = $true
        BaseUrl = $BaseUrl
        LauncherExited = $process.HasExited
        HealthyAfterLauncherExit = $healthyAfterLauncherExit
        HoldSeconds = $HoldSeconds
    } | Format-List
} finally {
    Stop-TemporaryUi

    if ($null -ne $previousPidFile) {
        Set-Content -LiteralPath $PidFile -Value $previousPidFile -Encoding UTF8
    } else {
        Remove-Item -LiteralPath $PidFile -Force -ErrorAction SilentlyContinue
    }
}
