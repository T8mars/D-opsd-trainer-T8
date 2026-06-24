[CmdletBinding()]
param(
    [int]$Port = 18782,
    [int]$ProbeDurationSeconds = 30,
    [int]$TimeoutSeconds = 150
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ProjectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$UiRoot = Join-Path $ProjectRoot "trainer-ui"
$ServerScript = Join-Path $PSScriptRoot "run_ui_server.ps1"
$BaseUrl = "http://127.0.0.1:$Port"
$HealthUrl = "$BaseUrl/api/project"
$LogRoot = Join-Path $ProjectRoot "trainer-data\launcher\ui-restart-check"
$JobsPath = Join-Path $ProjectRoot "trainer-data\jobs\jobs.json"
$StopDiagLog = Join-Path $LogRoot "stop-diagnostics.log"

function Resolve-PowerShellHost {
    $pwsh = Get-Command pwsh.exe -ErrorAction SilentlyContinue
    if ($pwsh) {
        return $pwsh.Source
    }

    $currentHost = [System.Diagnostics.Process]::GetCurrentProcess().Path
    if ($currentHost -and (Test-Path -LiteralPath $currentHost)) {
        return $currentHost
    }

    $windowsPowerShell = Get-Command powershell.exe -ErrorAction SilentlyContinue
    if ($windowsPowerShell) {
        return $windowsPowerShell.Source
    }

    throw "PowerShell was not found."
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

function Write-StopDiagnostic {
    param([string]$Message)

    New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null
    Add-Content -LiteralPath $StopDiagLog -Value "[$((Get-Date).ToString('o'))] $Message" -Encoding UTF8
}

function Test-UiReady {
    try {
        $response = Invoke-WebRequest -UseBasicParsing -Uri $HealthUrl -TimeoutSec 2
        return ([int]$response.StatusCode -ge 200 -and [int]$response.StatusCode -lt 500)
    } catch {
        return $false
    }
}

function Start-UiServer {
    param([string]$Name)

    New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null
    $stdout = Join-Path $LogRoot "$Name.stdout.log"
    $stderr = Join-Path $LogRoot "$Name.stderr.log"
    Remove-Item -LiteralPath $stdout, $stderr -Force -ErrorAction SilentlyContinue

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

    return Start-Process `
        -FilePath (Resolve-PowerShellHost) `
        -ArgumentList $serverArgs `
        -WorkingDirectory $ProjectRoot `
        -RedirectStandardOutput $stdout `
        -RedirectStandardError $stderr `
        -PassThru `
        -WindowStyle Hidden
}

function Stop-ProcessTree {
    param([int]$TargetPid)

    $children = @(Get-CimInstance Win32_Process -Filter "ParentProcessId=$TargetPid" -ErrorAction SilentlyContinue)
    foreach ($child in $children) {
        Stop-ProcessTree -TargetPid ([int]$child.ProcessId)
    }

    $target = Get-Process -Id $TargetPid -ErrorAction SilentlyContinue
    if ($target) {
        Stop-Process -Id $TargetPid -Force -ErrorAction SilentlyContinue
    }
}

function Stop-ProcessTreeHard {
    param([int]$TargetPid)

    if ($TargetPid -le 0 -or $TargetPid -eq $PID) {
        return
    }

    & cmd.exe /c "taskkill.exe /PID $TargetPid /T /F >nul 2>nul" | Out-Null
    Stop-ProcessTree -TargetPid $TargetPid
}

function Stop-ProcessHard {
    param([int]$TargetPid)

    if ($TargetPid -le 0 -or $TargetPid -eq $PID) {
        return
    }

    & cmd.exe /c "taskkill.exe /PID $TargetPid /F >nul 2>nul" | Out-Null
    Stop-Process -Id $TargetPid -Force -ErrorAction SilentlyContinue
}

function Get-PortOwnerPids {
    try {
        return @(
            Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue |
                Where-Object { $_.State -eq "Listen" } |
                Select-Object -ExpandProperty OwningProcess -Unique |
                Where-Object { $_ -gt 0 }
        )
    } catch {
        return @()
    }
}

function Get-NetstatPortOwnerPids {
    try {
        $owners = New-Object System.Collections.Generic.List[int]
        $lines = & netstat.exe -ano -p tcp 2>$null
        foreach ($line in $lines) {
            if ($line -notmatch "[:.]$Port\s") {
                continue
            }

            $parts = @($line -split "\s+" | Where-Object { $_ })
            if ($parts.Count -lt 5) {
                continue
            }

            if ($parts[-2] -ne "LISTENING") {
                continue
            }

            $processId = 0
            if ([int]::TryParse($parts[-1], [ref]$processId) -and $processId -gt 0 -and -not $owners.Contains($processId)) {
                $owners.Add($processId)
            }
        }
        return @($owners)
    } catch {
        return @()
    }
}

function Get-UiProcessPids {
    $escapedPort = [regex]::Escape("$Port")
    try {
        return @(
            Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
                Where-Object {
                    $_.ProcessId -ne $PID -and
                    $_.CommandLine -and
                    $_.CommandLine -match $escapedPort -and
                    (
                        $_.CommandLine -match "run_ui_server\.ps1" -or
                        $_.CommandLine -match "next dev" -or
                        $_.CommandLine -match "npm-cli\.js"
                    )
                } |
                Select-Object -ExpandProperty ProcessId -Unique |
                Where-Object { $_ -gt 0 }
        )
    } catch {
        return @()
    }
}

function Stop-UiServer {
    param(
        [System.Diagnostics.Process]$Process,
        [string]$Description
    )

    $deadline = (Get-Date).AddSeconds(60)
    while ((Get-Date) -lt $deadline) {
        $uiProcesses = @(Get-UiProcessPids)
        $portOwners = @((Get-PortOwnerPids) + (Get-NetstatPortOwnerPids) | Select-Object -Unique)
        Write-StopDiagnostic "$Description stop loop: uiPids=$($uiProcesses -join ',') portOwners=$($portOwners -join ',') processId=$($Process.Id)"

        foreach ($uiProcess in $uiProcesses) {
            Stop-ProcessTreeHard -TargetPid ([int]$uiProcess)
        }

        foreach ($owningProcess in $portOwners) {
            Stop-ProcessTreeHard -TargetPid ([int]$owningProcess)
        }

        if ($null -ne $Process) {
            try {
                $Process.Refresh()
                if (-not $Process.HasExited) {
                    Stop-ProcessTreeHard -TargetPid ([int]$Process.Id)
                }
            } catch {
                Write-Warning "Process tree stop failed for ${Description}: $($_.Exception.Message)"
            }
        }

        $ready = Test-UiReady
        Write-StopDiagnostic "$Description health after stop attempts: ready=$ready"
        if (-not $ready) {
            return
        }

        Start-Sleep -Milliseconds 750
    }

    throw "Timed out waiting for $Description to stop serving $HealthUrl"
}

function Wait-ForUiReady {
    param(
        [System.Diagnostics.Process]$Process,
        [string]$Name
    )

    $stderr = Join-Path $LogRoot "$Name.stderr.log"
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        if (Test-UiReady) {
            return
        }

        $Process.Refresh()
        if ($Process.HasExited) {
            $stderrTail = Get-LogTail -Path $stderr
            throw "UI server $Name exited before readiness. $stderrTail"
        }

        Start-Sleep -Milliseconds 750
    }

    $stderrTail = Get-LogTail -Path $stderr
    throw "Timed out waiting for $HealthUrl from UI server $Name. $stderrTail"
}

function Wait-ForUiStopped {
    param([string]$Description)

    $deadline = (Get-Date).AddSeconds(20)
    while ((Get-Date) -lt $deadline) {
        if (-not (Test-UiReady)) {
            return
        }
        Start-Sleep -Milliseconds 500
    }

    throw "Timed out waiting for $Description to stop serving $HealthUrl"
}

function Invoke-JsonApi {
    param(
        [ValidateSet("GET", "POST", "DELETE")]
        [string]$Method,
        [string]$Uri,
        [object]$Body = $null
    )

    $parameters = @{
        Method = $Method
        Uri = $Uri
        TimeoutSec = $TimeoutSeconds
    }

    if ($null -ne $Body) {
        $parameters.Body = ($Body | ConvertTo-Json -Depth 8)
        $parameters.ContentType = "application/json"
    }

    return Invoke-RestMethod @parameters
}

function Get-JobById {
    param([string]$JobId)

    $jobsResponse = Invoke-JsonApi -Method GET -Uri "$BaseUrl/api/jobs"
    $matches = @($jobsResponse.jobs | Where-Object { $_.id -eq $JobId } | Select-Object -First 1)
    if ($matches.Count -eq 0) {
        return $null
    }
    return $matches[0]
}

function Wait-ForJob {
    param(
        [string]$JobId,
        [scriptblock]$Predicate,
        [string]$Description
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    $lastJob = $null
    while ((Get-Date) -lt $deadline) {
        $lastJob = Get-JobById -JobId $JobId
        if ($null -ne $lastJob -and (& $Predicate $lastJob)) {
            return $lastJob
        }
        Start-Sleep -Milliseconds 750
    }

    if ($null -eq $lastJob) {
        throw "Timed out waiting for $Description. Last status: missing"
    }

    $details = $lastJob |
        Select-Object status, notes, runnerExitCode, runnerLogTail, errorTail |
        ConvertTo-Json -Depth 8 -Compress
    throw "Timed out waiting for $Description. Last job: $details"
}

function Wait-ForFile {
    param(
        [string]$Path,
        [string]$Description
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        if (Test-Path -LiteralPath $Path) {
            return
        }
        Start-Sleep -Milliseconds 500
    }

    throw "Timed out waiting for $Description at $Path"
}

function Get-JobsLedgerIds {
    if (-not (Test-Path -LiteralPath $JobsPath)) {
        return @()
    }

    $ledger = Get-Content -Raw -LiteralPath $JobsPath | ConvertFrom-Json
    return @($ledger.jobs | ForEach-Object { [string]$_.id })
}

function Assert-JobsLedgerPreserved {
    param([string[]]$ExpectedJobIds)

    $actualJobIds = @(Get-JobsLedgerIds)
    if ($actualJobIds.Count -ne $ExpectedJobIds.Count) {
        throw "jobs.json job count changed after UI restart recovery check: $($ExpectedJobIds.Count) -> $($actualJobIds.Count)"
    }

    $missingIds = @($ExpectedJobIds | Where-Object { $actualJobIds -notcontains $_ })
    if ($missingIds.Count -gt 0) {
        throw "jobs.json lost job ids after UI restart recovery check: $($missingIds -join ', ')"
    }
}

function Remove-SafeTree {
    param(
        [string]$Path,
        [string]$AllowedRoot
    )

    $resolvedRoot = (Resolve-Path -LiteralPath $AllowedRoot -ErrorAction SilentlyContinue)
    $resolvedTarget = (Resolve-Path -LiteralPath $Path -ErrorAction SilentlyContinue)
    if (-not $resolvedRoot -or -not $resolvedTarget) {
        return
    }

    foreach ($item in @($resolvedTarget)) {
        if ($item.Path.StartsWith($resolvedRoot.Path)) {
            Remove-Item -LiteralPath $item.Path -Recurse -Force -ErrorAction SilentlyContinue
        }
    }
}

function Remove-ProbeArtifacts {
    param(
        [string]$JobId,
        [string]$OutputDir
    )

    if ($OutputDir) {
        $outputPath = if ([System.IO.Path]::IsPathRooted($OutputDir)) { $OutputDir } else { Join-Path $ProjectRoot $OutputDir }
        Remove-SafeTree -Path $outputPath -AllowedRoot (Join-Path $ProjectRoot "trainer-data\probe-runs")
    }

    if ($JobId) {
        Remove-SafeTree `
            -Path (Join-Path $ProjectRoot "trainer-data\jobs\runner\$JobId") `
            -AllowedRoot (Join-Path $ProjectRoot "trainer-data\jobs\runner")
    }
}

function Cleanup-ProbeJob {
    param([string]$JobId)

    if (-not $JobId -or -not (Test-UiReady)) {
        return
    }

    $job = Get-JobById -JobId $JobId
    if ($null -eq $job) {
        return
    }

    if ($job.status -eq "running" -or $job.status -eq "queued") {
        Invoke-JsonApi -Method POST -Uri "$BaseUrl/api/jobs/$JobId/stop" | Out-Null
        Start-Sleep -Seconds 1
        $job = Get-JobById -JobId $JobId
    }

    if ($null -ne $job -and $job.status -ne "running" -and $job.status -ne "queued") {
        Invoke-JsonApi -Method DELETE -Uri "$BaseUrl/api/jobs/$JobId" | Out-Null
    }
}

$initialServer = $null
$restartedServer = $null
$createdJobId = $null
$createdOutputDir = $null
$preJobIds = @(Get-JobsLedgerIds)

try {
    if (Test-UiReady) {
        throw "Port $Port is already serving a UI. Use a free temporary port for this recovery check."
    }

    $initialServer = Start-UiServer -Name "initial"
    Wait-ForUiReady -Process $initialServer -Name "initial"

    $health = Invoke-JsonApi -Method GET -Uri "$BaseUrl/api/project"
    if ($null -eq $health.meta -or $health.meta.name -ne "D-OPSD Trainer") {
        throw "The UI at $BaseUrl is not the D-OPSD Trainer."
    }

    $createResponse = Invoke-JsonApi -Method POST -Uri "$BaseUrl/api/jobs" -Body @{
        probe = $true
        probeDurationSeconds = $ProbeDurationSeconds
    }
    $createdJobId = [string]$createResponse.job.id
    $createdOutputDir = [string]$createResponse.job.outputDir

    $startResponse = Invoke-JsonApi -Method POST -Uri "$BaseUrl/api/jobs/$createdJobId/start"
    if (-not $startResponse.ok) {
        throw "Failed to start runner probe: $($startResponse.error)"
    }

    $runningJob = Wait-ForJob -JobId $createdJobId -Description "runner probe to enter running state before UI restart" -Predicate {
        param($job)
        return $job.status -eq "running" -and $null -ne $job.runnerPid -and $null -ne $job.runner.linuxPidPath
    }
    $linuxPidHostPath = Join-Path $ProjectRoot ([string]$runningJob.runner.linuxPidPath)
    Wait-ForFile -Path $linuxPidHostPath -Description "Linux PID file before UI restart"

    Stop-UiServer -Process $initialServer -Description "initial UI server"

    $restartedServer = Start-UiServer -Name "restarted"
    Wait-ForUiReady -Process $restartedServer -Name "restarted"

    $restartedJob = Wait-ForJob -JobId $createdJobId -Description "same running job after full UI process restart" -Predicate {
        param($job)
        return $job.status -eq "running" -and $null -ne $job.runner -and $null -ne $job.runner.linuxPidPath
    }

    $completedJob = Wait-ForJob -JobId $createdJobId -Description "runner probe completion after UI restart replay" -Predicate {
        param($job)
        return $job.status -eq "completed" -and $job.runnerExitCode -eq 0
    }

    Cleanup-ProbeJob -JobId $createdJobId
    Remove-ProbeArtifacts -JobId $createdJobId -OutputDir $createdOutputDir
    Assert-JobsLedgerPreserved -ExpectedJobIds $preJobIds

    [pscustomobject]@{
        Ok = $true
        JobId = $createdJobId
        InitialServerProcessId = $initialServer.Id
        RestartedServerProcessId = $restartedServer.Id
        InitialStatus = $runningJob.status
        RestartedStatus = $restartedJob.status
        CompletedStatus = $completedJob.status
        RunnerExitCode = $completedJob.runnerExitCode
    } | Format-List
} finally {
    if ($createdJobId) {
        try {
            Cleanup-ProbeJob -JobId $createdJobId
            Remove-ProbeArtifacts -JobId $createdJobId -OutputDir $createdOutputDir
        } catch {
            Write-Warning "Probe cleanup failed: $($_.Exception.Message)"
        }
    }

    foreach ($server in @($initialServer, $restartedServer)) {
        if ($null -ne $server) {
            try {
                $server.Refresh()
                if (-not $server.HasExited) {
                    Stop-UiServer -Process $server -Description "temporary UI server"
                }
            } catch {
                Write-Warning "UI server cleanup failed: $($_.Exception.Message)"
            }
        }
    }
}
