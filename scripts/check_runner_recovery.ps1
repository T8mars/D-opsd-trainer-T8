[CmdletBinding()]
param(
    [string]$BaseUrl = "http://127.0.0.1:8675",
    [int]$ProbeDurationSeconds = 20,
    [int]$TimeoutSeconds = 90
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$ProjectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path

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
        TimeoutSec = 15
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

    $status = if ($null -ne $lastJob) { $lastJob.status } else { "missing" }
    throw "Timed out waiting for $Description. Last status: $status"
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

function Restore-JobsLedgerBackup {
    param(
        [string]$JobsPath,
        [string]$BackupPath
    )

    if (Test-Path -LiteralPath $BackupPath) {
        Copy-Item -LiteralPath $BackupPath -Destination $JobsPath -Force
    }
}

function Assert-JobsLedgerPreserved {
    param(
        [string[]]$preJobIds,
        [object]$AfterLedger
    )

    $postJobIds = @($AfterLedger.jobs | ForEach-Object { [string]$_.id })
    if ($postJobIds.Count -ne $preJobIds.Count) {
        throw "jobs.json job count changed during stale PID injection: $($preJobIds.Count) -> $($postJobIds.Count)"
    }

    $missingIds = @($preJobIds | Where-Object { $postJobIds -notcontains $_ })
    if ($missingIds.Count -gt 0) {
        throw "jobs.json lost job ids during stale PID injection: $($missingIds -join ', ')"
    }
}

function Write-Utf8NoBom {
    param(
        [string]$Path,
        [string]$Content
    )

    $encoding = [System.Text.UTF8Encoding]::new($false)
    [System.IO.File]::WriteAllText($Path, $Content, $encoding)
}

function Set-StaleRunnerPid {
    param(
        [string]$JobId,
        [int]$StalePid = 999999
    )

    $jobsPath = Join-Path $ProjectRoot "trainer-data\jobs\jobs.json"
    $backupPath = Join-Path (Split-Path -Parent $jobsPath) "jobs.json.recovery-check.bak"
    if (-not (Test-Path -LiteralPath $jobsPath)) {
        throw "jobs.json was not found at $jobsPath"
    }

    $ledger = Get-Content -Raw -LiteralPath $jobsPath | ConvertFrom-Json
    $preJobIds = @($ledger.jobs | ForEach-Object { [string]$_.id })
    $jobMatches = @($ledger.jobs | Where-Object { $_.id -eq $JobId } | Select-Object -First 1)
    if ($jobMatches.Count -eq 0) {
        throw "Job $JobId was not found in jobs.json"
    }
    $job = $jobMatches[0]
    if ($null -eq $job.runner) {
        throw "Job $JobId does not have runner metadata"
    }

    Copy-Item -LiteralPath $jobsPath -Destination $backupPath -Force
    $job.runner.windowsPid = $StalePid
    $tempPath = "$jobsPath.recovery-check.tmp"
    try {
        $ledgerJson = ($ledger | ConvertTo-Json -Depth 32) + [Environment]::NewLine
        Write-Utf8NoBom -Path $tempPath -Content $ledgerJson
        Move-Item -LiteralPath $tempPath -Destination $jobsPath -Force
        $afterLedger = Get-Content -Raw -LiteralPath $jobsPath | ConvertFrom-Json
        Assert-JobsLedgerPreserved -preJobIds $preJobIds -AfterLedger $afterLedger
    } catch {
        Restore-JobsLedgerBackup -JobsPath $jobsPath -BackupPath $backupPath
        throw
    }
}

$createdJobId = $null

try {
    $health = Invoke-JsonApi -Method GET -Uri "$BaseUrl/api/project"
    if ($null -eq $health.meta -or $health.meta.name -ne "D-OPSD Trainer") {
        throw "The UI at $BaseUrl is not the D-OPSD Trainer."
    }

    $createResponse = Invoke-JsonApi -Method POST -Uri "$BaseUrl/api/jobs" -Body @{
        probe = $true
        probeDurationSeconds = $ProbeDurationSeconds
    }
    $createdJobId = $createResponse.job.id

    $startResponse = Invoke-JsonApi -Method POST -Uri "$BaseUrl/api/jobs/$createdJobId/start"
    if (-not $startResponse.ok) {
        throw "Failed to start runner probe: $($startResponse.error)"
    }

    $runningJob = Wait-ForJob -JobId $createdJobId -Description "runner probe to enter running state" -Predicate {
        param($job)
        return $job.status -eq "running" -and $null -ne $job.runnerPid -and $null -ne $job.runner.linuxPidPath
    }

    $runnerPid = [int]$runningJob.runnerPid
    $linuxPidPath = [string]$runningJob.runner.linuxPidPath
    $linuxPidHostPath = Join-Path $ProjectRoot $linuxPidPath
    Wait-ForFile -Path $linuxPidHostPath -Description "Linux PID file"
    Set-StaleRunnerPid -JobId $createdJobId

    $recoveredJob = Wait-ForJob -JobId $createdJobId -Description "Linux PID recovery after Windows monitor exit" -Predicate {
        param($job)
        $notes = [string]$job.notes
        return $job.status -eq "running" -and $notes.Contains("Runner process monitor detached")
    }

    $completedJob = Wait-ForJob -JobId $createdJobId -Description "runner probe completion after recovery" -Predicate {
        param($job)
        return $job.status -eq "completed" -and $job.runnerExitCode -eq 0
    }

    [pscustomobject]@{
        Ok = $true
        JobId = $createdJobId
        LinuxPidPath = $linuxPidPath
        RecoveredStatus = $recoveredJob.status
        CompletedStatus = $completedJob.status
        RunnerExitCode = $completedJob.runnerExitCode
        RecoveryNote = $recoveredJob.notes
    } | Format-List
} finally {
    if ($createdJobId) {
        try {
            $job = Get-JobById -JobId $createdJobId
            if ($null -ne $job -and ($job.status -eq "running" -or $job.status -eq "queued")) {
                Invoke-JsonApi -Method POST -Uri "$BaseUrl/api/jobs/$createdJobId/stop" | Out-Null
            }
            if ($null -ne $job -and ($job.status -ne "running" -and $job.status -ne "queued")) {
                Invoke-JsonApi -Method DELETE -Uri "$BaseUrl/api/jobs/$createdJobId" | Out-Null
            }
        } catch {
            Write-Warning "Probe cleanup stop failed: $($_.Exception.Message)"
        }
    }
}
