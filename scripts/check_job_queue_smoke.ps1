[CmdletBinding()]
param(
    [string]$BaseUrl = "http://127.0.0.1:8675",
    [int]$FirstProbeDurationSeconds = 45,
    [int]$SecondProbeDurationSeconds = 8,
    [int]$TimeoutSeconds = 120
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$ProjectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$JobsPath = Join-Path $ProjectRoot "trainer-data\jobs\jobs.json"

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
        TimeoutSec = 20
    }

    if ($null -ne $Body) {
        $parameters.Body = ($Body | ConvertTo-Json -Depth 8)
        $parameters.ContentType = "application/json"
    }

    return Invoke-RestMethod @parameters
}

function Invoke-JsonApiStatus {
    param(
        [ValidateSet("GET", "POST", "DELETE")]
        [string]$Method,
        [string]$Uri,
        [object]$Body = $null
    )

    $parameters = @{
        Method = $Method
        Uri = $Uri
        TimeoutSec = 20
        UseBasicParsing = $true
    }

    if ($null -ne $Body) {
        $parameters.Body = ($Body | ConvertTo-Json -Depth 8)
        $parameters.ContentType = "application/json"
    }

    try {
        $response = Invoke-WebRequest @parameters
        return [pscustomobject]@{
            StatusCode = [int]$response.StatusCode
            Body = if ($response.Content) { $response.Content | ConvertFrom-Json } else { $null }
        }
    } catch {
        $statusCode = 0
        $content = if ($_.ErrorDetails) { $_.ErrorDetails.ToString() } else { "" }
        $response = $_.Exception.Response

        if ($response) {
            if ($response.StatusCode) {
                $statusCode = [int]$response.StatusCode
            }
            if (-not $content -and ($response | Get-Member -Name Content) -and $response.Content) {
                $content = $response.Content.ReadAsStringAsync().GetAwaiter().GetResult()
            }
            if (-not $content -and ($response | Get-Member -Name GetResponseStream)) {
                $reader = [System.IO.StreamReader]::new($response.GetResponseStream())
                try {
                    $content = $reader.ReadToEnd()
                } finally {
                    $reader.Dispose()
                }
            }
        }

        return [pscustomobject]@{
            StatusCode = $statusCode
            Body = if ($content) { $content | ConvertFrom-Json } else { $null }
        }
    }
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
        throw "jobs.json job count changed after queue smoke: $($ExpectedJobIds.Count) -> $($actualJobIds.Count)"
    }

    $missingIds = @($ExpectedJobIds | Where-Object { $actualJobIds -notcontains $_ })
    if ($missingIds.Count -gt 0) {
        throw "jobs.json lost job ids after queue smoke: $($missingIds -join ', ')"
    }
}

function Remove-SafeTree {
    param(
        [string]$Path,
        [string]$AllowedRoot
    )

    if (-not $Path) {
        return
    }

    $target = if ([System.IO.Path]::IsPathRooted($Path)) { $Path } else { Join-Path $ProjectRoot $Path }
    $resolvedRoot = Resolve-Path -LiteralPath $AllowedRoot -ErrorAction SilentlyContinue
    $resolvedTarget = Resolve-Path -LiteralPath $target -ErrorAction SilentlyContinue
    if (-not $resolvedRoot -or -not $resolvedTarget) {
        return
    }

    foreach ($item in @($resolvedTarget)) {
        if ($item.Path.StartsWith($resolvedRoot.Path)) {
            Remove-Item -LiteralPath $item.Path -Recurse -Force -ErrorAction SilentlyContinue
        }
    }
}

function Cleanup-Job {
    param([string]$JobId)

    if (-not $JobId) {
        return
    }

    $job = Get-JobById -JobId $JobId
    if ($null -eq $job) {
        return
    }

    if ($job.status -eq "running" -or $job.status -eq "queued") {
        Invoke-JsonApi -Method POST -Uri "$BaseUrl/api/jobs/$JobId/stop" | Out-Null
        Start-Sleep -Milliseconds 750
        $job = Get-JobById -JobId $JobId
    }

    if ($null -ne $job -and $job.status -ne "running" -and $job.status -ne "queued") {
        Invoke-JsonApiStatus -Method DELETE -Uri "$BaseUrl/api/jobs/$JobId" | Out-Null
    }

    if ($null -ne $job) {
        Remove-SafeTree -Path ([string]$job.outputDir) -AllowedRoot (Join-Path $ProjectRoot "trainer-data\probe-runs")
        Remove-SafeTree -Path (Join-Path $ProjectRoot "trainer-data\jobs\runner\$JobId") -AllowedRoot (Join-Path $ProjectRoot "trainer-data\jobs\runner")
    }
}

$firstJobId = $null
$secondJobId = $null
$clonedJobId = $null
$preJobIds = @(Get-JobsLedgerIds)

try {
    $health = Invoke-JsonApi -Method GET -Uri "$BaseUrl/api/project"
    if ($null -eq $health.meta -or $health.meta.name -ne "D-OPSD Trainer") {
        throw "The UI at $BaseUrl is not the D-OPSD Trainer."
    }

    $first = Invoke-JsonApi -Method POST -Uri "$BaseUrl/api/jobs" -Body @{
        probe = $true
        probeDurationSeconds = $FirstProbeDurationSeconds
    }
    $firstJobId = [string]$first.job.id

    $firstStart = Invoke-JsonApi -Method POST -Uri "$BaseUrl/api/jobs/$firstJobId/start"
    if (-not $firstStart.ok) {
        throw "Failed to start first runner probe: $($firstStart.error)"
    }

    $firstRunning = Wait-ForJob -JobId $firstJobId -Description "first runner probe to enter running state" -Predicate {
        param($job)
        return $job.status -eq "running" -and $null -ne $job.runnerPid
    }

    $second = Invoke-JsonApi -Method POST -Uri "$BaseUrl/api/jobs" -Body @{
        probe = $true
        probeDurationSeconds = $SecondProbeDurationSeconds
    }
    $secondJobId = [string]$second.job.id

    $secondStart = Invoke-JsonApi -Method POST -Uri "$BaseUrl/api/jobs/$secondJobId/start"
    if (-not $secondStart.ok) {
        throw "Failed to start second runner probe: $($secondStart.error)"
    }

    $secondQueued = Wait-ForJob -JobId $secondJobId -Description "second runner probe to enter queued state" -Predicate {
        param($job)
        return $job.status -eq "queued" -and $null -ne $job.queuedAt
    }

    $deleteActive = Invoke-JsonApiStatus -Method DELETE -Uri "$BaseUrl/api/jobs/$secondJobId"
    if ($deleteActive.StatusCode -ne 409 -or -not ([string]$deleteActive.Body.error).Contains("Stop the job before deleting it")) {
        throw "Queued job delete should fail with 'Stop the job before deleting it'; got HTTP $($deleteActive.StatusCode)"
    }

    $stopFirst = Invoke-JsonApi -Method POST -Uri "$BaseUrl/api/jobs/$firstJobId/stop"
    if (-not $stopFirst.ok) {
        throw "Failed to stop first runner probe: $($stopFirst.error)"
    }

    $firstStopped = Wait-ForJob -JobId $firstJobId -Description "first runner probe to stop" -Predicate {
        param($job)
        return $job.status -eq "stopped"
    }

    $secondRunning = Wait-ForJob -JobId $secondJobId -Description "auto-promoted second runner probe to enter running state" -Predicate {
        param($job)
        return $job.status -eq "running" -and $null -ne $job.runnerPid
    }

    $secondCompleted = Wait-ForJob -JobId $secondJobId -Description "second runner probe to complete after auto-promoted launch" -Predicate {
        param($job)
        return $job.status -eq "completed" -and $job.runnerExitCode -eq 0
    }

    $clone = Invoke-JsonApi -Method POST -Uri "$BaseUrl/api/jobs/$secondJobId/clone"
    if (-not $clone.ok -or $clone.job.status -ne "draft") {
        throw "Failed to clone completed probe job into a draft"
    }
    $clonedJobId = [string]$clone.job.id

    Cleanup-Job -JobId $clonedJobId
    $clonedJobId = $null
    Cleanup-Job -JobId $firstJobId
    $firstJobId = $null
    Cleanup-Job -JobId $secondJobId
    $secondJobId = $null
    Assert-JobsLedgerPreserved -ExpectedJobIds $preJobIds

    [pscustomobject]@{
        Ok = $true
        FirstJobId = $first.job.id
        FirstRunningStatus = $firstRunning.status
        FirstStoppedStatus = $firstStopped.status
        SecondJobId = $second.job.id
        SecondQueuedStatus = $secondQueued.status
        SecondRunningStatus = $secondRunning.status
        SecondCompletedStatus = $secondCompleted.status
        RunnerExitCode = $secondCompleted.runnerExitCode
        ClonedJobId = $clone.job.id
        ActiveDeleteStatusCode = $deleteActive.StatusCode
    } | Format-List
} finally {
    foreach ($jobId in @($clonedJobId, $secondJobId, $firstJobId)) {
        try {
            Cleanup-Job -JobId $jobId
        } catch {
            Write-Warning "Probe cleanup failed for ${jobId}: $($_.Exception.Message)"
        }
    }
}
