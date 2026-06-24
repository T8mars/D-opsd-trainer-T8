[CmdletBinding()]
param(
    [string]$BaseUrl = "http://127.0.0.1:8675",
    [int]$ProbeDurationSeconds = 12,
    [int]$TimeoutSeconds = 90
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$ProjectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$JobsPath = Join-Path $ProjectRoot "trainer-data\jobs\jobs.json"
$RequiredSseEventLines = @("event: snapshot", "event: append", "event: heartbeat")

function Join-Url {
    param(
        [string]$Root,
        [string]$Path
    )

    return "$($Root.TrimEnd('/'))/$($Path.TrimStart('/'))"
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
        TimeoutSec = 20
    }

    if ($null -ne $Body) {
        $parameters.Body = ($Body | ConvertTo-Json -Depth 8)
        $parameters.ContentType = "application/json"
    }

    return Invoke-RestMethod @parameters
}

function Invoke-TextRequest {
    param([string]$Uri)

    $response = Invoke-WebRequest -UseBasicParsing -Uri $Uri -TimeoutSec 20
    if ([int]$response.StatusCode -lt 200 -or [int]$response.StatusCode -ge 300) {
        throw "Expected HTTP 2xx from $Uri, got $($response.StatusCode)"
    }
    return [string]$response.Content
}

function Get-JobById {
    param([string]$JobId)

    $jobsResponse = Invoke-JsonApi -Method GET -Uri (Join-Url -Root $BaseUrl -Path "/api/jobs")
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
        throw "jobs.json job count changed after training observability smoke: $($ExpectedJobIds.Count) -> $($actualJobIds.Count)"
    }

    $missingIds = @($ExpectedJobIds | Where-Object { $actualJobIds -notcontains $_ })
    if ($missingIds.Count -gt 0) {
        throw "jobs.json lost job ids after training observability smoke: $($missingIds -join ', ')"
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
        Invoke-JsonApi -Method POST -Uri (Join-Url -Root $BaseUrl -Path "/api/jobs/$JobId/stop") | Out-Null
        Start-Sleep -Milliseconds 750
        $job = Get-JobById -JobId $JobId
    }

    if ($null -ne $job -and $job.status -ne "running" -and $job.status -ne "queued") {
        Invoke-JsonApi -Method DELETE -Uri (Join-Url -Root $BaseUrl -Path "/api/jobs/$JobId") | Out-Null
    }

    if ($null -ne $job) {
        Remove-SafeTree -Path ([string]$job.outputDir) -AllowedRoot (Join-Path $ProjectRoot "trainer-data\probe-runs")
        Remove-SafeTree -Path (Join-Path $ProjectRoot "trainer-data\jobs\runner\$JobId") -AllowedRoot (Join-Path $ProjectRoot "trainer-data\jobs\runner")
    }
}

function Read-SseEvents {
    param(
        [string]$Uri,
        [int]$StreamTimeoutSeconds
    )

    $request = [System.Net.HttpWebRequest][System.Net.WebRequest]::Create($Uri)
    $request.Method = "GET"
    $request.Accept = "text/event-stream"
    $request.Timeout = 10000
    $request.ReadWriteTimeout = 6000

    $response = $request.GetResponse()
    $reader = $null
    $events = New-Object System.Collections.Generic.List[string]
    $rawLines = New-Object System.Collections.Generic.List[string]

    try {
        $contentType = [string]$response.ContentType
        if (-not $contentType.Contains("text/event-stream")) {
            throw "Expected text/event-stream from $Uri, got $contentType"
        }

        $reader = [System.IO.StreamReader]::new($response.GetResponseStream())
        $deadline = (Get-Date).AddSeconds($StreamTimeoutSeconds)
        while ((Get-Date) -lt $deadline) {
            try {
                $line = $reader.ReadLine()
            } catch [System.IO.IOException] {
                break
            }

            if ($null -eq $line) {
                break
            }

            $rawLines.Add($line) | Out-Null
            if ($line.StartsWith("event: ")) {
                $events.Add($line.Substring("event: ".Length)) | Out-Null
            }

            if ($events.Contains("snapshot") -and $events.Contains("heartbeat") -and $events.Contains("append")) {
                break
            }
        }
    } finally {
        if ($null -ne $reader) {
            $reader.Dispose()
        }
        $response.Close()
    }

    foreach ($requiredLine in $RequiredSseEventLines) {
        $required = $requiredLine.Substring("event: ".Length)
        if (-not $events.Contains($required)) {
            throw "SSE stream did not include $requiredLine. Raw stream:`n$([string]::Join("`n", $rawLines))"
        }
    }

    return [pscustomobject]@{
        Events = @($events)
        Raw = [string]::Join("`n", $rawLines)
    }
}

function Assert-LogSource {
    param(
        [object[]]$Lines,
        [string]$Source,
        [string]$ContainsText
    )

    $matches = @($Lines | Where-Object { $_.source -eq $Source -and ([string]$_.line).Contains($ContainsText) })
    if ($matches.Count -eq 0) {
        throw "Combined logs did not contain $Source line with text '$ContainsText'"
    }
}

function Assert-Artifact {
    param(
        [string]$JobId,
        [string]$RelativePath
    )

    $artifactUri = Join-Url -Root $BaseUrl -Path "/api/jobs/$JobId/artifact?path=$([System.Uri]::EscapeDataString($RelativePath))"
    $content = Invoke-TextRequest -Uri $artifactUri
    if ($content.Length -eq 0) {
        throw "Artifact endpoint returned empty content for $RelativePath"
    }
}

$createdJobId = $null
$preJobIds = @(Get-JobsLedgerIds)

try {
    $health = Invoke-JsonApi -Method GET -Uri (Join-Url -Root $BaseUrl -Path "/api/project")
    if ($null -eq $health.meta -or $health.meta.name -ne "D-OPSD Trainer") {
        throw "The UI at $BaseUrl is not the D-OPSD Trainer."
    }

    $createResponse = Invoke-JsonApi -Method POST -Uri (Join-Url -Root $BaseUrl -Path "/api/jobs") -Body @{
        probe = $true
        probeDurationSeconds = $ProbeDurationSeconds
    }
    $createdJobId = [string]$createResponse.job.id

    $startResponse = Invoke-JsonApi -Method POST -Uri (Join-Url -Root $BaseUrl -Path "/api/jobs/$createdJobId/start")
    if (-not $startResponse.ok) {
        throw "Failed to start observability runner probe: $($startResponse.error)"
    }

    $runningJob = Wait-ForJob -JobId $createdJobId -Description "observability probe to enter running state" -Predicate {
        param($job)
        return $job.status -eq "running" -and $null -ne $job.runnerPid
    }

    $sse = Read-SseEvents `
        -Uri (Join-Url -Root $BaseUrl -Path "/api/jobs/$createdJobId/logs/stream") `
        -StreamTimeoutSeconds ($ProbeDurationSeconds + 30)

    $completedJob = Wait-ForJob -JobId $createdJobId -Description "observability probe completion" -Predicate {
        param($job)
        return $job.status -eq "completed" -and $job.runnerExitCode -eq 0
    }

    if ($completedJob.latestStep -ne 0 -or $completedJob.latestLoss -ne 0) {
        throw "Probe loss parsing mismatch: step=$($completedJob.latestStep), loss=$($completedJob.latestLoss)"
    }

    if ($completedJob.artifactCounts.samples -lt 1 -or $completedJob.artifactCounts.sampleTrajectories -lt 1 -or $completedJob.artifactCounts.checkpoints -lt 1) {
        throw "Probe artifact counts were incomplete"
    }

    $samplePaths = @($completedJob.artifactItems.samples | ForEach-Object { [string]$_.relativePath })
    $trajectoryPaths = @($completedJob.artifactItems.sampleTrajectories | ForEach-Object { [string]$_.relativePath })
    $checkpointPaths = @($completedJob.artifactItems.checkpoints | ForEach-Object { [string]$_.relativePath })
    if ($samplePaths -notcontains "samples/probe-sample.png") {
        throw "Probe sample artifact was not listed"
    }
    if ($trajectoryPaths -notcontains "samples_trajectory/probe-trajectory.png") {
        throw "Probe trajectory artifact was not listed"
    }
    if ($checkpointPaths -notcontains "checkpoints/probe-adapter.safetensors") {
        throw "Probe checkpoint artifact was not listed"
    }

    Assert-Artifact -JobId $createdJobId -RelativePath "samples/probe-sample.png"
    Assert-Artifact -JobId $createdJobId -RelativePath "samples_trajectory/probe-trajectory.png"
    Assert-Artifact -JobId $createdJobId -RelativePath "checkpoints/probe-adapter.safetensors"

    $logs = Invoke-JsonApi -Method GET -Uri (Join-Url -Root $BaseUrl -Path "/api/jobs/$createdJobId/logs")
    if (-not $logs.ok) {
        throw "/logs returned ok=false for probe job"
    }
    Assert-LogSource -Lines @($logs.combined) -Source "runner" -ContainsText "[runner] started"
    Assert-LogSource -Lines @($logs.combined) -Source "training" -ContainsText "Probe started"
    Assert-LogSource -Lines @($logs.combined) -Source "training" -ContainsText "Training completed"

    $telemetry = Invoke-JsonApi -Method GET -Uri (Join-Url -Root $BaseUrl -Path "/api/telemetry")
    if (-not $telemetry.ok -or $null -eq $telemetry.gpu) {
        throw "/api/telemetry returned an invalid payload"
    }
    if (-not $telemetry.gpu.available) {
        throw "/api/telemetry did not report an available GPU"
    }

    Cleanup-Job -JobId $createdJobId
    $createdJobId = $null
    Assert-JobsLedgerPreserved -ExpectedJobIds $preJobIds

    [pscustomobject]@{
        Ok = $true
        JobId = $createResponse.job.id
        RunningStatus = $runningJob.status
        CompletedStatus = $completedJob.status
        RunnerExitCode = $completedJob.runnerExitCode
        SseEvents = @($sse.Events) -join ","
        ArtifactCounts = $completedJob.artifactCounts
        TelemetryGpu = $telemetry.gpu.gpus[0].name
    } | Format-List
} finally {
    if ($createdJobId) {
        try {
            Cleanup-Job -JobId $createdJobId
        } catch {
            Write-Warning "Observability probe cleanup failed for ${createdJobId}: $($_.Exception.Message)"
        }
    }
}
