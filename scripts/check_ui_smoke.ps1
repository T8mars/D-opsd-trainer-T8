[CmdletBinding()]
param(
    [string]$BaseUrl = "http://127.0.0.1:8675",
    [int]$TimeoutSeconds = 20
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Invoke-Text {
    param([string]$Uri)

    $response = Invoke-WebRequest -UseBasicParsing -Uri $Uri -TimeoutSec $TimeoutSeconds
    if ([int]$response.StatusCode -lt 200 -or [int]$response.StatusCode -ge 300) {
        throw "Expected HTTP 2xx from $Uri, got $($response.StatusCode)"
    }
    return [string]$response.Content
}

function Invoke-Json {
    param([string]$Uri)

    return Invoke-RestMethod -Uri $Uri -TimeoutSec $TimeoutSeconds
}

function Assert-Contains {
    param(
        [string]$Text,
        [string]$Needle,
        [string]$Description
    )

    if (-not $Text.Contains($Needle)) {
        throw "$Description did not contain expected text: $Needle"
    }
}

function Join-Url {
    param(
        [string]$Root,
        [string]$Path
    )

    if ($Path.StartsWith("http://") -or $Path.StartsWith("https://")) {
        return $Path
    }

    return "$($Root.TrimEnd('/'))/$($Path.TrimStart('/'))"
}

function ConvertFrom-Utf8Base64 {
    param([string]$Value)

    return [System.Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($Value))
}

$project = Invoke-Json -Uri (Join-Url -Root $BaseUrl -Path "/api/project")
if ($null -eq $project.meta -or $project.meta.name -ne "D-OPSD Trainer" -or $project.meta.slug -ne "d-opsd-trainer") {
    throw "/api/project did not return D-OPSD Trainer metadata"
}

$models = Invoke-Json -Uri (Join-Url -Root $BaseUrl -Path "/api/models")
if (-not $models.ok) {
    throw "/api/models returned ok=false"
}
if (-not ($models.PSObject.Properties.Name -contains "customPaths")) {
    throw "/api/models did not include customPaths"
}
$defaultModelsCached = @($models.models | Where-Object { $_.spec.default -and $_.cached }).Count
if ($defaultModelsCached -lt 3) {
    throw "Expected 3 cached default models, got $defaultModelsCached"
}

$jobs = Invoke-Json -Uri (Join-Url -Root $BaseUrl -Path "/api/jobs")
if (-not $jobs.ok) {
    throw "/api/jobs returned ok=false"
}

$settings = Invoke-Json -Uri (Join-Url -Root $BaseUrl -Path "/api/settings")
if (-not $settings.ok) {
    throw "/api/settings returned ok=false"
}

$datasets = Invoke-Json -Uri (Join-Url -Root $BaseUrl -Path "/api/datasets")
if (-not $datasets.ok) {
    throw "/api/datasets returned ok=false"
}
$datasetReadyCount = @($datasets.datasets | Where-Object { $_.summary.ok }).Count
if ($datasetReadyCount -lt 1) {
    throw "Dataset ready preflight failed: expected at least one valid bundled dataset"
}

$brandTitle = ConvertFrom-Utf8Base64 "VDggRC1PUFNEIFRyYW5pZXI="
$zhNewTraining = ConvertFrom-Utf8Base64 "5paw5bu66K6t57uD"
$zhDashboard = ConvertFrom-Utf8Base64 "5Luq6KGo55uY"
$zhJobs = ConvertFrom-Utf8Base64 "5Lu75Yqh"
$zhDatasets = ConvertFrom-Utf8Base64 "5pWw5o2u6ZuG"
$zhModels = ConvertFrom-Utf8Base64 "5qih5Z6L"
$zhSettings = ConvertFrom-Utf8Base64 "6K6+572u"
$zhChinese = ConvertFrom-Utf8Base64 "5Lit5paH"
$zhPairPreflight = ConvertFrom-Utf8Base64 "6YWN5a+56aKE5qOA"
$zhCreateDraft = ConvertFrom-Utf8Base64 "5Yib5bu66I2J56i/"
$zhMemoryLaunch = ConvertFrom-Utf8Base64 "5pi+5a2Y5LiO5ZCv5Yqo"
$zhLowVramOffload = ConvertFrom-Utf8Base64 "5L2O5pi+5a2Y5Y246L29"
$zhRecommended16gb = ConvertFrom-Utf8Base64 "5o6o6I2QIDE2R0Ig6LW35q2l6YWN572u"
$zhSampleScale = ConvertFrom-Utf8Base64 "5qC35pys57yp5pS+"
$zhDatasetWeight = ConvertFrom-Utf8Base64 "6K6t57uD5p2D6YeN"
$zhSamplePrompts = ConvertFrom-Utf8Base64 "5qC35Zu+5o+Q56S66K+N"
$zhCommandPreview = ConvertFrom-Utf8Base64 "5ZG95Luk6aKE6KeI"
$zhDatasetBlocked = ConvertFrom-Utf8Base64 "5pWw5o2u6ZuG6Zi75aGe"
$zhDurableJobLedger = ConvertFrom-Utf8Base64 "5oyB5LmF5Lu75Yqh6LSm5pys"
$zhJobLedger = ConvertFrom-Utf8Base64 "5Lu75Yqh6LSm5pys"
$zhGpuTelemetry = ConvertFrom-Utf8Base64 "R1BVIOmBpea1iw=="
$zhRuntimePaths = ConvertFrom-Utf8Base64 "6L+Q6KGM5pe26Lev5b6E"
$zhDatasetValidator = ConvertFrom-Utf8Base64 "5pWw5o2u6ZuG6aqM6K+B5Zmo"
$zhDatasetPath = ConvertFrom-Utf8Base64 "5pWw5o2u6ZuG6Lev5b6E"
$zhImportDataset = ConvertFrom-Utf8Base64 "5a+85YWl5pWw5o2u6ZuG"
$zhUploadImages = ConvertFrom-Utf8Base64 "5LiK5Lyg5Zu+5YOP"
$zhCaptionFile = ConvertFrom-Utf8Base64 "5omT5qCH5paH5Lu2"
$zhMultiDatasetSelection = ConvertFrom-Utf8Base64 "5aSa6YCJ5pWw5o2u6ZuG"
$zhCreateDraftFromMerged = ConvertFrom-Utf8Base64 "5ZCI5bm25ZCO5Yib5bu66I2J56i/"
$zhModelCache = ConvertFrom-Utf8Base64 "5qih5Z6L57yT5a2Y"
$zhCustomModelPath = ConvertFrom-Utf8Base64 "6Ieq5a6a5LmJ5qih5Z6L6Lev5b6E"
$zhOpenFolder = ConvertFrom-Utf8Base64 "5omT5byA5paH5Lu25aS5"

$pages = @(
    @{ Path = "/"; Required = @($brandTitle, $zhNewTraining, $zhDashboard, $zhJobs, $zhDatasets, $zhModels, $zhSettings, $zhChinese, "EN") },
    @{ Path = "/jobs/new"; Required = @($zhNewTraining, $zhPairPreflight, $zhCreateDraft, $zhMultiDatasetSelection, $zhCreateDraftFromMerged, $zhDatasetWeight, $zhMemoryLaunch, $zhLowVramOffload, $zhRecommended16gb, $zhSampleScale, $zhSamplePrompts, $zhCommandPreview, $zhDatasetBlocked) },
    @{ Path = "/jobs"; Required = @($zhJobs, $zhDurableJobLedger, $zhGpuTelemetry, $zhJobLedger) },
    @{ Path = "/datasets"; Required = @($zhDatasets, "D-OPSD JSONL", $zhDatasetValidator, $zhDatasetPath, $zhImportDataset, $zhUploadImages, $zhCaptionFile, $zhMultiDatasetSelection, $zhCreateDraftFromMerged) },
    @{ Path = "/models"; Required = @($zhModels, "Hugging Face", $zhModelCache, $zhCustomModelPath, $zhOpenFolder) },
    @{ Path = "/settings"; Required = @($zhSettings, $zhRuntimePaths) }
)

$cssAssets = New-Object System.Collections.Generic.HashSet[string]
$pagesChecked = 0
foreach ($page in $pages) {
    $path = [string]$page.Path
    $html = Invoke-Text -Uri (Join-Url -Root $BaseUrl -Path $path)
    Assert-Contains -Text $html -Needle "__next_f.push" -Description "Page $path"
    Assert-Contains -Text $html -Needle "/_next/static/css/" -Description "Page $path"

    foreach ($requiredText in @($page.Required)) {
        Assert-Contains -Text $html -Needle ([string]$requiredText) -Description "Page $path"
    }

    foreach ($match in [regex]::Matches($html, 'href="(?<asset>/_next/static/css/[^"]+)"')) {
        $null = $cssAssets.Add($match.Groups["asset"].Value.Replace("\u0026", "&"))
    }
    $pagesChecked += 1
}

if ($cssAssets.Count -eq 0) {
    throw "No Next CSS assets were found in checked pages"
}

$cssChecked = 0
foreach ($asset in $cssAssets) {
    $css = Invoke-Text -Uri (Join-Url -Root $BaseUrl -Path $asset)
    Assert-Contains -Text $css -Needle ".glass" -Description "CSS asset $asset"
    Assert-Contains -Text $css -Needle "backdrop-filter" -Description "CSS asset $asset"
    $cssChecked += 1
}

[pscustomobject]@{
    Ok = $true
    BaseUrl = $BaseUrl
    PagesChecked = $pagesChecked
    CssAssets = $cssChecked
    DefaultModelsCached = $defaultModelsCached
    Jobs = @($jobs.jobs).Count
    Datasets = @($datasets.datasets).Count
    DatasetReady = $datasetReadyCount
} | Format-List
