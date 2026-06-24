[CmdletBinding()]
param(
    [string]$BaseUrl = "http://127.0.0.1:8675",
    [int]$UiTimeoutSeconds = 30,
    [switch]$SkipLiveQueue,
    [switch]$SkipTrainingObservability,
    [switch]$SkipRunnerRecovery
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

if ($PSVersionTable.PSEdition -ne "Core") {
    throw "check_release_readiness.ps1 must be run with PowerShell 7 (pwsh.exe) so WSL distro visibility matches the launcher host."
}

$ProjectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
Set-Location -LiteralPath $ProjectRoot

$results = New-Object System.Collections.Generic.List[object]

function Add-Result {
    param(
        [string]$Name,
        [string]$Status,
        [string]$Details = ""
    )

    $results.Add([pscustomobject]@{
        Name = $Name
        Status = $Status
        Details = $Details
    }) | Out-Null
}

function Get-OutputTail {
    param([object[]]$Output)

    $lines = @($Output | ForEach-Object { [string]$_ })
    if ($lines.Count -eq 0) {
        return ""
    }
    return ($lines | Select-Object -Last 20) -join "`n"
}

function Invoke-CheckedCommand {
    param(
        [string]$Name,
        [string]$FilePath,
        [string[]]$ArgumentList,
        [int[]]$AllowedExitCodes = @(0)
    )

    $global:LASTEXITCODE = 0
    $previousErrorActionPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        $output = & $FilePath @ArgumentList 2>&1
    } finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }
    $exitCode = if ($null -eq $global:LASTEXITCODE) { 0 } else { [int]$global:LASTEXITCODE }
    $tail = Get-OutputTail -Output $output

    if ($AllowedExitCodes -notcontains $exitCode) {
        throw "Check '$Name' failed with exit code $exitCode.`n$tail"
    }

    Add-Result -Name $Name -Status "passed" -Details $tail
    return $output
}

function Assert-PowerShellParses {
    param([string[]]$Files)

    foreach ($file in $Files) {
        $errors = $null
        [System.Management.Automation.PSParser]::Tokenize((Get-Content -Raw -LiteralPath $file), [ref]$errors) | Out-Null
        if ($errors) {
            $detail = ($errors | Format-List | Out-String)
            throw "PowerShell parse failed for ${file}:`n$detail"
        }
    }
    Add-Result -Name "PowerShell parser checks" -Status "passed" -Details ($Files -join ", ")
}

function Assert-NoTrailingWhitespace {
    param([string[]]$Files)

    $matches = & rg -n "[ \t]+$" @Files 2>&1
    $exitCode = if ($null -eq $global:LASTEXITCODE) { 0 } else { [int]$global:LASTEXITCODE }
    if ($exitCode -eq 0) {
        throw "Trailing whitespace found:`n$($matches -join "`n")"
    }
    if ($exitCode -ne 1) {
        throw "Trailing whitespace check failed with exit code $exitCode.`n$($matches -join "`n")"
    }
    Add-Result -Name "Trailing whitespace checks" -Status "passed" -Details ($Files -join ", ")
}

function Assert-PackagedWorkspaceTemplateFresh {
    $templateRoot = Join-Path $ProjectRoot "trainer-ui\release\win-unpacked\resources\workspace-template"
    if (-not (Test-Path -LiteralPath $templateRoot)) {
        Add-Result -Name "Packaged workspace template freshness" -Status "passed" -Details "No local packaged workspace template present"
        return
    }

    function Assert-TemplateFile {
        param(
            [string]$RelativePath,
            [string[]]$MustContain = @(),
            [string[]]$MustNotContain = @()
        )

        $file = Join-Path $templateRoot $RelativePath
        if (-not (Test-Path -LiteralPath $file)) {
            throw "Packaged workspace template is missing ${RelativePath}"
        }

        $source = Get-Content -Raw -LiteralPath $file
        foreach ($needle in $MustContain) {
            if (-not $source.Contains($needle)) {
                throw "Packaged workspace template ${RelativePath} is stale; missing expected text: ${needle}"
            }
        }
        foreach ($needle in $MustNotContain) {
            if ($source.Contains($needle)) {
                throw "Packaged workspace template ${RelativePath} is stale; found forbidden text: ${needle}"
            }
        }
    }

    foreach ($relativePath in @(
        "flux2-klein_self-distill-edit\arguments.py",
        "flux2-klein-edit-self-distill-gt-ref\arguments.py",
        "z-image-turbo_self-distill-vlm\arguments.py"
    )) {
        Assert-TemplateFile -RelativePath $relativePath `
            -MustContain @(
                '--block-offload-num-blocks", type=int, default=1',
                '--layer-offload-transformer-percent", type=float, default=1.0',
                '--layer-offload-text-encoder-percent", type=float, default=1.0'
            ) `
            -MustNotContain @('--block-offload-num-blocks", type=int, default=2')
    }

    foreach ($relativePath in @(
        "scripts\run_flux2_smoke.sh",
        "scripts\run_flux2_editing_smoke.sh",
        "scripts\run_zimage_smoke.sh"
    )) {
        Assert-TemplateFile -RelativePath $relativePath `
            -MustContain @(
                'BLOCK_OFFLOAD_NUM_BLOCKS:-1',
                'LAYER_OFFLOAD_TRANSFORMER_PERCENT:-1.0',
                'LAYER_OFFLOAD_TEXT_ENCODER_PERCENT:-1.0'
            ) `
            -MustNotContain @('BLOCK_OFFLOAD_NUM_BLOCKS:-2')
    }

    Assert-TemplateFile -RelativePath "flux2-klein_self-distill-edit\train_dopsd.py" `
        -MustContain @(
            '"final_sampler_cpu_offload": bool(args.final_sampler_cpu_offload) and not bool(args.block_offload)',
            '"block_offload": bool(args.block_offload)',
            '"block_offload_num_blocks": int(args.block_offload_num_blocks)',
            'attach_layer_offload(',
            'AI Toolkit-style layer offloading enabled'
        ) `
        -MustNotContain @(
            '"final_sampler_cpu_offload": bool(args.final_sampler_cpu_offload),',
            '"block_offload": False',
            '"block_offload_num_blocks": 0'
        )

    Assert-TemplateFile -RelativePath "flux2-klein_self-distill-edit\sample_flux2_final.py" `
        -MustContain @(
            'request.get("block_offload_num_blocks", 1)',
            'request.get("block_offload") and is_cuda_oom(error)',
            '"block_offload": False',
            '"final_sampler_cpu_offload": True'
        ) `
        -MustNotContain @('request.get("block_offload_num_blocks", 2)')

    Assert-TemplateFile -RelativePath "trainer_runtime\dopsd_trainer\cli.py" `
        -MustContain @(
            'command_parser.add_argument("--block-offload-num-blocks", type=int, default=1)',
            'command_parser.add_argument("--layer-offload-transformer-percent", type=float, default=1.0)'
        ) `
        -MustNotContain @('command_parser.add_argument("--block-offload-num-blocks", type=int, default=2)')

    Assert-TemplateFile -RelativePath "trainer_runtime\dopsd_trainer\recipes.py" `
        -MustContain @(
            'block_offload_num_blocks: int = 1',
            'layer_offload_transformer_percent: float = 1.0'
        ) `
        -MustNotContain @('block_offload_num_blocks: int = 2')

    Assert-TemplateFile -RelativePath "trainer_runtime\dopsd_trainer\profiles.py" `
        -MustContain @('"block_offload_num_blocks": 1') `
        -MustNotContain @('"block_offload_num_blocks": 2')

    foreach ($privateDoc in @("SKILL.md", "skill.md", "roadmap.md", "ROADMAP.md")) {
        if (Test-Path -LiteralPath (Join-Path $templateRoot $privateDoc)) {
            throw "Packaged workspace template must not include ${privateDoc}"
        }
    }

    Add-Result -Name "Packaged workspace template freshness" -Status "passed" -Details $templateRoot
}

function Assert-NoWslProcess {
    param(
        [string]$Name,
        [string]$Pattern
    )

    $output = & wsl -d Ubuntu-22.04 -- pgrep -af $Pattern 2>&1
    $exitCode = if ($null -eq $global:LASTEXITCODE) { 0 } else { [int]$global:LASTEXITCODE }
    if ($exitCode -eq 0) {
        throw "Unexpected WSL process matching ${Pattern}:`n$($output -join "`n")"
    }
    if ($exitCode -ne 1) {
        throw "Process check '$Name' failed with exit code $exitCode.`n$($output -join "`n")"
    }
    Add-Result -Name $Name -Status "passed" -Details "No process matched $Pattern"
}

# Gate command strings are kept literal so tests and humans can audit the release contract:
# python -m unittest discover -s trainer_runtime\tests -v
# npm run typecheck --prefix trainer-ui
# python -m json.tool features.json
# python -m json.tool meta.json
# git -c safe.directory=E:/D-opsd-T8-Tranier diff --check
# pgrep -af train_dopsd
# pgrep -af accelerate
# pgrep -af deepspeed

Invoke-CheckedCommand -Name "Python runtime tests" -FilePath "python" -ArgumentList @("-m", "unittest", "discover", "-s", "trainer_runtime\tests", "-v") | Out-Null
Invoke-CheckedCommand -Name "TypeScript typecheck" -FilePath "npm.cmd" -ArgumentList @("run", "typecheck", "--prefix", "trainer-ui") | Out-Null
Invoke-CheckedCommand -Name "features.json parses" -FilePath "python" -ArgumentList @("-m", "json.tool", "features.json") | Out-Null
Invoke-CheckedCommand -Name "meta.json parses" -FilePath "python" -ArgumentList @("-m", "json.tool", "meta.json") | Out-Null
Invoke-CheckedCommand -Name "git diff whitespace" -FilePath "git" -ArgumentList @("-c", "safe.directory=E:/D-opsd-T8-Tranier", "diff", "--check") | Out-Null

Assert-PowerShellParses -Files @(
    "scripts\check_ui_smoke.ps1",
    "scripts\check_job_queue_smoke.ps1",
    "scripts\check_runner_recovery.ps1",
    "scripts\check_training_observability.ps1",
    "scripts\check_production_profiles.ps1",
    "scripts\check_ui_restart_recovery.ps1",
    "scripts\check_release_readiness.ps1",
    "scripts\start_trainer.ps1",
    "scripts\run_ui_server.ps1"
)

Invoke-CheckedCommand -Name "FLUX2 smoke shell syntax" -FilePath "wsl" -ArgumentList @("-d", "Ubuntu-22.04", "--cd", "/mnt/e/D-opsd-T8-Tranier", "--", "bash", "-n", "scripts/run_flux2_smoke.sh") | Out-Null
Invoke-CheckedCommand -Name "FLUX2 Editing smoke shell syntax" -FilePath "wsl" -ArgumentList @("-d", "Ubuntu-22.04", "--cd", "/mnt/e/D-opsd-T8-Tranier", "--", "bash", "-n", "scripts/run_flux2_editing_smoke.sh") | Out-Null
Invoke-CheckedCommand -Name "Z-Image smoke shell syntax" -FilePath "wsl" -ArgumentList @("-d", "Ubuntu-22.04", "--cd", "/mnt/e/D-opsd-T8-Tranier", "--", "bash", "-n", "scripts/run_zimage_smoke.sh") | Out-Null

Assert-NoTrailingWhitespace -Files @(
    "features.json",
    "scripts\check_release_readiness.ps1",
    "scripts\check_ui_smoke.ps1",
    "scripts\check_job_queue_smoke.ps1",
    "scripts\check_training_observability.ps1",
    "scripts\check_production_profiles.ps1",
    "trainer_runtime\tests\test_runtime.py"
)

Assert-PackagedWorkspaceTemplateFresh

Invoke-CheckedCommand -Name "Production profile contract" -FilePath "powershell" -ArgumentList @(
    "-NoProfile",
    "-ExecutionPolicy",
    "Bypass",
    "-File",
    "scripts\check_production_profiles.ps1"
) | Out-Null

Invoke-CheckedCommand -Name "UI smoke" -FilePath "powershell" -ArgumentList @(
    "-NoProfile",
    "-ExecutionPolicy",
    "Bypass",
    "-File",
    "scripts\check_ui_smoke.ps1",
    "-BaseUrl",
    $BaseUrl,
    "-TimeoutSeconds",
    ([string]$UiTimeoutSeconds)
) | Out-Null

if ($SkipTrainingObservability) {
    Add-Result -Name "Training observability smoke" -Status "skipped" -Details "Skipped by -SkipTrainingObservability"
} else {
    Invoke-CheckedCommand -Name "Training observability smoke" -FilePath "powershell" -ArgumentList @(
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        "scripts\check_training_observability.ps1",
        "-BaseUrl",
        $BaseUrl,
        "-ProbeDurationSeconds",
        "12",
        "-TimeoutSeconds",
        "90"
    ) | Out-Null
}

if ($SkipLiveQueue) {
    Add-Result -Name "Live queue smoke" -Status "skipped" -Details "Skipped by -SkipLiveQueue"
} else {
    Invoke-CheckedCommand -Name "Live queue smoke" -FilePath "powershell" -ArgumentList @(
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        "scripts\check_job_queue_smoke.ps1",
        "-BaseUrl",
        $BaseUrl,
        "-FirstProbeDurationSeconds",
        "35",
        "-SecondProbeDurationSeconds",
        "6",
        "-TimeoutSeconds",
        "120"
    ) | Out-Null
}

if ($SkipRunnerRecovery) {
    Add-Result -Name "Runner recovery smoke" -Status "skipped" -Details "Skipped by -SkipRunnerRecovery"
} else {
    Invoke-CheckedCommand -Name "Runner recovery smoke" -FilePath "powershell" -ArgumentList @(
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        "scripts\check_runner_recovery.ps1",
        "-BaseUrl",
        $BaseUrl,
        "-ProbeDurationSeconds",
        "20",
        "-TimeoutSeconds",
        "90"
    ) | Out-Null
}

Assert-NoWslProcess -Name "No train_dopsd process" -Pattern "train_dopsd"
Assert-NoWslProcess -Name "No accelerate process" -Pattern "accelerate"
Assert-NoWslProcess -Name "No deepspeed process" -Pattern "deepspeed"

$gpuMemory = Invoke-CheckedCommand -Name "GPU memory query" -FilePath "wsl" -ArgumentList @(
    "-d",
    "Ubuntu-22.04",
    "--",
    "nvidia-smi",
    "--query-gpu=memory.used,memory.total",
    "--format=csv,noheader"
)

[pscustomobject]@{
    Ok = $true
    BaseUrl = $BaseUrl
    Checks = $results.Count
    LiveQueueSkipped = [bool]$SkipLiveQueue
    TrainingObservabilitySkipped = [bool]$SkipTrainingObservability
    RunnerRecoverySkipped = [bool]$SkipRunnerRecovery
    GpuMemory = (($gpuMemory | ForEach-Object { [string]$_ }) -join "`n")
    Results = $results
} | Format-List
