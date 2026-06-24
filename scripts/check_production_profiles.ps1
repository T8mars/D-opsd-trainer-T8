[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ProjectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
Set-Location -LiteralPath $ProjectRoot

function Invoke-JsonCommand {
    param(
        [string]$Name,
        [string[]]$ArgumentList
    )

    $global:LASTEXITCODE = 0
    $output = & python @ArgumentList 2>&1
    $exitCode = if ($null -eq $global:LASTEXITCODE) { 0 } else { [int]$global:LASTEXITCODE }
    if ($exitCode -ne 0) {
        throw "$Name failed with exit code ${exitCode}:`n$($output -join "`n")"
    }
    return (($output | ForEach-Object { [string]$_ }) -join "`n") | ConvertFrom-Json
}

function Assert-Equal {
    param(
        [string]$Name,
        [object]$Actual,
        [object]$Expected
    )

    if ([string]$Actual -ne [string]$Expected) {
        throw "$Name expected '$Expected' but got '$Actual'"
    }
}

function Assert-True {
    param(
        [string]$Name,
        [bool]$Value
    )

    if (-not $Value) {
        throw "$Name expected true"
    }
}

function Assert-Contains {
    param(
        [string]$Name,
        [string]$Haystack,
        [string]$Needle
    )

    if (-not $Haystack.Contains($Needle)) {
        throw "$Name missing '$Needle'"
    }
}

$profilesResult = Invoke-JsonCommand -Name "profiles" -ArgumentList @("scripts/check_runtime.py", "profiles", "--project-root", $ProjectRoot)
$profiles = @($profilesResult | ForEach-Object { $_ })
Assert-Equal -Name "profile count" -Actual $profiles.Count -Expected 3

$byRecipe = @{}
foreach ($profile in $profiles) {
    Assert-Equal -Name "$($profile.recipe_id) tier" -Actual $profile.tier -Expected "recommended_16gb"
    Assert-Equal -Name "$($profile.recipe_id) launcher" -Actual $profile.launcher -Expected "python"
    Assert-True -Name "$($profile.recipe_id) low_vram" -Value ([bool]$profile.low_vram)
    Assert-True -Name "$($profile.recipe_id) use_8bit_adam" -Value ([bool]$profile.use_8bit_adam)
    Assert-True -Name "$($profile.recipe_id) save_samples" -Value ([bool]$profile.save_samples)
    Assert-True -Name "$($profile.recipe_id) save_checkpoints" -Value ([bool]$profile.save_checkpoints)
    $byRecipe[$profile.recipe_id] = $profile
}

foreach ($recipeId in @("flux2-klein-identity", "flux2-klein-editing", "z-image-turbo-vlm")) {
    if (-not $byRecipe.ContainsKey($recipeId)) {
        throw "Missing production profile for $recipeId"
    }
}

$identity = $byRecipe["flux2-klein-identity"]
Assert-Equal -Name "identity resolution" -Actual $identity.resolution_scale -Expected "0.625"
Assert-Equal -Name "identity sample scale" -Actual $identity.sample_resolution_scale -Expected "0.5"
Assert-Equal -Name "identity steps" -Actual $identity.max_train_steps -Expected 5
Assert-Contains -Name "identity evidence" -Haystack ($identity.evidence -join " ") -Needle "flux2_identity_res0625_artifacts_scale05_5step_20260623045623"

$editing = $byRecipe["flux2-klein-editing"]
Assert-Equal -Name "editing resolution" -Actual $editing.resolution_scale -Expected "0.5625"
Assert-Equal -Name "editing sample scale" -Actual $editing.sample_resolution_scale -Expected "0.5"
Assert-Equal -Name "editing steps" -Actual $editing.max_train_steps -Expected 5
Assert-Contains -Name "editing evidence" -Haystack ($editing.evidence -join " ") -Needle "flux2_editing_res05625_artifacts_scale05_5step_20260623044037"

$zimage = $byRecipe["z-image-turbo-vlm"]
Assert-Equal -Name "zimage resolution" -Actual $zimage.resolution_scale -Expected "0.5"
Assert-Equal -Name "zimage sample scale" -Actual $zimage.sample_resolution_scale -Expected ""
Assert-Equal -Name "zimage steps" -Actual $zimage.max_train_steps -Expected 2
Assert-Contains -Name "zimage evidence" -Haystack ($zimage.evidence -join " ") -Needle "zimage_style_res05_artifacts_2step_202606221528"

$settings = Invoke-JsonCommand -Name "settings" -ArgumentList @("scripts/check_runtime.py", "settings", "--project-root", $ProjectRoot)
Assert-Equal -Name "settings profile count" -Actual @($settings.production_profiles).Count -Expected 3

$recipesSource = Get-Content -Raw -LiteralPath "trainer-ui\src\lib\recipes.ts"
$jobsSource = Get-Content -Raw -LiteralPath "trainer-ui\src\lib\jobs.ts"
$wizardSource = Get-Content -Raw -LiteralPath "trainer-ui\src\components\NewJobWizard.tsx"

Assert-Contains -Name "recipes source" -Haystack $recipesSource -Needle "productionProfile"
Assert-Contains -Name "jobs source" -Haystack $jobsSource -Needle "productionProfileForRecipe"
Assert-Contains -Name "jobs source" -Haystack $jobsSource -Needle "RESOLUTION_SCALE=`${profile.resolutionScale}"
Assert-Contains -Name "jobs source" -Haystack $jobsSource -Needle "MAX_TRAIN_STEPS=`${profile.maxTrainSteps}"
Assert-Contains -Name "jobs source" -Haystack $jobsSource -Needle "SAVE_SAMPLES=1"
Assert-Contains -Name "jobs source" -Haystack $jobsSource -Needle "SAVE_CHECKPOINTS=1"
Assert-Contains -Name "wizard source" -Haystack $wizardSource -Needle "Recommended 16GB starter"
Assert-Contains -Name "wizard source" -Haystack $wizardSource -Needle "Training scale"
Assert-Contains -Name "wizard source" -Haystack $wizardSource -Needle "Sample scale"

[pscustomobject]@{
    Ok = $true
    Profiles = $profiles.Count
    Identity = "$($identity.resolution_scale) train / $($identity.sample_resolution_scale) sample / $($identity.max_train_steps) steps"
    Editing = "$($editing.resolution_scale) train / $($editing.sample_resolution_scale) sample / $($editing.max_train_steps) steps"
    ZImage = "$($zimage.resolution_scale) train / $($zimage.max_train_steps) steps"
} | Format-List
