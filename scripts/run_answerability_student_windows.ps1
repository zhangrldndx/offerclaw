param(
    [Parameter(Mandatory=$true)][string]$Dataset,
    [Parameter(Mandatory=$true)][string]$OutputRoot,
    [Parameter(Mandatory=$true)][string]$DatasetSha256,
    [string]$BaseModel = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$actualDatasetSha256 = (Get-FileHash -Algorithm SHA256 $Dataset).Hash.ToLowerInvariant()
if ($actualDatasetSha256 -ne $DatasetSha256.ToLowerInvariant()) {
    throw "Dataset SHA256 mismatch: expected $DatasetSha256, got $actualDatasetSha256"
}
Write-Host "Dataset SHA256 verified: $actualDatasetSha256"

& $Python -c "import torch; assert torch.cuda.is_available(), 'CUDA is unavailable'; print(torch.cuda.get_device_name(0))"

foreach ($seed in @(17, 29, 43)) {
    $seedDir = Join-Path $OutputRoot "seed_$seed"
    & $Python train_answerability_student.py `
        --dataset $Dataset `
        --output $seedDir `
        --base-model $BaseModel `
        --seed $seed `
        --device cuda `
        --run-training
    if ($LASTEXITCODE -ne 0) {
        throw "Student training failed for seed $seed"
    }
}

$summary = Join-Path $OutputRoot "pilot_seed_summary.json"
& $Python scripts/summarize_answerability_student_seeds.py `
    --root $OutputRoot `
    --output $summary
if ($LASTEXITCODE -ne 0) {
    throw "Three-seed summary failed"
}

Write-Host "Training complete. Decision artifact: $summary"
