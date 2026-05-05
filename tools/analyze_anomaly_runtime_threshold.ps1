param(
    [string]$DataRoot = "D:\doanchuyennganh_aiops\Data4.4\data_rca_balanced_v2\data_rca_balanced_v2",
    [string]$ArtifactDir,
    [string]$OutputDir = ""
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

if ($OutputDir -eq "") {
    $OutputDir = Join-Path $DataRoot "analysis\anomaly_threshold_runtime"
}
if (-not $ArtifactDir) {
    $ArtifactDir = Join-Path $DataRoot "models\anomaly_xgb_lgbm_runtime_faults"
}
$env:AIOPS_ANOMALY_ARTIFACT_DIR = $ArtifactDir

Write-Host "Analyzing anomaly score distribution..." -ForegroundColor Cyan
Write-Host "Data root: $DataRoot" -ForegroundColor Yellow
Write-Host "Artifact dir: $ArtifactDir" -ForegroundColor Yellow
Write-Host "Output dir: $OutputDir" -ForegroundColor Yellow

python .\tools\analyze_anomaly_scores.py `
  --data-root $DataRoot `
  --artifact-dir $ArtifactDir `
  --output-dir $OutputDir
