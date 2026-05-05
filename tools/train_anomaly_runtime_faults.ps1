param(
    [string]$DataRoot = "D:\doanchuyennganh_aiops\Data4.4\data_rca_balanced_v2\data_rca_balanced_v2",
    [string]$OutputDir = "",
    [string]$ModelKind = "ensemble",
    [double]$ThresholdBias = -0.08,
    [string]$OptimizeFor = "anomaly"
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

if ($OutputDir -eq "") {
    $OutputDir = Join-Path $DataRoot "models\anomaly_xgb_lgbm_runtime_faults"
}

Write-Host "Training runtime-fault-oriented anomaly model..." -ForegroundColor Cyan
Write-Host "Data root: $DataRoot" -ForegroundColor Yellow
Write-Host "Output dir: $OutputDir" -ForegroundColor Yellow
Write-Host "Model kind: $ModelKind" -ForegroundColor Yellow
Write-Host "Optimize for: $OptimizeFor" -ForegroundColor Yellow
Write-Host "Threshold bias: $ThresholdBias" -ForegroundColor Yellow

python .\pipeline\rca_data_pipeline\scripts\27_train_anomaly.py `
  --data-root $DataRoot `
  --output-dir $OutputDir `
  --model-kind $ModelKind `
  --optimize-for $OptimizeFor `
  --threshold-bias $ThresholdBias
