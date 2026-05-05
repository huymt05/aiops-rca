param(
    [string]$ArtifactDir = "D:\HOCTAP\2025-2026\HK2\DACN\microservices-demo\data_anomaly_balanced_v3\models\anomaly_gbrt_balanced",
    [string]$ThresholdOverride = "",
    [string]$BindHost = "0.0.0.0",
    [int]$Port = 8000
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
if (-not (Test-Path $ArtifactDir)) {
    throw "Missing anomaly artifact directory: $ArtifactDir"
}

Set-Location $repoRoot
$env:AIOPS_ANOMALY_ARTIFACT_DIR = $ArtifactDir
if ($ThresholdOverride -ne "") {
    $env:AIOPS_ANOMALY_THRESHOLD_OVERRIDE = $ThresholdOverride
} else {
    Remove-Item Env:AIOPS_ANOMALY_THRESHOLD_OVERRIDE -ErrorAction SilentlyContinue
}

Write-Host "Starting anomaly inference service..." -ForegroundColor Cyan
Write-Host "Repo root: $repoRoot" -ForegroundColor Yellow
Write-Host "Artifact dir: $ArtifactDir" -ForegroundColor Yellow
if ($ThresholdOverride -ne "") {
    Write-Host "Threshold override: $ThresholdOverride" -ForegroundColor Yellow
}
Write-Host "URL: http://127.0.0.1:$Port" -ForegroundColor Yellow

python -m uvicorn aiops_framework.inference.anomaly_service.app:app --host $BindHost --port $Port
