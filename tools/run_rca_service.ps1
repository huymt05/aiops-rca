param(
    [string]$ArtifactDir = "",
    [string]$ModelRegistryPath = "",
    [string]$DefaultModelKey = "rf_ml_ranker",
    [ValidateSet("cpu", "cuda")]
    [string]$Device = "cuda",
    [string]$BindHost = "0.0.0.0",
    [int]$Port = 8001
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot
if ($ModelRegistryPath) {
    if (-not (Test-Path $ModelRegistryPath)) {
        throw "Missing RCA model registry file: $ModelRegistryPath"
    }
    $env:AIOPS_RCA_MODEL_REGISTRY_PATH = $ModelRegistryPath
    $env:AIOPS_RCA_DEFAULT_MODEL_KEY = $DefaultModelKey
} elseif ($ArtifactDir) {
    if (-not (Test-Path $ArtifactDir)) {
        throw "Missing RCA artifact directory: $ArtifactDir"
    }
    $env:AIOPS_RCA_ARTIFACT_DIR = $ArtifactDir
    $env:AIOPS_RCA_DEFAULT_MODEL_KEY = $DefaultModelKey
} else {
    throw "Specify either -ArtifactDir or -ModelRegistryPath."
}
$env:AIOPS_RCA_DEVICE = $Device

Write-Host "Starting RCA inference service..." -ForegroundColor Cyan
Write-Host "Repo root: $repoRoot" -ForegroundColor Yellow
if ($ModelRegistryPath) {
    Write-Host "Model registry: $ModelRegistryPath" -ForegroundColor Yellow
    Write-Host "Default model key: $DefaultModelKey" -ForegroundColor Yellow
} else {
    Write-Host "Artifact dir: $ArtifactDir" -ForegroundColor Yellow
}
Write-Host "Device: $Device" -ForegroundColor Yellow
Write-Host "URL: http://127.0.0.1:$Port" -ForegroundColor Yellow

python -m uvicorn aiops_framework.inference.rca_service.app:app --host $BindHost --port $Port
