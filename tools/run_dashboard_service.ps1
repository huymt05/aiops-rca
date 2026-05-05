param(
    [string]$AnomalyBaseUrl = "http://127.0.0.1:8000",
    [string]$RcaBaseUrl = "http://127.0.0.1:8001",
    [string]$OrchestratorBaseUrl = "http://127.0.0.1:8002",
    [string]$GraphRoot = "D:\HOCTAP\2025-2026\HK2\DACN\microservices-demo\data_rca_balanced_v3\processed\rca\graph_tensors",
    [ValidateSet("demo", "real")]
    [string]$RecoveryMode = "demo",
    [string]$RecoveryNamespace = "default",
    [int]$RecoveryTimeoutSeconds = 120,
    [string]$BindHost = "0.0.0.0",
    [int]$Port = 8010
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$env:AIOPS_DASHBOARD_ANOMALY_BASE_URL = $AnomalyBaseUrl
$env:AIOPS_DASHBOARD_RCA_BASE_URL = $RcaBaseUrl
$env:AIOPS_DASHBOARD_ORCH_BASE_URL = $OrchestratorBaseUrl
$env:AIOPS_DASHBOARD_GRAPH_ROOT = $GraphRoot
$env:AIOPS_RECOVERY_MODE = $RecoveryMode
$env:AIOPS_RECOVERY_NAMESPACE = $RecoveryNamespace
$env:AIOPS_RECOVERY_TIMEOUT_SECONDS = "$RecoveryTimeoutSeconds"

Write-Host "Starting dashboard service..." -ForegroundColor Cyan
Write-Host "Repo root: $repoRoot" -ForegroundColor Yellow
Write-Host "Anomaly URL: $AnomalyBaseUrl" -ForegroundColor Yellow
Write-Host "RCA URL: $RcaBaseUrl" -ForegroundColor Yellow
Write-Host "Orchestrator URL: $OrchestratorBaseUrl" -ForegroundColor Yellow
Write-Host "Graph root: $GraphRoot" -ForegroundColor Yellow
Write-Host "Recovery mode: $RecoveryMode" -ForegroundColor Yellow
Write-Host "Recovery namespace: $RecoveryNamespace" -ForegroundColor Yellow
Write-Host "Recovery timeout: $RecoveryTimeoutSeconds s" -ForegroundColor Yellow
Write-Host "Dashboard URL: http://127.0.0.1:$Port" -ForegroundColor Yellow

python -m uvicorn aiops_framework.dashboard.app:app --host $BindHost --port $Port
