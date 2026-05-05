param(
    [string]$RcaBaseUrl = "http://127.0.0.1:8001",
    [string]$JaegerUrl = "http://192.168.138.131:16686",
    [string]$PrometheusUrl = "http://127.0.0.1:9090",
    [string]$SystemId = "online-boutique",
    [string]$SourceService = "frontend",
    [int]$LookbackMinutes = 2,
    [int]$QueryLimit = 150,
    [string]$PythonExe = "D:\doanchuyennganh_aiops\doan_env\Scripts\python.exe"
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$script = @"
import json
import sys
sys.path.insert(0, r"$repoRoot")
from aiops_framework.dashboard.live_data import collect_live_inputs

payload = collect_live_inputs(
    system_id=r"$SystemId",
    source_service=r"$SourceService",
    jaeger_url=r"$JaegerUrl",
    prometheus_url=r"$PrometheusUrl",
    lookback_minutes=int("$LookbackMinutes"),
    query_limit=int("$QueryLimit"),
)
print(json.dumps(payload["graph"], ensure_ascii=False))
"@

$graphBody = $script | & $PythonExe -
if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($graphBody)) {
    throw "Failed to collect live RCA payload from Python helper. Check Python environment and Jaeger connectivity."
}
$graphPayload = $graphBody | ConvertFrom-Json

Write-Host "Collected live graph payload" -ForegroundColor Cyan
Write-Host ("Graph ID: " + $graphPayload.graph_id) -ForegroundColor Yellow
Write-Host ("Node count: " + $graphPayload.node_names.Count) -ForegroundColor Yellow
Write-Host ("Feature count: " + $graphPayload.metadata.feature_count) -ForegroundColor Yellow
Write-Host ("Feature schema: " + $graphPayload.metadata.feature_schema) -ForegroundColor Yellow

Invoke-RestMethod -Uri "$RcaBaseUrl/predict/graph" -Method Post -ContentType "application/json" -Body $graphBody
