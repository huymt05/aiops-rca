# RCA Benchmark Runtime Integration

This repo already exposes `dashboard -> orchestrator -> rca-service`.
To reuse benchmark models from `aiops_rca_benchmark`, the first practical step is:

1. package the benchmark artifact into an `rca_service`-compatible directory
2. point `AIOPS_RCA_ARTIFACT_DIR` to that directory
3. run `rca_service`

## Supported benchmark model families

- `rf`: RandomForest ML Ranker
- `gat`: GAT baseline
- `hgnn`: Heterogeneous Graph / HGNN-style model

## Package an artifact

### RF

```powershell
python tools\package_benchmark_rca_artifacts.py `
  --model-family rf `
  --source-dir D:\doanchuyennganh_aiops\FINAL_SYSTEM\aiops_rca_benchmark\outputs\re2_ob\rf_ml_ranker\final_model `
  --dest-dir D:\DACN\CI_CD\aiops-rca-latest\data_rca_balanced_v3\models\re2_ob_rf_ml_ranker `
  --model-name re2_ob_rf_ml_ranker
```

### HGNN

```powershell
python tools\package_benchmark_rca_artifacts.py `
  --model-family hgnn `
  --source-dir D:\doanchuyennganh_aiops\FINAL_SYSTEM\aiops_rca_benchmark\outputs\re2_ob\hgnn_rca\train_run_01 `
  --dest-dir D:\DACN\CI_CD\aiops-rca-latest\data_rca_balanced_v3\models\re2_ob_hgnn_rca `
  --model-name re2_ob_hgnn_rca
```

## Run rca-service locally against packaged artifact

```powershell
$env:AIOPS_RCA_ARTIFACT_DIR = "D:\DACN\CI_CD\aiops-rca-latest\data_rca_balanced_v3\models\re2_ob_rf_ml_ranker"
$env:AIOPS_RCA_DEVICE = "cpu"
python -m uvicorn aiops_framework.inference.rca_service.app:app --host 0.0.0.0 --port 8001
```

## Important note

The runtime dashboard currently builds a lightweight 12-feature graph payload from live Jaeger spans.
The benchmark RF/HGNN models expect the newer 480-feature service representation.

So there are two phases:

1. make `rca_service` understand benchmark artifacts (done by this integration work)
2. upgrade live feature building so dashboard/orchestrator can send 480-feature payloads

This keeps the current API stable while letting the team migrate the runtime model safely.
