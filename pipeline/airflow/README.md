# Airflow DAG for MLOps Narrative

This directory contains a presentation-ready Airflow DAG for the project's retraining lifecycle:

- export operator feedback from the dashboard store
- retrain anomaly and RCA candidates in parallel
- register the resulting candidates into the per-system local registry
- emit a review bundle for human-in-the-loop promotion

## DAG

- `dags/aiops_retraining_candidate_lifecycle.py`

The DAG is intentionally scoped to the offline MLOps lifecycle. It does not replace:

- live anomaly inference
- live RCA inference
- dashboard-driven incident response
- dashboard-driven production promotion

That separation is deliberate and should be highlighted in the report:

- Airflow orchestrates scheduled retraining work.
- The AIOps dashboard remains the runtime control plane.

## Why this design fits the thesis

The DAG stops at candidate generation and a manual promotion gate. This keeps the model lifecycle auditable:

1. telemetry and operator feedback are exported
2. candidate models are retrained
3. MLflow captures training metadata when enabled
4. local registries are updated under the target `system_id`
5. a human reviews candidates before promotion to production

This gives the project a clear MLOps story without weakening the human-in-the-loop governance that already exists in the dashboard.

## Runtime assumptions

The DAG expects the repository layout to remain unchanged and relies on the existing scripts:

- `tools/export_feedback_training_set.py`
- `pipeline/rca_data_pipeline/scripts/27_train_anomaly.py`
- `pipeline/rca_data_pipeline/scripts/28_train_rca.py`
- `pipeline/rca_data_pipeline/scripts/31_promote_model.py`

## Local real runtime (recommended for the thesis demo)

This repository now also contains a local Docker-based Airflow runtime:

- `Dockerfile`
- `docker-compose.local.yml`
- `requirements-airflow-local.txt`

The runtime is intentionally a single-container `airflow standalone` setup. This keeps the demo light enough for a laptop while still giving you:

- a real Airflow UI
- real DAG runs
- task logs
- execution timestamps
- a reproducible retraining execution trace for the paper

The compose runtime also mounts the external handoff package:

- host: `E:\mlops_train_handoff`
- container: `/opt/airflow/handoff`

This lets you train from the handed-off datasets while still registering new candidates into the repository model roots that the dashboard already uses.

### Why this runtime shape is a good fit

For the thesis, the value comes from executing the retraining lifecycle with a real scheduler/orchestrator, not from operating a large Airflow cluster. A single-container local runtime provides authentic Airflow evidence without adding unnecessary operational complexity to the Kubernetes demo stack.

### Important environment assumptions

- The repository is mounted into the container at `/opt/airflow/repo`.
- The dashboard feedback source is read from the JSON store under `.tmp_aiops_dashboard`.
- MLflow is expected to be reachable from the container through:
  - `http://host.docker.internal:15000`

That means you should port-forward MLflow on the host before triggering a DAG run:

```powershell
kubectl port-forward svc/mlflow -n mlops 15000:5000
```

Then start Airflow from the repository root:

```powershell
docker compose -f pipeline/airflow/docker-compose.local.yml up --build -d
```

The UI will be available at:

- `http://127.0.0.1:18088`

Default local credentials:

- username: `admin`
- password: `admin`

### Triggering the first real run

Once the UI is up, trigger the DAG with:

- `system_id = online-boutique`
- `anomaly_data_root = /opt/airflow/handoff/datasets/anomaly_train_dataset`
- `rca_data_root = /opt/airflow/handoff/datasets/rca_train_dataset`
- `anomaly_models_root = /opt/airflow/repo/data_anomaly_balanced_v3/models`
- `rca_models_root = /opt/airflow/repo/data_rca_balanced_v3/models`
- `enable_mlflow = true`
- `anomaly_model_kind = auto`
- `anomaly_optimize_for = anomaly`
- `rca_device = cpu`

Artifacts from the DAG will be written under:

- `artifacts/airflow/<system_id>/<run_slug>/`

The most important outputs for the paper are:

- `validation_summary.json`
- `feedback_training_snapshot_summary.json`
- `candidate_review_bundle.json`
- `promotion_commands.json`

The training scripts now support `--system-id`, so candidates are registered into the correct per-system registry block.

## Recommended parameters

- `system_id`: `online-boutique` for the strongest end-to-end demo
- `enable_mlflow`: `true`
- `anomaly_model_kind`: `auto`
- `anomaly_optimize_for`: `anomaly`
- `rca_device`: `cpu`

For `sock_shop`, the DAG can still register candidates under `sock_shop`, but the report should clearly state whether the underlying dataset is dedicated or shared/bootstrap.

## Suggested report framing

Use this DAG to support the following narrative:

> Runtime incidents and operator feedback are retained by the dashboard store. Airflow periodically exports this feedback, retrains anomaly and RCA candidates, logs experiments to MLflow, updates the local registry, and prepares a candidate review bundle. Promotion remains a deliberate operator decision exposed through the dashboard Model Management interface.

## Validation tips

Before demoing the DAG:

1. confirm dashboard feedback exists
2. confirm anomaly and RCA dataset roots exist
3. confirm MLflow is reachable if `enable_mlflow=true`
4. run one DAG execution for `online-boutique`
5. show the generated review bundle and promotion commands
