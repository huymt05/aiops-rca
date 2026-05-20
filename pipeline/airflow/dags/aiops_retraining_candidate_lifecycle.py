from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from airflow.decorators import dag, task
from airflow.models.param import Param
from airflow.operators.empty import EmptyOperator
from airflow.utils.task_group import TaskGroup

try:
    from airflow.sdk import get_current_context
except ImportError:  # pragma: no cover - compatibility fallback
    from airflow.operators.python import get_current_context


REPO_ROOT = Path(__file__).resolve().parents[3]
TOOLS_ROOT = REPO_ROOT / "tools"
PIPELINE_SCRIPTS = REPO_ROOT / "pipeline" / "rca_data_pipeline" / "scripts"
AIRFLOW_ARTIFACT_ROOT = REPO_ROOT / "artifacts" / "airflow"


def _default_dataset_root(env_name: str, repo_fallback: Path, handoff_fallback: str) -> Path:
    configured = os.environ.get(env_name)
    if configured:
        return Path(configured)
    return Path(handoff_fallback)


DEFAULT_ANOMALY_ROOT = _default_dataset_root(
    "AIOPS_AIRFLOW_ANOMALY_DATA_ROOT",
    REPO_ROOT / "data_anomaly_balanced_v3",
    "/opt/airflow/handoff/datasets/anomaly_train_dataset",
)
DEFAULT_RCA_ROOT = _default_dataset_root(
    "AIOPS_AIRFLOW_RCA_DATA_ROOT",
    REPO_ROOT / "data_rca_balanced_v3",
    "/opt/airflow/handoff/datasets/rca_train_dataset",
)
DEFAULT_ANOMALY_MODELS_ROOT = REPO_ROOT / "data_anomaly_balanced_v3" / "models"
DEFAULT_RCA_MODELS_ROOT = REPO_ROOT / "data_rca_balanced_v3" / "models"
DEFAULT_MLFLOW_EXPERIMENT = "aiops-retraining-airflow"


def _python_bin() -> str:
    return os.environ.get("AIOPS_PYTHON", sys.executable)


def _run_python(script: Path, *args: object, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    cmd = [_python_bin(), str(script), *[str(arg) for arg in args]]
    completed = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        env={**os.environ, **(env or {})},
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed ({completed.returncode}): {' '.join(cmd)}\n"
            f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )
    return completed


def _context_bundle() -> dict[str, Any]:
    context = get_current_context()
    logical_date = context.get("logical_date")
    dag_run = context.get("dag_run")
    if logical_date is None:
        if dag_run is not None:
            logical_date = getattr(dag_run, "logical_date", None)
    if logical_date is None:
        logical_date = context.get("data_interval_start")
    if logical_date is None:
        ts = context.get("ts")
        if ts:
            logical_date = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    if logical_date is None:
        raise KeyError(
            "Unable to resolve logical_date from Airflow task context. "
            f"Available keys: {sorted(context.keys())}"
        )
    params = dict(context.get("params") or {})
    if dag_run is not None:
        dag_run_conf = getattr(dag_run, "conf", None) or {}
        if isinstance(dag_run_conf, dict):
            for key, value in dag_run_conf.items():
                if value is not None:
                    params[key] = value
    run_slug = logical_date.strftime("%Y%m%dT%H%M%S")
    system_id = str(params["system_id"]).strip()
    artifacts_dir = AIRFLOW_ARTIFACT_ROOT / system_id / run_slug
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    return {
        "params": params,
        "system_id": system_id,
        "run_slug": run_slug,
        "artifacts_dir": artifacts_dir,
        "logical_date": logical_date,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return str(path)


@dag(
    dag_id="aiops_retraining_candidate_lifecycle",
    description="Feedback-aware candidate retraining DAG for anomaly and RCA models with human-in-the-loop promotion.",
    start_date=datetime(2026, 5, 18),
    schedule="0 2 * * *",
    catchup=False,
    max_active_runs=4,
    dagrun_timeout=timedelta(hours=3),
    render_template_as_native_obj=True,
    tags=["aiops", "mlops", "retraining", "candidate-lifecycle"],
    default_args={
        "owner": "aiops-team",
        "retries": 0,
        "retry_delay": timedelta(minutes=1),
    },
    params={
        "system_id": Param("online-boutique", type="string"),
        "anomaly_data_root": Param(str(DEFAULT_ANOMALY_ROOT), type="string"),
        "rca_data_root": Param(str(DEFAULT_RCA_ROOT), type="string"),
        "anomaly_models_root": Param(str(DEFAULT_ANOMALY_MODELS_ROOT), type="string"),
        "rca_models_root": Param(str(DEFAULT_RCA_MODELS_ROOT), type="string"),
        "enable_mlflow": Param(True, type="boolean"),
        "mlflow_experiment": Param(DEFAULT_MLFLOW_EXPERIMENT, type="string"),
        "include_unknown_feedback": Param(False, type="boolean"),
        "anomaly_model_kind": Param("auto", enum=["auto", "ensemble", "xgb", "lgbm", "gbrt"]),
        "anomaly_optimize_for": Param("anomaly", enum=["anomaly", "normal"]),
        "anomaly_threshold_bias": Param(0.0, type="number"),
        "rca_device": Param("cpu", enum=["cpu", "cuda", "auto"]),
        "rca_epochs": Param(140, type="integer", minimum=1),
        "rca_hidden_dim": Param(48, type="integer", minimum=8),
        "rca_patience": Param(24, type="integer", minimum=1),
    },
)
def aiops_retraining_candidate_lifecycle():
    start = EmptyOperator(task_id="start")
    manual_promotion_gate = EmptyOperator(task_id="manual_promotion_gate")
    done = EmptyOperator(task_id="done")

    @task(task_id="validate_training_inputs")
    def validate_training_inputs() -> dict[str, Any]:
        bundle = _context_bundle()
        params = bundle["params"]
        anomaly_root = Path(str(params["anomaly_data_root"]))
        rca_root = Path(str(params["rca_data_root"]))

        required_paths = {
            "anomaly_feature_file": anomaly_root / "processed" / "anomaly" / "window_features_labeled.parquet",
            "rca_graph_tensor_root": rca_root / "processed" / "rca" / "graph_tensors",
            "feedback_export_script": TOOLS_ROOT / "export_feedback_training_set.py",
            "anomaly_train_script": PIPELINE_SCRIPTS / "27_train_anomaly.py",
            "rca_train_script": PIPELINE_SCRIPTS / "28_train_rca.py",
        }
        missing = {name: str(path) for name, path in required_paths.items() if not path.exists()}
        if missing:
            raise FileNotFoundError(f"Missing required DAG inputs: {json.dumps(missing, indent=2)}")

        validation = {
            "system_id": bundle["system_id"],
            "anomaly_data_root": str(anomaly_root),
            "rca_data_root": str(rca_root),
            "validated_at": bundle["logical_date"].isoformat(),
            "artifacts_dir": str(bundle["artifacts_dir"]),
        }
        _write_json(bundle["artifacts_dir"] / "validation_summary.json", validation)
        return validation

    @task(task_id="export_feedback_snapshot")
    def export_feedback_snapshot(_: dict[str, Any]) -> dict[str, Any]:
        bundle = _context_bundle()
        params = bundle["params"]
        output_path = bundle["artifacts_dir"] / "feedback_training_snapshot.jsonl"
        cmd_args: list[object] = [
            "--system-id",
            bundle["system_id"],
            "--output",
            output_path,
        ]
        if bool(params["include_unknown_feedback"]):
            cmd_args.append("--include-unknown")
        result = _run_python(TOOLS_ROOT / "export_feedback_training_set.py", *cmd_args)
        summary_path = output_path.with_name(output_path.stem + "_summary.json")
        return {
            "snapshot_path": str(output_path),
            "summary_path": str(summary_path),
            "stdout": result.stdout.strip(),
        }

    validation = validate_training_inputs()
    feedback_snapshot = export_feedback_snapshot(validation)

    with TaskGroup(group_id="candidate_training") as candidate_training:
        @task(task_id="train_anomaly_candidate")
        def train_anomaly_candidate(_: dict[str, Any]) -> dict[str, Any]:
            bundle = _context_bundle()
            params = bundle["params"]
            output_dir = Path(str(params["anomaly_models_root"])) / f"airflow_{bundle['system_id']}_anomaly_{bundle['run_slug']}"
            cmd_args: list[object] = [
                "--data-root",
                params["anomaly_data_root"],
                "--output-dir",
                output_dir,
                "--system-id",
                bundle["system_id"],
                "--model-kind",
                params["anomaly_model_kind"],
                "--optimize-for",
                params["anomaly_optimize_for"],
                "--threshold-bias",
                params["anomaly_threshold_bias"],
                "--mlflow-experiment",
                params["mlflow_experiment"],
            ]
            if bool(params["enable_mlflow"]):
                cmd_args.append("--mlflow")
            result = _run_python(PIPELINE_SCRIPTS / "27_train_anomaly.py", *cmd_args)
            return {
                "artifact_dir": str(output_dir),
                "run_manifest": str(output_dir / "run_manifest.json"),
                "metrics_path": str(output_dir / "metrics.json"),
                "stdout_tail": result.stdout.strip().splitlines()[-6:],
            }

        @task(task_id="train_rca_candidate")
        def train_rca_candidate(_: dict[str, Any]) -> dict[str, Any]:
            bundle = _context_bundle()
            params = bundle["params"]
            output_dir = Path(str(params["rca_models_root"])) / f"airflow_{bundle['system_id']}_rca_{bundle['run_slug']}"
            cmd_args: list[object] = [
                "--data-root",
                params["rca_data_root"],
                "--output-dir",
                output_dir,
                "--system-id",
                bundle["system_id"],
                "--device",
                params["rca_device"],
                "--epochs",
                params["rca_epochs"],
                "--hidden-dim",
                params["rca_hidden_dim"],
                "--patience",
                params["rca_patience"],
                "--mlflow-experiment",
                params["mlflow_experiment"],
            ]
            if bool(params["enable_mlflow"]):
                cmd_args.append("--mlflow")
            result = _run_python(PIPELINE_SCRIPTS / "28_train_rca.py", *cmd_args)
            return {
                "artifact_dir": str(output_dir),
                "run_manifest": str(output_dir / "run_manifest.json"),
                "metrics_path": str(output_dir / "metrics.json"),
                "stdout_tail": result.stdout.strip().splitlines()[-6:],
            }

        anomaly_candidate = train_anomaly_candidate(feedback_snapshot)
        rca_candidate = train_rca_candidate(feedback_snapshot)

    @task(task_id="build_registry_review_bundle")
    def build_registry_review_bundle(
        validation: dict[str, Any],
        feedback_snapshot: dict[str, Any],
        anomaly_result: dict[str, Any],
        rca_result: dict[str, Any],
    ) -> dict[str, Any]:
        from aiops_framework.inference.common.artifact_registry import get_model_summary

        bundle = _context_bundle()
        params = bundle["params"]
        anomaly_root = Path(str(params["anomaly_models_root"]))
        rca_root = Path(str(params["rca_models_root"]))

        summary = {
            "system_id": bundle["system_id"],
            "generated_at": bundle["logical_date"].isoformat(),
            "validation": validation,
            "feedback_snapshot": feedback_snapshot,
            "anomaly_training": anomaly_result,
            "rca_training": rca_result,
            "registry": {
                "anomaly": get_model_summary(anomaly_root, system_id=bundle["system_id"], model_type="anomaly"),
                "rca": get_model_summary(rca_root, system_id=bundle["system_id"], model_type="rca"),
            },
            "next_actions": [
                "Review candidate metrics and MLflow artifacts.",
                "Approve promotion from the dashboard Model Management view or by running script 31_promote_model.py.",
                "Keep promotion manual to preserve human-in-the-loop governance.",
            ],
        }
        review_bundle_path = bundle["artifacts_dir"] / "candidate_review_bundle.json"
        _write_json(review_bundle_path, summary)
        promotion_hint = {
            "anomaly": f"{_python_bin()} {PIPELINE_SCRIPTS / '31_promote_model.py'} --task anomaly --models-root {anomaly_root} --system-id {bundle['system_id']}",
            "rca": f"{_python_bin()} {PIPELINE_SCRIPTS / '31_promote_model.py'} --task rca --models-root {rca_root} --system-id {bundle['system_id']}",
        }
        hint_path = bundle["artifacts_dir"] / "promotion_commands.json"
        _write_json(hint_path, promotion_hint)
        return {
            "review_bundle_path": str(review_bundle_path),
            "promotion_hint_path": str(hint_path),
        }

    review_bundle = build_registry_review_bundle(validation, feedback_snapshot, anomaly_candidate, rca_candidate)

    start >> validation >> feedback_snapshot >> candidate_training >> review_bundle >> manual_promotion_gate >> done


dag = aiops_retraining_candidate_lifecycle()
