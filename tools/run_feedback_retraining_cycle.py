from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aiops_framework.inference.common.artifact_registry import get_model_summary


TOOLS_ROOT = REPO_ROOT / "tools"
PIPELINE_SCRIPTS = REPO_ROOT / "pipeline" / "rca_data_pipeline" / "scripts"
DEFAULT_ANOMALY_ROOT = REPO_ROOT / "data_anomaly_balanced_v3"
DEFAULT_RCA_ROOT = REPO_ROOT / "data_rca_balanced_v3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a real feedback-aware retraining cycle: export feedback, train candidates, and build a review bundle."
    )
    parser.add_argument("--system-id", default="online-boutique")
    parser.add_argument("--anomaly-data-root", type=Path, default=DEFAULT_ANOMALY_ROOT)
    parser.add_argument("--rca-data-root", type=Path, default=DEFAULT_RCA_ROOT)
    parser.add_argument("--anomaly-models-root", type=Path, default=DEFAULT_ANOMALY_ROOT / "models")
    parser.add_argument("--rca-models-root", type=Path, default=DEFAULT_RCA_ROOT / "models")
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/retraining_runs"))
    parser.add_argument("--include-unknown-feedback", action="store_true")
    parser.add_argument("--enable-mlflow", action="store_true")
    parser.add_argument("--mlflow-experiment", default="aiops-retraining-local")
    parser.add_argument("--anomaly-model-kind", choices=["auto", "ensemble", "xgb", "lgbm", "gbrt"], default="auto")
    parser.add_argument("--anomaly-optimize-for", choices=["anomaly", "normal"], default="anomaly")
    parser.add_argument("--anomaly-threshold-bias", type=float, default=0.0)
    parser.add_argument("--rca-device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument("--rca-epochs", type=int, default=140)
    parser.add_argument("--rca-hidden-dim", type=int, default=48)
    parser.add_argument("--rca-patience", type=int, default=24)
    parser.add_argument("--python-exe", default=sys.executable)
    return parser.parse_args()


def now_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _run_python(python_exe: str, script: Path, *args: object) -> subprocess.CompletedProcess[str]:
    cmd = [python_exe, str(script), *[str(arg) for arg in args]]
    result = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {' '.join(cmd)}\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    return result


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_dir = args.output_root / args.system_id / now_slug()
    run_dir.mkdir(parents=True, exist_ok=True)

    feedback_snapshot = run_dir / "feedback_training_snapshot.jsonl"
    export_args: list[object] = ["--system-id", args.system_id, "--output", feedback_snapshot]
    if args.include_unknown_feedback:
        export_args.append("--include-unknown")
    export_result = _run_python(args.python_exe, TOOLS_ROOT / "export_feedback_training_set.py", *export_args)

    anomaly_output = args.anomaly_models_root / f"feedback_{args.system_id}_anomaly_{run_dir.name}"
    anomaly_args: list[object] = [
        "--data-root",
        args.anomaly_data_root,
        "--output-dir",
        anomaly_output,
        "--system-id",
        args.system_id,
        "--model-kind",
        args.anomaly_model_kind,
        "--optimize-for",
        args.anomaly_optimize_for,
        "--threshold-bias",
        args.anomaly_threshold_bias,
        "--mlflow-experiment",
        args.mlflow_experiment,
    ]
    if args.enable_mlflow:
        anomaly_args.append("--mlflow")
    anomaly_result = _run_python(args.python_exe, PIPELINE_SCRIPTS / "27_train_anomaly.py", *anomaly_args)

    rca_output = args.rca_models_root / f"feedback_{args.system_id}_rca_{run_dir.name}"
    rca_args: list[object] = [
        "--data-root",
        args.rca_data_root,
        "--output-dir",
        rca_output,
        "--system-id",
        args.system_id,
        "--device",
        args.rca_device,
        "--epochs",
        args.rca_epochs,
        "--hidden-dim",
        args.rca_hidden_dim,
        "--patience",
        args.rca_patience,
        "--mlflow-experiment",
        args.mlflow_experiment,
    ]
    if args.enable_mlflow:
        rca_args.append("--mlflow")
    rca_result = _run_python(args.python_exe, PIPELINE_SCRIPTS / "28_train_rca.py", *rca_args)

    bundle = {
        "system_id": args.system_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "feedback_snapshot": {
            "path": str(feedback_snapshot),
            "summary_path": str(feedback_snapshot.with_name(feedback_snapshot.stem + "_summary.json")),
            "stdout": export_result.stdout.strip().splitlines(),
        },
        "anomaly_training": {
            "artifact_dir": str(anomaly_output),
            "run_manifest_path": str(anomaly_output / "run_manifest.json"),
            "stdout_tail": anomaly_result.stdout.strip().splitlines()[-8:],
        },
        "rca_training": {
            "artifact_dir": str(rca_output),
            "run_manifest_path": str(rca_output / "run_manifest.json"),
            "stdout_tail": rca_result.stdout.strip().splitlines()[-8:],
        },
        "registry_snapshot": {
            "anomaly": get_model_summary(args.anomaly_models_root, system_id=args.system_id, model_type="anomaly"),
            "rca": get_model_summary(args.rca_models_root, system_id=args.system_id, model_type="rca"),
        },
        "next_actions": {
            "promote_anomaly": f"{args.python_exe} {PIPELINE_SCRIPTS / '31_promote_model.py'} --task anomaly --models-root {args.anomaly_models_root} --system-id {args.system_id}",
            "promote_rca": f"{args.python_exe} {PIPELINE_SCRIPTS / '31_promote_model.py'} --task rca --models-root {args.rca_models_root} --system-id {args.system_id}",
        },
    }
    bundle_path = run_dir / "feedback_retraining_review_bundle.json"
    _write_json(bundle_path, bundle)

    print(json.dumps({"run_dir": str(run_dir), "bundle_path": str(bundle_path)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
