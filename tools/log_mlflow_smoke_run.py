from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aiops_framework.core.config import load_system_config
from aiops_framework.inference.common.artifact_registry import get_model_summary


def _request_json(method: str, url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = Request(url, data=data, method=method.upper(), headers=headers)
    try:
        with urlopen(request, timeout=15) as response:
            return json.loads(response.read().decode("utf-8") or "{}")
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {url} failed: HTTP {exc.code} {body}") from exc
    except URLError as exc:
        raise RuntimeError(f"{method} {url} failed: {exc}") from exc


def _system_model_root(system_id: str, model_type: str) -> Path:
    cfg = load_system_config(system_id)
    profile = cfg.get("model_profile", {}).get(model_type, {})
    registry_root = profile.get("registry_root")
    if not registry_root:
        raise RuntimeError(f"System {system_id} has no {model_type} registry_root configured")
    path = Path(str(registry_root))
    if not path.is_absolute():
        path = (Path(cfg["system_root"]) / path).resolve()
    return path


def _ensure_experiment(tracking_uri: str, experiment_name: str) -> str:
    encoded = quote(experiment_name, safe="")
    get_url = f"{tracking_uri.rstrip('/')}/api/2.0/mlflow/experiments/get-by-name?experiment_name={encoded}"
    try:
        payload = _request_json("GET", get_url)
        experiment = payload.get("experiment") or {}
        experiment_id = experiment.get("experiment_id")
        if experiment_id:
            return str(experiment_id)
    except RuntimeError:
        pass

    payload = _request_json(
        "POST",
        f"{tracking_uri.rstrip('/')}/api/2.0/mlflow/experiments/create",
        {"name": experiment_name},
    )
    return str(payload["experiment_id"])


def _create_run(tracking_uri: str, experiment_id: str, tags: dict[str, str]) -> str:
    payload = {
        "experiment_id": str(experiment_id),
        "tags": [{"key": key, "value": value} for key, value in tags.items()],
    }
    response = _request_json("POST", f"{tracking_uri.rstrip('/')}/api/2.0/mlflow/runs/create", payload)
    run = response.get("run") or {}
    info = run.get("info") or {}
    run_id = info.get("run_id")
    if not run_id:
        raise RuntimeError("MLflow did not return a run_id")
    return str(run_id)


def _log_batch(
    tracking_uri: str,
    run_id: str,
    *,
    metrics: dict[str, float],
    params: dict[str, str],
    tags: dict[str, str],
) -> None:
    now_ms = int(time.time() * 1000)
    payload = {
        "run_id": run_id,
        "metrics": [
            {"key": key, "value": float(value), "timestamp": now_ms, "step": 0}
            for key, value in metrics.items()
        ],
        "params": [{"key": key, "value": str(value)} for key, value in params.items()],
        "tags": [{"key": key, "value": str(value)} for key, value in tags.items()],
    }
    _request_json("POST", f"{tracking_uri.rstrip('/')}/api/2.0/mlflow/runs/log-batch", payload)


def _terminate_run(tracking_uri: str, run_id: str, status: str = "FINISHED") -> None:
    payload = {"run_id": run_id, "status": status, "end_time": int(time.time() * 1000)}
    _request_json("POST", f"{tracking_uri.rstrip('/')}/api/2.0/mlflow/runs/update", payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="Log a bootstrap MLflow run for the AIOps project.")
    parser.add_argument("--tracking-uri", default="http://127.0.0.1:5000")
    parser.add_argument("--experiment-name", default="aiops-runtime")
    parser.add_argument("--system-id", default="online-boutique")
    args = parser.parse_args()

    anomaly_summary = get_model_summary(_system_model_root(args.system_id, "anomaly"), system_id=args.system_id, model_type="anomaly")
    rca_summary = get_model_summary(_system_model_root(args.system_id, "rca"), system_id=args.system_id, model_type="rca")

    experiment_id = _ensure_experiment(args.tracking_uri, args.experiment_name)
    run_id = _create_run(
        args.tracking_uri,
        experiment_id,
        {
            "mlflow.runName": f"bootstrap-{args.system_id}",
            "project": "aiops-graduation-project",
            "system_id": args.system_id,
            "run_purpose": "bootstrap-smoke-test",
        },
    )

    anomaly_prod = anomaly_summary.get("production") or {}
    rca_prod = rca_summary.get("production") or {}
    anomaly_candidates = anomaly_summary.get("candidates") or []
    rca_candidates = rca_summary.get("candidates") or []

    metrics = {
        "anomaly.rank_score": float(anomaly_prod.get("rank_score") or 0.0),
        "rca.rank_score": float(rca_prod.get("rank_score") or 0.0),
        "anomaly.candidate_count": float(len(anomaly_candidates)),
        "rca.candidate_count": float(len(rca_candidates)),
    }
    params = {
        "system_id": args.system_id,
        "registry_backend": "hybrid",
        "anomaly.production_model": str(anomaly_prod.get("model_name") or "none"),
        "rca.production_model": str(rca_prod.get("model_name") or "none"),
        "experiment_bootstrap": "true",
    }
    tags = {
        "stage": "bootstrap",
        "model_registry_backend": "hybrid",
        "source": "tools/log_mlflow_smoke_run.py",
    }

    _log_batch(args.tracking_uri, run_id, metrics=metrics, params=params, tags=tags)
    _terminate_run(args.tracking_uri, run_id)

    print(
        json.dumps(
            {
                "status": "ok",
                "tracking_uri": args.tracking_uri,
                "experiment_name": args.experiment_name,
                "experiment_id": experiment_id,
                "run_id": run_id,
                "system_id": args.system_id,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
