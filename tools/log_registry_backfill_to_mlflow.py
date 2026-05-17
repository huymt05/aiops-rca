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
        with urlopen(request, timeout=20) as response:
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


def _flatten_metrics(metrics: dict[str, Any], prefix: str = "") -> dict[str, float]:
    flat: dict[str, float] = {}
    for key, value in (metrics or {}).items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, bool):
            flat[full_key] = 1.0 if value else 0.0
        elif isinstance(value, (int, float)):
            flat[full_key] = float(value)
        elif isinstance(value, dict):
            flat.update(_flatten_metrics(value, full_key))
    return flat


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
            {"key": key[:250], "value": float(value), "timestamp": now_ms, "step": 0}
            for key, value in metrics.items()
        ],
        "params": [{"key": key[:250], "value": str(value)[:6000]} for key, value in params.items()],
        "tags": [{"key": key[:250], "value": str(value)[:6000]} for key, value in tags.items()],
    }
    _request_json("POST", f"{tracking_uri.rstrip('/')}/api/2.0/mlflow/runs/log-batch", payload)


def _terminate_run(tracking_uri: str, run_id: str, status: str = "FINISHED") -> None:
    payload = {"run_id": run_id, "status": status, "end_time": int(time.time() * 1000)}
    _request_json("POST", f"{tracking_uri.rstrip('/')}/api/2.0/mlflow/runs/update", payload)


def _log_entry(
    tracking_uri: str,
    experiment_id: str,
    *,
    system_id: str,
    model_type: str,
    stage_name: str,
    entry: dict[str, Any],
    models_root: Path,
) -> str:
    model_name = str(entry.get("model_name") or entry.get("model_id") or "unknown")
    run_id = _create_run(
        tracking_uri,
        experiment_id,
        {
            "mlflow.runName": f"{model_type}-{stage_name}-{model_name}",
            "project": "aiops-graduation-project",
            "system_id": system_id,
            "model_type": model_type,
            "registry_stage": stage_name,
            "source": "tools/log_registry_backfill_to_mlflow.py",
        },
    )

    metrics = _flatten_metrics(entry.get("metrics") or {})
    if "rank_score" in entry:
        metrics["rank_score"] = float(entry["rank_score"])

    params = {
        "system_id": system_id,
        "model_type": model_type,
        "model_name": model_name,
        "model_version": str(entry.get("model_version") or ""),
        "artifact_dir": str((models_root / str(entry.get("artifact_dir") or model_name)).resolve()),
        "run_manifest_path": str(entry.get("run_manifest_path") or ""),
        "status": str(entry.get("status") or stage_name),
        "notes": str(entry.get("notes") or ""),
        "trained_at": str(entry.get("trained_at") or ""),
        "updated_at": str(entry.get("updated_at") or ""),
        "backfill": "true",
    }
    tags = {
        "registry_stage": stage_name,
        "model_registry_backend": "hybrid",
    }

    _log_batch(tracking_uri, run_id, metrics=metrics, params=params, tags=tags)
    _terminate_run(tracking_uri, run_id)
    return run_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill current local model registry entries into MLflow runs.")
    parser.add_argument("--tracking-uri", default="http://127.0.0.1:5000")
    parser.add_argument("--experiment-name", default="aiops-registry-backfill")
    parser.add_argument("--system-id", default="online-boutique")
    args = parser.parse_args()

    experiment_id = _ensure_experiment(args.tracking_uri, args.experiment_name)
    created: list[dict[str, str]] = []

    for model_type in ("anomaly", "rca"):
        models_root = _system_model_root(args.system_id, model_type)
        summary = get_model_summary(models_root, system_id=args.system_id, model_type=model_type)

        production = summary.get("production")
        if production:
            created.append(
                {
                    "model_type": model_type,
                    "stage": "production",
                    "run_id": _log_entry(
                        args.tracking_uri,
                        experiment_id,
                        system_id=args.system_id,
                        model_type=model_type,
                        stage_name="production",
                        entry=production,
                        models_root=models_root,
                    ),
                }
            )

        for index, candidate in enumerate(summary.get("candidates") or [], start=1):
            created.append(
                {
                    "model_type": model_type,
                    "stage": f"candidate_{index}",
                    "run_id": _log_entry(
                        args.tracking_uri,
                        experiment_id,
                        system_id=args.system_id,
                        model_type=model_type,
                        stage_name=f"candidate_{index}",
                        entry=candidate,
                        models_root=models_root,
                    ),
                }
            )

    print(
        json.dumps(
            {
                "status": "ok",
                "tracking_uri": args.tracking_uri,
                "experiment_name": args.experiment_name,
                "experiment_id": experiment_id,
                "system_id": args.system_id,
                "created_runs": created,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
