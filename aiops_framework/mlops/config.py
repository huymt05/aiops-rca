from __future__ import annotations

import os
from typing import Any


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def get_mlops_config() -> dict[str, Any]:
    backend = os.environ.get("AIOPS_MODEL_REGISTRY_BACKEND", "local").strip().lower() or "local"
    tracking_uri = os.environ.get("AIOPS_MLFLOW_TRACKING_URI", "").strip()
    s3_endpoint_url = os.environ.get("AIOPS_MLFLOW_S3_ENDPOINT_URL", "").strip()
    artifact_root = os.environ.get("AIOPS_MLFLOW_ARTIFACT_ROOT", "").strip()
    experiment_name = os.environ.get("AIOPS_MLFLOW_DEFAULT_EXPERIMENT", "aiops-runtime").strip() or "aiops-runtime"
    enabled = _env_bool("AIOPS_MLFLOW_ENABLED", default=bool(tracking_uri))

    return {
        "backend": backend,
        "mlflow_enabled": enabled,
        "tracking_uri": tracking_uri,
        "s3_endpoint_url": s3_endpoint_url,
        "artifact_root": artifact_root,
        "default_experiment": experiment_name,
    }
