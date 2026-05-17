from __future__ import annotations

import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .config import get_mlops_config


def _try_fetch_json(url: str, timeout: float = 3.0) -> tuple[bool, dict[str, Any], int | None, str | None]:
    request = Request(url, headers={"Accept": "application/json"})
    try:
        with urlopen(request, timeout=timeout) as response:
            status = getattr(response, "status", None)
            payload = json.loads(response.read().decode("utf-8") or "{}")
            if isinstance(payload, dict):
                return True, payload, status, None
            return True, {"value": payload}, status, None
    except HTTPError as exc:
        return False, {}, exc.code, str(exc)
    except URLError as exc:
        return False, {}, None, str(exc)
    except Exception as exc:  # pragma: no cover - defensive health summary
        return False, {}, None, str(exc)


def _mlflow_health(tracking_uri: str) -> dict[str, Any]:
    if not tracking_uri:
        return {
            "enabled": False,
            "status": "disabled",
            "tracking_uri": "",
            "version": None,
            "detail": "AIOPS_MLFLOW_TRACKING_URI is not configured.",
        }

    base = tracking_uri.rstrip("/")
    health_ok, health_payload, health_status, health_error = _try_fetch_json(f"{base}/health")
    version_ok, version_payload, version_status, version_error = _try_fetch_json(f"{base}/version")

    version = (
        version_payload.get("version")
        or version_payload.get("mlflow_version")
        or version_payload.get("value")
        if version_ok
        else None
    )
    if health_ok or version_ok:
        return {
            "enabled": True,
            "status": "ok",
            "tracking_uri": tracking_uri,
            "health_status_code": health_status,
            "version_status_code": version_status,
            "version": version,
            "detail": health_payload or version_payload or {},
        }

    return {
        "enabled": True,
        "status": "down",
        "tracking_uri": tracking_uri,
        "health_status_code": health_status,
        "version_status_code": version_status,
        "version": None,
        "detail": {
            "health_error": health_error,
            "version_error": version_error,
        },
    }


def get_mlops_status() -> dict[str, Any]:
    cfg = get_mlops_config()
    health = _mlflow_health(cfg["tracking_uri"])
    return {
        "backend": cfg["backend"],
        "mlflow_enabled": cfg["mlflow_enabled"],
        "tracking_uri": cfg["tracking_uri"],
        "artifact_root": cfg["artifact_root"],
        "s3_endpoint_url": cfg["s3_endpoint_url"],
        "default_experiment": cfg["default_experiment"],
        "mlflow": health,
    }
