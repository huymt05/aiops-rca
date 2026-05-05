from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import torch

from aiops_framework.inference.common.artifact_registry import DEFAULT_STAGE, resolve_artifact_dir

from .model_def import HeteroTelemetryGNN, SimpleGraphAttention, infer_feature_groups


DEFAULT_MODEL_ROOT = Path(
    os.environ.get(
        "AIOPS_RCA_MODEL_ROOT",
        r"D:\HOCTAP\2025-2026\HK2\DACN\microservices-demo\data_rca_balanced_v3\models",
    )
)
DEFAULT_MODEL_STAGE = os.environ.get("AIOPS_RCA_MODEL_STAGE", DEFAULT_STAGE).strip() or DEFAULT_STAGE
DEFAULT_MODEL_REGISTRY_PATH = os.environ.get("AIOPS_RCA_MODEL_REGISTRY_PATH", "").strip()
DEFAULT_MODEL_REGISTRY_JSON = os.environ.get("AIOPS_RCA_MODEL_REGISTRY_JSON", "").strip()


def _resolve_default_artifact_dir() -> Path:
    explicit = str(os.environ.get("AIOPS_RCA_ARTIFACT_DIR", "")).strip()
    if explicit:
        return Path(explicit)
    try:
        return Path(resolve_artifact_dir(DEFAULT_MODEL_ROOT, DEFAULT_MODEL_STAGE))
    except FileNotFoundError:
        if DEFAULT_MODEL_REGISTRY_PATH or DEFAULT_MODEL_REGISTRY_JSON:
            return Path(".")
        raise


DEFAULT_ARTIFACT_DIR = _resolve_default_artifact_dir()


@dataclass
class ModelRegistryEntry:
    model_key: str
    artifact_dir: Path
    label: str


@dataclass
class LoadedRcaArtifacts:
    artifact_dir: Path
    inference_config: dict[str, object]
    model_config: dict[str, object]
    model: Any
    device: torch.device
    model_type: str
    feature_cols: list[str]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _coerce_registry_entry(model_key: str, value: Any, base_dir: Path | None = None) -> ModelRegistryEntry:
    if isinstance(value, str):
        artifact_dir = Path(value)
        if base_dir is not None and not artifact_dir.is_absolute():
            artifact_dir = (base_dir / artifact_dir).resolve()
        return ModelRegistryEntry(model_key=model_key, artifact_dir=artifact_dir, label=model_key)
    if isinstance(value, dict):
        artifact_dir = Path(str(value.get("artifact_dir") or value.get("path") or "").strip())
        if not str(artifact_dir):
            raise ValueError(f"Missing artifact_dir for RCA model_key '{model_key}'")
        if base_dir is not None and not artifact_dir.is_absolute():
            artifact_dir = (base_dir / artifact_dir).resolve()
        label = str(value.get("label") or model_key).strip() or model_key
        return ModelRegistryEntry(model_key=model_key, artifact_dir=artifact_dir, label=label)
    raise TypeError(f"Unsupported registry entry for RCA model_key '{model_key}'")


def _load_registry_payload() -> tuple[dict[str, Any] | None, Path | None]:
    if DEFAULT_MODEL_REGISTRY_PATH:
        registry_path = Path(DEFAULT_MODEL_REGISTRY_PATH)
        if registry_path.exists():
            return json.loads(registry_path.read_text(encoding="utf-8-sig")), registry_path.parent
    if DEFAULT_MODEL_REGISTRY_JSON:
        return json.loads(DEFAULT_MODEL_REGISTRY_JSON), None
    return None, None


def load_model_registry(default_artifact_dir: Path | None = None) -> tuple[str, dict[str, ModelRegistryEntry]]:
    artifact_dir = Path(default_artifact_dir or DEFAULT_ARTIFACT_DIR)
    payload, base_dir = _load_registry_payload()

    if not payload:
        default_key = str(os.environ.get("AIOPS_RCA_DEFAULT_MODEL_KEY", "")).strip() or artifact_dir.name
        return default_key, {
            default_key: ModelRegistryEntry(model_key=default_key, artifact_dir=artifact_dir, label=default_key)
        }

    raw_models = payload.get("models", payload)
    if not isinstance(raw_models, dict) or not raw_models:
        raise ValueError("RCA model registry must define a non-empty 'models' mapping.")

    registry = {key: _coerce_registry_entry(str(key), value, base_dir=base_dir) for key, value in raw_models.items()}
    default_key = str(
        payload.get("default_model_key")
        or os.environ.get("AIOPS_RCA_DEFAULT_MODEL_KEY", "")
        or next(iter(registry.keys()))
    ).strip()
    if default_key not in registry:
        raise KeyError(f"Default RCA model key '{default_key}' is not present in the registry.")
    return default_key, registry


DEFAULT_MODEL_KEY, DEFAULT_MODEL_REGISTRY = load_model_registry()


def resolve_model_entry(
    model_key: str | None,
    registry: dict[str, ModelRegistryEntry] | None = None,
    default_model_key: str | None = None,
) -> ModelRegistryEntry:
    resolved_registry = registry or DEFAULT_MODEL_REGISTRY
    resolved_default = default_model_key or DEFAULT_MODEL_KEY
    key = str(model_key or resolved_default).strip() or resolved_default
    if key not in resolved_registry:
        known = ", ".join(sorted(resolved_registry.keys()))
        raise KeyError(f"Unknown RCA model_key '{key}'. Available: {known}")
    return resolved_registry[key]


def serialize_model_registry(
    registry: dict[str, ModelRegistryEntry] | None = None,
    default_model_key: str | None = None,
) -> list[dict[str, str]]:
    resolved_registry = registry or DEFAULT_MODEL_REGISTRY
    resolved_default = default_model_key or DEFAULT_MODEL_KEY
    items: list[dict[str, str]] = []
    for key, entry in resolved_registry.items():
        items.append(
            {
                "model_key": key,
                "label": entry.label,
                "artifact_dir": str(entry.artifact_dir),
                "is_default": str(key == resolved_default).lower(),
            }
        )
    return items


def _resolve_artifact_binary(artifact_dir: Path, model_config: dict[str, Any], *keys: str) -> Path:
    for key in keys:
        value = str(model_config.get(key, "")).strip()
        if value:
            return artifact_dir / value
    raise ValueError(f"Missing artifact filename in model_config. Expected one of: {', '.join(keys)}")


def _load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and "state_dict" in payload:
        return payload["state_dict"]
    if isinstance(payload, dict):
        return payload
    raise TypeError(f"Unsupported torch payload in {path}")


def _normalize_state_dict(model_type: str, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if model_type == "simple_graph_attention":
        remapped: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            if key.startswith("ff."):
                remapped[f"ffn.{key[3:]}"] = value
            else:
                remapped[key] = value
        return remapped
    return state_dict


def _load_torch_model(
    artifact_dir: Path,
    model_type: str,
    model_config: dict[str, Any],
    device: torch.device,
):
    weight_path = _resolve_artifact_binary(
        artifact_dir,
        model_config,
        "state_dict_artifact",
        "weights_artifact",
        "model_artifact",
    )
    state_dict = _load_state_dict(weight_path)
    state_dict = _normalize_state_dict(model_type, state_dict)

    if model_type == "simple_graph_attention":
        model = SimpleGraphAttention(
            in_dim=int(model_config["in_dim"]),
            hidden_dim=int(model_config["hidden_dim"]),
            dropout=float(model_config["dropout"]),
        )
    elif model_type == "hetero_telemetry_gnn":
        feature_groups = model_config.get("feature_groups") or infer_feature_groups(int(model_config["in_dim"]))
        model = HeteroTelemetryGNN(
            feature_groups={key: list(value) for key, value in feature_groups.items()},
            hidden_dim=int(model_config["hidden_dim"]),
            dropout=float(model_config["dropout"]),
            num_layers=int(model_config.get("num_layers", 2)),
        )
    else:
        raise ValueError(f"Unsupported torch RCA model_type: {model_type}")

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def _load_rf_model(artifact_dir: Path, model_config: dict[str, Any]):
    joblib_path = _resolve_artifact_binary(
        artifact_dir,
        model_config,
        "joblib_artifact",
        "model_artifact",
        "state_dict_artifact",
    )
    payload = joblib.load(joblib_path)
    if isinstance(payload, dict) and "model" in payload:
        model = payload["model"]
        feature_cols = list(payload.get("feature_cols", []))
    else:
        model = payload
        feature_cols = list(model_config.get("feature_cols", []))
    return model, feature_cols


def load_artifacts(artifact_dir: Path | None = None, device: str = "cpu") -> LoadedRcaArtifacts:
    artifact_dir = Path(artifact_dir or DEFAULT_ARTIFACT_DIR)
    inference_config = _read_json(artifact_dir / "inference_config.json")
    model_config = _read_json(artifact_dir / "model_config.json")
    resolved_device = torch.device(device)

    model_type = str(
        inference_config.get("model_type")
        or model_config.get("model_type")
        or "simple_graph_attention"
    ).strip()

    feature_cols: list[str] = list(model_config.get("feature_cols", []))
    if model_type == "random_forest_service_ranker":
        model, payload_feature_cols = _load_rf_model(artifact_dir, model_config)
        feature_cols = payload_feature_cols or feature_cols
    else:
        model = _load_torch_model(
            artifact_dir=artifact_dir,
            model_type=model_type,
            model_config=model_config,
            device=resolved_device,
        )
        if not feature_cols and "in_dim" in model_config:
            feature_cols = [f"feature_{idx}" for idx in range(int(model_config["in_dim"]))]

    return LoadedRcaArtifacts(
        artifact_dir=artifact_dir,
        inference_config=inference_config,
        model_config=model_config,
        model=model,
        device=resolved_device,
        model_type=model_type,
        feature_cols=feature_cols,
    )
