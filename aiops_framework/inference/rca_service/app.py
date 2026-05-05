from __future__ import annotations

import os
from typing import Any

import torch
from fastapi import FastAPI, HTTPException

from .model_loader import (
    DEFAULT_MODEL_KEY,
    DEFAULT_MODEL_REGISTRY,
    LoadedRcaArtifacts,
    load_artifacts,
    resolve_model_entry,
    serialize_model_registry,
)
from .predictor import predict_graph
from .schemas import GraphPredictRequest, GraphPredictResponse, RankedNode


device_name = "cuda" if torch.cuda.is_available() and os.environ.get("AIOPS_RCA_DEVICE", "cpu") == "cuda" else "cpu"
MODEL_CACHE: dict[str, LoadedRcaArtifacts] = {}
app = FastAPI(title="AIOps RCA Inference Service", version="0.2.0")


def _available_models() -> list[dict[str, str]]:
    return serialize_model_registry(DEFAULT_MODEL_REGISTRY, DEFAULT_MODEL_KEY)


def _get_artifacts(model_key: str | None = None) -> tuple[str, LoadedRcaArtifacts]:
    try:
        entry = resolve_model_entry(model_key, DEFAULT_MODEL_REGISTRY, DEFAULT_MODEL_KEY)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if entry.model_key not in MODEL_CACHE:
        MODEL_CACHE[entry.model_key] = load_artifacts(entry.artifact_dir, device=device_name)
    return entry.model_key, MODEL_CACHE[entry.model_key]


@app.get("/health")
def health() -> dict[str, Any]:
    model_key, artifacts = _get_artifacts()
    return {
        "status": "ok",
        "artifact_dir": str(artifacts.artifact_dir),
        "default_model_key": DEFAULT_MODEL_KEY,
        "model_key": model_key,
        "model_name": artifacts.inference_config.get("model_name", artifacts.artifact_dir.name),
        "model_type": artifacts.model_type,
        "device": str(artifacts.device),
        "feature_count": len(artifacts.feature_cols),
        "available_models": _available_models(),
    }


@app.get("/metadata")
def metadata(model_key: str | None = None) -> dict[str, Any]:
    resolved_model_key, artifacts = _get_artifacts(model_key)
    return {
        "artifact_dir": str(artifacts.artifact_dir),
        "default_model_key": DEFAULT_MODEL_KEY,
        "model_key": resolved_model_key,
        "available_models": _available_models(),
        "model_config": artifacts.model_config,
        "inference_config": artifacts.inference_config,
        "model_type": artifacts.model_type,
        "feature_count": len(artifacts.feature_cols),
        "feature_cols": artifacts.feature_cols,
    }


@app.post("/predict/graph", response_model=GraphPredictResponse)
def predict(payload: GraphPredictRequest) -> GraphPredictResponse:
    model_key, artifacts = _get_artifacts(payload.model_key)
    result = predict_graph(
        artifacts=artifacts,
        x_rows=payload.x,
        edge_index_rows=payload.edge_index,
        node_names=payload.node_names,
        top_k=payload.top_k,
        graph_id=payload.graph_id,
        metadata=payload.metadata,
    )
    result_metadata = dict(result.get("metadata") or {})
    result_metadata.setdefault("model_key", model_key)
    return GraphPredictResponse(
        graph_id=result["graph_id"],
        model_key=model_key,
        top1=RankedNode(**result["top1"]),
        topk=[RankedNode(**item) for item in result["topk"]],
        model_name=result["model_name"],
        model_type=result["model_type"],
        metadata=result_metadata,
    )
