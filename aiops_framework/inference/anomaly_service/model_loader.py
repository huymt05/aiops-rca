from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from aiops_framework.inference.common.artifact_registry import DEFAULT_STAGE, resolve_artifact_dir


DEFAULT_MODEL_ROOT = Path(
    os.environ.get(
        "AIOPS_ANOMALY_MODEL_ROOT",
        r"D:\HOCTAP\2025-2026\HK2\DACN\microservices-demo\data_anomaly_balanced_v3\models",
    )
)
DEFAULT_MODEL_STAGE = os.environ.get("AIOPS_ANOMALY_MODEL_STAGE", DEFAULT_STAGE).strip() or DEFAULT_STAGE
_ARTIFACT_DIR_OVERRIDE = os.environ.get("AIOPS_ANOMALY_ARTIFACT_DIR", "").strip()
DEFAULT_ARTIFACT_DIR = Path(_ARTIFACT_DIR_OVERRIDE) if _ARTIFACT_DIR_OVERRIDE else resolve_artifact_dir(DEFAULT_MODEL_ROOT, DEFAULT_MODEL_STAGE)
THRESHOLD_OVERRIDE = os.environ.get("AIOPS_ANOMALY_THRESHOLD_OVERRIDE", "").strip()


@dataclass
class LoadedAnomalyArtifacts:
    artifact_dir: Path
    inference_config: dict[str, object]
    feature_columns: list[str]
    imputer: object
    model_kind: str
    single_model: object | None
    ensemble_members: list[tuple[str, object]]


def load_artifacts(artifact_dir: Path | None = None) -> LoadedAnomalyArtifacts:
    artifact_dir = Path(artifact_dir or DEFAULT_ARTIFACT_DIR)
    inference_config = json.loads((artifact_dir / "inference_config.json").read_text(encoding="utf-8-sig"))
    if THRESHOLD_OVERRIDE:
        inference_config = dict(inference_config)
        inference_config["threshold"] = float(THRESHOLD_OVERRIDE)
        inference_config["threshold_source"] = "env_override"
    feature_columns = json.loads((artifact_dir / "feature_columns.json").read_text(encoding="utf-8"))
    imputer = joblib.load(artifact_dir / str(inference_config["imputer_artifact"]))
    # Older serialized SimpleImputer artifacts may only carry `_fit_dtype`,
    # while newer sklearn runtime paths expect `_fill_dtype` during transform.
    if not hasattr(imputer, "_fill_dtype") and hasattr(imputer, "_fit_dtype"):
        setattr(imputer, "_fill_dtype", getattr(imputer, "_fit_dtype"))

    model_artifacts = inference_config["model_artifacts"]
    ensemble_members: list[tuple[str, object]] = []
    single_model = None
    model_kind = str(model_artifacts["kind"])
    if model_kind == "ensemble":
        for member in model_artifacts["members"]:
            member_kind = str(member["kind"])
            model = joblib.load(artifact_dir / str(member["artifact"]))
            ensemble_members.append((member_kind, model))
    else:
        single_model = joblib.load(artifact_dir / str(model_artifacts["artifact"]))

    return LoadedAnomalyArtifacts(
        artifact_dir=artifact_dir,
        inference_config=inference_config,
        feature_columns=feature_columns,
        imputer=imputer,
        model_kind=model_kind,
        single_model=single_model,
        ensemble_members=ensemble_members,
    )


def transform_features(artifacts: LoadedAnomalyArtifacts, rows: list[dict[str, float]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    for column in artifacts.feature_columns:
        if column not in df.columns:
            df[column] = np.nan
    df = df[artifacts.feature_columns].copy()
    transformed = artifacts.imputer.transform(df)
    return pd.DataFrame(transformed, columns=artifacts.feature_columns)
