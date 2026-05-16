from __future__ import annotations

from pathlib import Path

import pandas as pd

from .io_utils import load_json


DEFAULT_SERVICE_METADATA = {
    "service_role": "unknown",
    "service_tier": "unknown",
    "criticality": "unknown",
    "is_entrypoint": 0,
    "is_stateful": 0,
}


def _normalize_service_name(service_name: str) -> str:
    normalized = str(service_name or "unknown").strip().lower()
    for suffix in ("-staging", "_staging"):
        if normalized.endswith(suffix):
            return normalized[: -len(suffix)]
    return normalized


def load_service_catalog(path: Path) -> pd.DataFrame:
    payload = load_json(path)
    catalog_df = pd.DataFrame(payload)
    if catalog_df.empty:
        return pd.DataFrame(columns=["service_name", *DEFAULT_SERVICE_METADATA.keys()])

    catalog_df["service_name"] = catalog_df["service_name"].astype(str).str.lower().str.strip()
    for key, default_value in DEFAULT_SERVICE_METADATA.items():
        if key not in catalog_df.columns:
            catalog_df[key] = default_value
        catalog_df[key] = catalog_df[key].fillna(default_value)
    return catalog_df


def service_lookup(catalog_df: pd.DataFrame) -> dict[str, dict[str, object]]:
    if catalog_df.empty:
        return {}
    return {
        _normalize_service_name(row["service_name"]): {
            "service_role": row["service_role"],
            "service_tier": row["service_tier"],
            "criticality": row["criticality"],
            "is_entrypoint": int(row["is_entrypoint"]),
            "is_stateful": int(row["is_stateful"]),
        }
        for row in catalog_df.to_dict(orient="records")
    }


def map_service_metadata(service_name: str, lookup: dict[str, dict[str, object]]) -> dict[str, object]:
    normalized_name = _normalize_service_name(service_name)
    metadata = lookup.get(normalized_name, DEFAULT_SERVICE_METADATA)
    return {
        "service_name": normalized_name,
        "service_role": metadata["service_role"],
        "service_tier": metadata["service_tier"],
        "criticality": metadata["criticality"],
        "is_entrypoint": int(metadata["is_entrypoint"]),
        "is_stateful": int(metadata["is_stateful"]),
    }
