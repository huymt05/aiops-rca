from __future__ import annotations

import os
from pathlib import Path
from typing import Any

DEFAULT_GRAPH_ROOT = Path(
    os.environ.get(
        "AIOPS_DASHBOARD_GRAPH_ROOT",
        r"D:\HOCTAP\2025-2026\HK2\DACN\microservices-demo\data_rca_balanced_v3\processed\rca\graph_tensors",
    )
)


WINDOW_PRESETS: dict[str, dict[str, float]] = {
    "healthy": {
        "trace_count": 100,
        "service_count": 6,
        "service_role_count": 5,
        "avg_trace_duration_ms": 120,
        "p95_trace_duration_ms": 350,
        "max_trace_duration_ms": 800,
        "error_trace_ratio": 0.01,
        "error_span_ratio": 0.001,
        "request_fanout_mean": 4,
        "critical_path_mean_ms": 110,
        "latency_cv": 1.2,
        "mean_span_count": 18,
        "call_type_diversity": 3,
        "entrypoint_trace_ratio": 1,
        "stateful_trace_ratio": 0,
    },
    "suspicious": {
        "trace_count": 220,
        "service_count": 8,
        "service_role_count": 6,
        "avg_trace_duration_ms": 4200,
        "p95_trace_duration_ms": 9000,
        "max_trace_duration_ms": 18000,
        "error_trace_ratio": 0.42,
        "error_span_ratio": 0.28,
        "request_fanout_mean": 7.0,
        "critical_path_mean_ms": 5200,
        "latency_cv": 4.1,
        "mean_span_count": 34,
        "call_type_diversity": 6,
        "entrypoint_trace_ratio": 1,
        "stateful_trace_ratio": 0.3,
    },
}


def list_graph_samples(limit: int = 30) -> list[dict[str, Any]]:
    if not DEFAULT_GRAPH_ROOT.exists():
        return []
    items = []
    for path in sorted(DEFAULT_GRAPH_ROOT.glob("*.pt"))[:limit]:
        items.append({"name": path.name, "path": str(path)})
    return items


def load_graph_payload(sample_name: str) -> dict[str, Any]:
    sample_name = str(sample_name or "").strip()
    if not sample_name:
        raise ValueError("Please choose a graph sample before running demo analysis.")
    path = DEFAULT_GRAPH_ROOT / sample_name
    if not path.exists():
        raise FileNotFoundError(f"Graph tensor not found: {path}")

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "Demo graph sample loading requires the optional 'torch' dependency, which is not installed in this dashboard runtime."
        ) from exc

    obj = torch.load(path, map_location="cpu")
    edge_rows = obj["edge_index"].t().tolist() if obj["edge_index"].numel() > 0 else []
    return {
        "graph_id": str(obj["graph_id"]),
        "node_names": list(obj["node_names"]),
        "node_roles": list(obj.get("node_roles", [])),
        "x": obj["x"].tolist(),
        "edge_index": edge_rows,
        "top_k": 3,
        "metadata": {"run_id": str(obj["run_id"]), "sample_name": path.name},
    }


def build_demo_pipeline_payload(sample_name: str, preset: str, run_rca_on_any_input: bool) -> dict[str, Any]:
    features = WINDOW_PRESETS.get(preset, WINDOW_PRESETS["healthy"])
    graph_payload = load_graph_payload(sample_name)
    return {
        "window": {
            "features": features,
            "metadata": {
                "run_id": graph_payload["metadata"].get("run_id", "demo_run"),
                "window_id": f"{preset}_window",
                "preset": preset,
            },
        },
        "graph": graph_payload,
        "run_rca_on_any_input": run_rca_on_any_input,
    }
