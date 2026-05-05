from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd

from aiops_framework.registry.service_catalog import load_service_catalog, service_lookup
from pipeline.rca_data_pipeline.feature_engineering import (
    CRITICALITY_TO_ID,
    ROLE_TO_ID,
    TIER_TO_ID,
    build_window_features,
    clean_spans,
)
from pipeline.rca_data_pipeline.jaeger_parser import build_service_edges, parse_jaeger_payload


DEFAULT_JAEGER_URL = os.environ.get("AIOPS_LIVE_JAEGER_URL", "http://127.0.0.1:16686")
DEFAULT_PROMETHEUS_URL = os.environ.get("AIOPS_LIVE_PROM_URL", "http://127.0.0.1:9090")
DEFAULT_SYSTEM_ID = os.environ.get("AIOPS_LIVE_SYSTEM_ID", "online-boutique")
DEFAULT_SOURCE_SERVICE = os.environ.get("AIOPS_LIVE_SOURCE_SERVICE", "frontend")
JAEGER_INTERNAL_SERVICES = {"jaeger"}
BENCHMARK_WINDOW_SECONDS = 300
BENCHMARK_SERVICE_ORDER = [
    "frontend",
    "cartservice",
    "checkoutservice",
    "currencyservice",
    "emailservice",
    "paymentservice",
    "productcatalogservice",
    "recommendationservice",
    "shippingservice",
    "adservice",
    "redis-cart",
]
SERVICE_ALIASES = {
    "frontendservice": "frontend",
    "redis": "redis-cart",
}


def _load_json(url: str) -> dict[str, Any]:
    with urlopen(url, timeout=30) as response:
        return json.loads(response.read().decode("utf-8-sig"))


def fetch_jaeger_services(jaeger_url: str = DEFAULT_JAEGER_URL) -> list[str]:
    payload = _load_json(f"{jaeger_url.rstrip('/')}/api/services")
    services = [str(item) for item in payload.get("data", []) if str(item)]
    app_services = [svc for svc in services if svc not in JAEGER_INTERNAL_SERVICES]
    return app_services or services


def _resolve_source_services(jaeger_url: str, source_service: str) -> list[str]:
    requested = (source_service or "").strip()
    if requested.lower() in {"all", "*"}:
        return fetch_jaeger_services(jaeger_url)
    services = [item.strip() for item in requested.split(",") if item.strip()]
    return services or [DEFAULT_SOURCE_SERVICE]


def _merge_jaeger_payloads(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    merged: dict[str, Any] = {"data": [], "total": 0, "limit": 0, "offset": 0, "errors": None}
    seen_trace_ids: set[str] = set()
    errors: list[Any] = []
    for payload in payloads:
        for trace in payload.get("data", []):
            trace_id = str(trace.get("traceID") or trace.get("traceId") or "")
            if trace_id and trace_id in seen_trace_ids:
                continue
            if trace_id:
                seen_trace_ids.add(trace_id)
            merged["data"].append(trace)
        if payload.get("errors"):
            errors.append(payload["errors"])
    merged["total"] = len(merged["data"])
    merged["errors"] = errors or None
    return merged


def _fetch_jaeger_payload_for_service(
    jaeger_url: str,
    source_service: str,
    lookback_minutes: int,
    query_limit: int,
) -> dict[str, Any]:
    end = datetime.now(timezone.utc)
    start = end - timedelta(minutes=max(1, int(lookback_minutes)))
    params = urlencode(
        {
            "service": source_service,
            "start": int(start.timestamp() * 1_000_000),
            "end": int(end.timestamp() * 1_000_000),
            "limit": int(query_limit),
        }
    )
    url = f"{jaeger_url.rstrip('/')}/api/traces?{params}"
    return _load_json(url)


def fetch_jaeger_payload(
    jaeger_url: str = DEFAULT_JAEGER_URL,
    source_service: str = DEFAULT_SOURCE_SERVICE,
    lookback_minutes: int = 2,
    query_limit: int = 150,
) -> dict[str, Any]:
    services = _resolve_source_services(jaeger_url, source_service)
    payloads = [
        _fetch_jaeger_payload_for_service(
            jaeger_url=jaeger_url,
            source_service=service,
            lookback_minutes=lookback_minutes,
            query_limit=query_limit,
        )
        for service in services
    ]
    return _merge_jaeger_payloads(payloads)


def _extract_prom_value(payload: dict[str, Any]) -> float | None:
    data = payload.get("data", {})
    result = data.get("result", [])
    if not result:
        scalar = data.get("result")
        if isinstance(scalar, list) and len(scalar) == 2:
            try:
                return float(scalar[1])
            except Exception:
                return None
        return None
    first = result[0]
    value = first.get("value")
    if isinstance(value, list) and len(value) == 2:
        try:
            return float(value[1])
        except Exception:
            return None
    return None


def _prom_query(prometheus_url: str, query: str) -> float | None:
    params = urlencode({"query": query})
    url = f"{prometheus_url.rstrip('/')}/api/v1/query?{params}"
    payload = _load_json(url)
    return _extract_prom_value(payload)


def fetch_prometheus_snapshot(prometheus_url: str = DEFAULT_PROMETHEUS_URL) -> dict[str, Any]:
    query_candidates = {
        "up_targets": ["sum(up)"],
        "cpu_usage": ['sum(rate(container_cpu_usage_seconds_total{container!="",pod!=""}[2m]))'],
        "memory_usage": ['sum(container_memory_working_set_bytes{container!="",pod!=""})'],
        "request_rate": [
            "sum(rate(http_server_requests_seconds_count[1m]))",
            "sum(rate(traces_spanmetrics_calls_total[1m]))",
        ],
        "error_rate": [
            'sum(rate(http_server_requests_seconds_count{status=~"5.."}[1m]))',
            'sum(rate(traces_spanmetrics_calls_total{status_code="STATUS_CODE_ERROR"}[1m]))',
        ],
    }
    snapshot: dict[str, Any] = {"prometheus_url": prometheus_url, "status": "ok", "values": {}, "missing": []}
    for label, queries in query_candidates.items():
        value = None
        for query in queries:
            try:
                value = _prom_query(prometheus_url, query)
                if value is not None:
                    snapshot["values"][label] = {"value": value, "query": query}
                    break
            except Exception:
                continue
        if value is None:
            snapshot["missing"].append(label)
    if not snapshot["values"]:
        snapshot["status"] = "unavailable"
    return snapshot


def _normalize_benchmark_service_name(name: str) -> str:
    return SERVICE_ALIASES.get(str(name or "").strip().lower(), str(name or "").strip().lower())


def _safe_mean(frame: pd.DataFrame, column: str) -> float:
    return float(frame[column].mean()) if not frame.empty else 0.0


def _safe_max(frame: pd.DataFrame, column: str) -> float:
    return float(frame[column].max()) if not frame.empty else 0.0


def _stat_features(prefix: str, pre_values: pd.Series, post_values: pd.Series) -> dict[str, float]:
    pre = pd.to_numeric(pre_values, errors="coerce").fillna(0.0)
    post = pd.to_numeric(post_values, errors="coerce").fillna(0.0)
    eps = 1e-6

    pre_mean = float(pre.mean()) if len(pre) else 0.0
    post_mean = float(post.mean()) if len(post) else 0.0
    pre_max = float(pre.max()) if len(pre) else 0.0
    post_max = float(post.max()) if len(post) else 0.0

    return {
        f"{prefix}_pre_mean": pre_mean,
        f"{prefix}_post_mean": post_mean,
        f"{prefix}_delta_mean": post_mean - pre_mean,
        f"{prefix}_ratio_mean": post_mean / (pre_mean + eps),
        f"{prefix}_pre_max": pre_max,
        f"{prefix}_post_max": post_max,
        f"{prefix}_delta_max": post_max - pre_max,
        f"{prefix}_ratio_max": post_max / (pre_max + eps),
    }


def _add_case_relative_features(df: pd.DataFrame) -> pd.DataFrame:
    excluded = {
        "service",
        "service_role",
        "service_tier",
        "criticality",
        "is_entrypoint",
        "is_stateful",
    }
    numeric_cols = [c for c in df.columns if c not in excluded and pd.api.types.is_numeric_dtype(df[c])]

    derived = {}
    for col in numeric_cols:
        case_mean = df[col].mean()
        case_std = df[col].std()
        if not case_std:
            case_std = 1.0
        rank_desc = df[col].rank(ascending=False, method="min")

        derived[f"{col}_case_z"] = (df[col] - case_mean) / case_std
        derived[f"{col}_case_rank_desc"] = rank_desc
        derived[f"{col}_is_case_top1"] = (rank_desc <= 1).astype(int)
        derived[f"{col}_is_case_top2"] = (rank_desc <= 2).astype(int)
        derived[f"{col}_is_case_top3"] = (rank_desc <= 3).astype(int)

    return pd.concat([df.reset_index(drop=True), pd.DataFrame(derived).reset_index(drop=True)], axis=1)


def _resolve_runtime_service_order(catalog_df: pd.DataFrame) -> list[str]:
    catalog_services = {
        _normalize_benchmark_service_name(name)
        for name in catalog_df["service_name"].astype(str).tolist()
    }
    if set(BENCHMARK_SERVICE_ORDER).issubset(catalog_services | {"redis-cart"}):
        return list(BENCHMARK_SERVICE_ORDER)
    return sorted(catalog_services)


def _build_benchmark_feature_graph_payload(
    spans_df: pd.DataFrame,
    catalog_df: pd.DataFrame,
    run_id: str,
    window_id: str,
    lookback_minutes: int,
) -> dict[str, Any]:
    service_order = _resolve_runtime_service_order(catalog_df)
    spans = spans_df.copy()
    spans["service_name"] = spans["service_name"].astype(str).map(_normalize_benchmark_service_name)
    spans["ts_sec"] = pd.to_numeric(spans["start_time_ms"], errors="coerce").fillna(0.0) / 1000.0
    spans["duration_num"] = pd.to_numeric(spans["duration_ms"], errors="coerce").fillna(0.0)
    spans["status_num"] = pd.to_numeric(spans["is_error"], errors="coerce").fillna(0.0)

    max_ts = float(spans["ts_sec"].max()) if not spans.empty else 0.0
    split_ts = max_ts - max(1, int(lookback_minutes)) * 30
    pre_spans = spans[spans["ts_sec"] < split_ts]
    post_spans = spans[spans["ts_sec"] >= split_ts]

    rows: list[dict[str, Any]] = []
    for svc in service_order:
        svc_pre = pre_spans[pre_spans["service_name"] == svc]
        svc_post = post_spans[post_spans["service_name"] == svc]
        role_row = catalog_df[catalog_df["service_name"].astype(str).map(_normalize_benchmark_service_name) == svc]

        if role_row.empty:
            service_role = "unknown"
            service_tier = "unknown"
            criticality = "unknown"
            is_entrypoint = 0
            is_stateful = 0
        else:
            meta = role_row.iloc[0]
            service_role = str(meta.get("service_role", "unknown"))
            service_tier = str(meta.get("service_tier", "unknown"))
            criticality = str(meta.get("criticality", "unknown"))
            is_entrypoint = int(meta.get("is_entrypoint", 0))
            is_stateful = int(meta.get("is_stateful", 0))

        row = {
            "service": svc,
            "service_role": service_role,
            "service_tier": service_tier,
            "criticality": criticality,
            "is_entrypoint": is_entrypoint,
            "is_stateful": is_stateful,
        }

        zero_metric = pd.Series(dtype="float64")
        for prefix in ("cpu", "mem", "socket"):
            row.update(_stat_features(prefix, zero_metric, zero_metric))

        row.update(
            _stat_features(
                "workload",
                svc_pre["duration_num"] * 0 + len(svc_pre),
                svc_post["duration_num"] * 0 + len(svc_post),
            )
        )
        row.update(
            _stat_features(
                "error",
                svc_pre["status_num"],
                svc_post["status_num"],
            )
        )
        row.update(
            _stat_features(
                "lat50",
                svc_pre["duration_num"],
                svc_post["duration_num"],
            )
        )
        row.update(
            _stat_features(
                "lat90",
                svc_pre["duration_num"],
                svc_post["duration_num"],
            )
        )

        row.update(
            {
                "log_total_pre": 0,
                "log_total_post": 0,
                "log_total_delta": 0,
                "log_error_pre": 0,
                "log_error_post": 0,
                "log_error_delta": 0,
                "log_warn_pre": 0,
                "log_warn_post": 0,
                "log_warn_delta": 0,
            }
        )

        pre_trace_count = len(svc_pre)
        post_trace_count = len(svc_post)
        pre_trace_error = int((svc_pre["status_num"] > 0).sum()) if pre_trace_count else 0
        post_trace_error = int((svc_post["status_num"] > 0).sum()) if post_trace_count else 0
        eps = 1e-6
        pre_dur_mean = _safe_mean(svc_pre, "duration_num")
        post_dur_mean = _safe_mean(svc_post, "duration_num")
        pre_dur_max = _safe_max(svc_pre, "duration_num")
        post_dur_max = _safe_max(svc_post, "duration_num")

        row.update(
            {
                "trace_count_pre": pre_trace_count,
                "trace_count_post": post_trace_count,
                "trace_count_delta": post_trace_count - pre_trace_count,
                "trace_count_ratio": post_trace_count / (pre_trace_count + eps),
                "trace_error_pre": pre_trace_error,
                "trace_error_post": post_trace_error,
                "trace_error_delta": post_trace_error - pre_trace_error,
                "trace_duration_pre_mean": pre_dur_mean,
                "trace_duration_post_mean": post_dur_mean,
                "trace_duration_delta_mean": post_dur_mean - pre_dur_mean,
                "trace_duration_ratio_mean": post_dur_mean / (pre_dur_mean + eps),
                "trace_duration_pre_max": pre_dur_max,
                "trace_duration_post_max": post_dur_max,
                "trace_duration_delta_max": post_dur_max - pre_dur_max,
                "trace_duration_ratio_max": post_dur_max / (pre_dur_max + eps),
            }
        )
        rows.append(row)

    feature_df = _add_case_relative_features(pd.DataFrame(rows))
    feature_cols = [
        c
        for c in feature_df.columns
        if c
        not in {"service", "service_role", "service_tier", "criticality", "is_entrypoint", "is_stateful"}
        and pd.api.types.is_numeric_dtype(feature_df[c])
    ]

    edges_df = build_service_edges(spans_df)
    node_to_idx = {name: idx for idx, name in enumerate(service_order)}
    edge_index = []
    for edge in edges_df.groupby(["src_service", "dst_service"])["edge_count"].sum().reset_index().itertuples(index=False):
        src = _normalize_benchmark_service_name(str(getattr(edge, "src_service")))
        dst = _normalize_benchmark_service_name(str(getattr(edge, "dst_service")))
        if src in node_to_idx and dst in node_to_idx:
            edge_index.append([node_to_idx[src], node_to_idx[dst]])

    return {
        "graph_id": f"{run_id}__{window_id}",
        "node_names": service_order,
        "node_roles": feature_df["service_role"].astype(str).tolist(),
        "x": feature_df[feature_cols].fillna(0.0).astype(float).values.tolist(),
        "edge_index": edge_index,
        "top_k": 3,
        "metadata": {
            "run_id": run_id,
            "window_id": window_id,
            "source": "live_jaeger_benchmark_features",
            "feature_count": len(feature_cols),
            "feature_schema": "re2_ob_service_features_v1",
            "lookback_minutes": lookback_minutes,
        },
    }


def build_live_graph_payload(spans_df, run_id: str, window_id: str) -> dict[str, Any]:
    node_stats = (
        spans_df.groupby("service_name")
        .agg(
            avg_latency_ms=("duration_ms", "mean"),
            p95_latency_ms=("duration_ms", lambda s: float(s.quantile(0.95))),
            error_rate=("is_error", "mean"),
            request_count=("span_id", "count"),
            service_role=("service_role", "first"),
            service_tier=("service_tier", "first"),
            criticality=("criticality", "first"),
            is_entrypoint=("is_entrypoint", "max"),
            is_stateful=("is_stateful", "max"),
        )
        .reset_index()
    )
    total_requests = int(node_stats["request_count"].sum())
    edges_df = build_service_edges(spans_df)
    degree_in: dict[str, int] = {}
    degree_out: dict[str, int] = {}
    edges = []
    for edge in edges_df.groupby(["src_service", "dst_service"])["edge_count"].sum().reset_index().itertuples(index=False):
        src = str(getattr(edge, "src_service"))
        dst = str(getattr(edge, "dst_service"))
        weight = int(getattr(edge, "edge_count"))
        degree_out[src] = degree_out.get(src, 0) + weight
        degree_in[dst] = degree_in.get(dst, 0) + weight
        edges.append({"src": src, "dst": dst, "weight": weight})

    node_names = node_stats["service_name"].astype(str).tolist()
    node_roles = node_stats["service_role"].astype(str).tolist()
    node_to_idx = {name: idx for idx, name in enumerate(node_names)}
    x_rows: list[list[float]] = []
    for row in node_stats.itertuples(index=False):
        service_name = str(getattr(row, "service_name"))
        service_role = str(getattr(row, "service_role") or "unknown").lower()
        service_tier = str(getattr(row, "service_tier") or "unknown").lower()
        criticality = str(getattr(row, "criticality") or "unknown").lower()
        request_count = int(getattr(row, "request_count"))
        x_rows.append(
            [
                float(getattr(row, "avg_latency_ms")),
                float(getattr(row, "p95_latency_ms")),
                float(getattr(row, "error_rate")),
                float(request_count),
                float(request_count / total_requests) if total_requests else 0.0,
                float(degree_in.get(service_name, 0)),
                float(degree_out.get(service_name, 0)),
                float(ROLE_TO_ID.get(service_role, ROLE_TO_ID["unknown"])),
                float(TIER_TO_ID.get(service_tier, TIER_TO_ID["unknown"])),
                float(CRITICALITY_TO_ID.get(criticality, CRITICALITY_TO_ID["unknown"])),
                float(int(getattr(row, "is_entrypoint"))),
                float(int(getattr(row, "is_stateful"))),
            ]
        )

    edge_index = [
        [node_to_idx[e["src"]], node_to_idx[e["dst"]]]
        for e in edges
        if e["src"] in node_to_idx and e["dst"] in node_to_idx
    ]

    return {
        "graph_id": f"{run_id}__{window_id}",
        "node_names": node_names,
        "node_roles": node_roles,
        "x": x_rows,
        "edge_index": edge_index,
        "top_k": 3,
        "metadata": {"run_id": run_id, "window_id": window_id, "source": "live_jaeger"},
    }


def collect_live_inputs(
    system_id: str = DEFAULT_SYSTEM_ID,
    source_service: str = DEFAULT_SOURCE_SERVICE,
    jaeger_url: str = DEFAULT_JAEGER_URL,
    prometheus_url: str = DEFAULT_PROMETHEUS_URL,
    lookback_minutes: int = 2,
    query_limit: int = 150,
) -> dict[str, Any]:
    catalog_df = load_service_catalog(system_id)
    lookup = service_lookup(catalog_df)
    selected_services = _resolve_source_services(jaeger_url, source_service)

    payload = fetch_jaeger_payload(
        jaeger_url=jaeger_url,
        source_service=source_service,
        lookback_minutes=lookback_minutes,
        query_limit=query_limit,
    )
    run_id = f"live_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    window_id = "live_recent"
    spans_df, traces_df = parse_jaeger_payload(
        payload=payload,
        run_id=run_id,
        window_id=window_id,
        service_metadata_lookup=lookup,
        system_id=system_id,
    )
    spans_df = clean_spans(spans_df)
    if spans_df.empty:
        raise ValueError("No live spans were returned from Jaeger in the selected time window.")

    window_df = build_window_features(spans_df, run_catalog_df=None)
    if window_df.empty:
        raise ValueError("Unable to build live window features from the current trace window.")
    window_row = window_df.iloc[0].to_dict()

    graph_payload = _build_benchmark_feature_graph_payload(
        spans_df=spans_df,
        catalog_df=catalog_df,
        run_id=run_id,
        window_id=window_id,
        lookback_minutes=lookback_minutes,
    )
    metrics_snapshot = fetch_prometheus_snapshot(prometheus_url=prometheus_url)
    trace_snapshot = {
        "trace_count": int(traces_df["trace_id"].nunique()) if not traces_df.empty else 0,
        "span_count": int(len(spans_df)),
        "service_count": int(spans_df["service_name"].nunique()),
        "source_service": source_service,
        "selected_services": selected_services,
        "jaeger_url": jaeger_url,
        "lookback_minutes": lookback_minutes,
    }
    return {
        "window": {
            "features": {k: float(v) for k, v in window_row.items() if k not in {"system_id", "run_id", "window_id", "window_phase"}},
            "metadata": {
                "run_id": run_id,
                "window_id": str(window_row.get("window_id", window_id)),
                "window_phase": str(window_row.get("window_phase", "steady")),
                "source": "live_jaeger",
            },
        },
        "graph": graph_payload,
        "trace_snapshot": trace_snapshot,
        "metrics_snapshot": metrics_snapshot,
    }
