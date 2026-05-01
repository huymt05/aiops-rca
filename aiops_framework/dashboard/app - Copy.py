from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from aiops_framework.inference.orchestrator.clients import get_json, post_json
from aiops_framework.inference.anomaly_service.schemas import WindowPredictResponse
from aiops_framework.inference.rca_service.schemas import GraphPredictResponse, RankedNode
from aiops_framework.registry.system_catalog import get_system, list_systems

from .demo_data import DEFAULT_GRAPH_ROOT, WINDOW_PRESETS, build_demo_pipeline_payload, list_graph_samples
from .live_data import DEFAULT_JAEGER_URL, DEFAULT_PROMETHEUS_URL, DEFAULT_SOURCE_SERVICE, DEFAULT_SYSTEM_ID, collect_live_inputs
from .logs import fetch_recent_logs
from .policy import recommend_actions
from .store import (
    create_monitoring_event,
    get_monitoring_event,
    init_store,
    list_audit_logs,
    list_monitoring_events,
    save_feedback,
    write_audit_log,
)


DASHBOARD_DIR = Path(__file__).resolve().parent
STATIC_DIR = DASHBOARD_DIR / "static"
TEMPLATES_DIR = DASHBOARD_DIR / "templates"

ANOMALY_BASE_URL = os.environ.get("AIOPS_DASHBOARD_ANOMALY_BASE_URL", "http://127.0.0.1:8000")
RCA_BASE_URL = os.environ.get("AIOPS_DASHBOARD_RCA_BASE_URL", "http://127.0.0.1:8001")
ORCHESTRATOR_BASE_URL = os.environ.get("AIOPS_DASHBOARD_ORCH_BASE_URL", "http://127.0.0.1:8002")
RECOVERY_MODE = os.environ.get("AIOPS_RECOVERY_MODE", "demo").strip().lower()
RECOVERY_NAMESPACE = os.environ.get("AIOPS_RECOVERY_NAMESPACE", "default").strip()
RECOVERY_TIMEOUT_SECONDS = int(os.environ.get("AIOPS_RECOVERY_TIMEOUT_SECONDS", "120"))

app = FastAPI(title="AIOps RCA Dashboard", version="0.1.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class DemoAnalyzeRequest(BaseModel):
    sample_name: str
    preset: str = "healthy"
    run_rca_on_any_input: bool = False


class RecoveryActionRequest(BaseModel):
    action: str
    service_name: str | None = None
    severity: str | None = None
    source: str = "dashboard"
    context: dict[str, Any] = {}


class FeedbackRequest(BaseModel):
    feedback: str
    actor: str = "operator"
    notes: str = ""
    context: dict[str, Any] = {}


RECOVERY_HISTORY: list[dict[str, Any]] = []


class LiveAnalyzeRequest(BaseModel):
    system_id: str = DEFAULT_SYSTEM_ID
    source_service: str = DEFAULT_SOURCE_SERVICE
    jaeger_url: str = DEFAULT_JAEGER_URL
    prometheus_url: str = DEFAULT_PROMETHEUS_URL
    lookback_minutes: int = 1
    query_limit: int = 150
    run_rca_on_any_input: bool = False


def _read_index_html() -> str:
    html = (TEMPLATES_DIR / "index_v2.html").read_text(encoding="utf-8")
    config = {
        "anomalyBaseUrl": ANOMALY_BASE_URL,
        "rcaBaseUrl": RCA_BASE_URL,
        "orchestratorBaseUrl": ORCHESTRATOR_BASE_URL,
        "jaegerUrl": DEFAULT_JAEGER_URL,
        "prometheusUrl": DEFAULT_PROMETHEUS_URL,
    }
    return html.replace("__DASHBOARD_CONFIG__", json.dumps(config, ensure_ascii=False))


@app.on_event("startup")
def startup() -> None:
    init_store()


def _best_effort_get(url: str) -> dict[str, Any]:
    try:
        return get_json(url)
    except Exception as exc:
        return {"status": "down", "error": str(exc)}


def _kubectl(args: list[str]) -> subprocess.CompletedProcess[str]:
    cmd = ["kubectl", "-n", RECOVERY_NAMESPACE, *args]
    return subprocess.run(cmd, check=True, text=True, capture_output=True)


def _deployment_payload(name: str) -> dict[str, Any]:
    result = _kubectl(["get", "deployment", name, "-o", "json"])
    return json.loads(result.stdout)


def _execute_recovery_action(action: str, service_name: str) -> dict[str, str]:
    if RECOVERY_MODE != "real":
        action_map = {
            "restart_pod": {
                "label": "Restart Pod",
                "status": "accepted",
                "notes": "Demo mode: would call Kubernetes API to restart the predicted service pod.",
            },
            "scale_service": {
                "label": "Scale Service",
                "status": "accepted",
                "notes": "Demo mode: would increase replicas or trigger HPA guidance for the predicted service.",
            },
            "alert_only": {
                "label": "Alert Only",
                "status": "accepted",
                "notes": "Demo mode: would notify the operator and wait for manual approval.",
            },
        }
        if action not in action_map:
            raise HTTPException(status_code=400, detail=f"Unsupported recovery action: {action}")
        return action_map[action]

    if action == "alert_only":
        return {
            "label": "Alert Only",
            "status": "accepted",
            "notes": f"Real mode: alert recorded for {service_name}; no Kubernetes mutation was executed.",
        }

    if action == "restart_pod":
        _kubectl(["rollout", "restart", f"deployment/{service_name}"])
        _kubectl(["rollout", "status", f"deployment/{service_name}", f"--timeout={RECOVERY_TIMEOUT_SECONDS}s"])
        return {
            "label": "Restart Pod",
            "status": "executed",
            "notes": (
                f"Real mode: restarted deployment/{service_name} in namespace {RECOVERY_NAMESPACE} "
                f"and rollout completed within {RECOVERY_TIMEOUT_SECONDS}s."
            ),
        }

    if action == "scale_service":
        deployment = _deployment_payload(service_name)
        current_replicas = int(deployment.get("spec", {}).get("replicas", 1) or 1)
        target_replicas = current_replicas + 1
        _kubectl(["scale", f"deployment/{service_name}", f"--replicas={target_replicas}"])
        _kubectl(["rollout", "status", f"deployment/{service_name}", f"--timeout={RECOVERY_TIMEOUT_SECONDS}s"])
        return {
            "label": "Scale Service",
            "status": "executed",
            "notes": (
                f"Real mode: scaled deployment/{service_name} in namespace {RECOVERY_NAMESPACE} "
                f"from {current_replicas} to {target_replicas} replicas."
            ),
        }

    raise HTTPException(status_code=400, detail=f"Unsupported recovery action: {action}")


def _run_pipeline_locally(pipeline_payload: dict[str, Any]) -> dict[str, Any]:
    anomaly_result = WindowPredictResponse(
        **post_json(f"{ANOMALY_BASE_URL}/predict/window", pipeline_payload["window"])
    ).model_dump()

    should_run_rca = bool(
        pipeline_payload.get("run_rca_on_any_input", False) or anomaly_result.get("is_anomaly", False)
    )
    if not should_run_rca or pipeline_payload.get("graph") is None:
        state = "anomaly_only" if pipeline_payload.get("graph") is None else "no_anomaly_skip_rca"
        return {
            "anomaly": anomaly_result,
            "rca": None,
            "pipeline_state": state,
            "metadata": {"reason": "RCA not triggered", "execution_mode": "dashboard_fallback"},
        }

    rca_raw = post_json(f"{RCA_BASE_URL}/predict/graph", pipeline_payload["graph"])
    rca_result = GraphPredictResponse(
        top1=RankedNode(**rca_raw["top1"]),
        topk=[RankedNode(**item) for item in rca_raw.get("topk", [])],
        graph_id=rca_raw.get("graph_id"),
        model_name=rca_raw["model_name"],
        model_type=rca_raw["model_type"],
        metadata=rca_raw.get("metadata", {}),
    ).model_dump()
    return {
        "anomaly": anomaly_result,
        "rca": rca_result,
        "pipeline_state": "anomaly_then_rca",
        "metadata": {"reason": "RCA triggered after anomaly detection", "execution_mode": "dashboard_fallback"},
    }


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return _read_index_html()


@app.get("/api/health")
def dashboard_health() -> dict[str, Any]:
    return {
        "dashboard": {
            "status": "ok",
            "graph_root": str(DEFAULT_GRAPH_ROOT),
            "recovery_mode": RECOVERY_MODE,
            "recovery_namespace": RECOVERY_NAMESPACE,
        },
        "anomaly": _best_effort_get(f"{ANOMALY_BASE_URL}/health"),
        "rca": _best_effort_get(f"{RCA_BASE_URL}/health"),
        "orchestrator": _best_effort_get(f"{ORCHESTRATOR_BASE_URL}/health"),
    }


@app.get("/api/metadata")
def dashboard_metadata() -> dict[str, Any]:
    return {
        "window_presets": list(WINDOW_PRESETS.keys()),
        "graph_root": str(DEFAULT_GRAPH_ROOT),
        "recovery": {
            "mode": RECOVERY_MODE,
            "namespace": RECOVERY_NAMESPACE,
            "timeout_seconds": RECOVERY_TIMEOUT_SECONDS,
        },
        "live_defaults": {
            "system_id": DEFAULT_SYSTEM_ID,
            "source_service": DEFAULT_SOURCE_SERVICE,
            "jaeger_url": DEFAULT_JAEGER_URL,
            "prometheus_url": DEFAULT_PROMETHEUS_URL,
        },
        "samples": list_graph_samples(),
        "systems": list_systems(),
        "anomaly": _best_effort_get(f"{ANOMALY_BASE_URL}/metadata"),
        "rca": _best_effort_get(f"{RCA_BASE_URL}/metadata"),
        "orchestrator": _best_effort_get(f"{ORCHESTRATOR_BASE_URL}/metadata"),
    }


@app.get("/api/systems")
def systems() -> dict[str, Any]:
    return {"items": list_systems(), "default_system_id": DEFAULT_SYSTEM_ID}


@app.get("/api/systems/{system_id}")
def system_detail(system_id: str) -> dict[str, Any]:
    try:
        return get_system(system_id)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/samples")
def samples() -> dict[str, Any]:
    return {"items": list_graph_samples(), "graph_root": str(DEFAULT_GRAPH_ROOT)}


@app.get("/api/monitoring-events")
def monitoring_events(limit: int = 50, system_id: str | None = None) -> dict[str, Any]:
    return {"items": list_monitoring_events(limit=limit, system_id=system_id)}


@app.get("/api/monitoring-events/{event_id}")
def monitoring_event_detail(event_id: int) -> dict[str, Any]:
    event = get_monitoring_event(event_id)
    if event is None:
        raise HTTPException(status_code=404, detail=f"Monitoring event not found: {event_id}")
    return event


@app.post("/api/monitoring-events/{event_id}/feedback")
def record_feedback(event_id: int, payload: FeedbackRequest) -> dict[str, Any]:
    allowed = {"accepted_incident", "rejected_false_positive", "unknown"}
    if payload.feedback not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported feedback: {payload.feedback}")
    try:
        return save_feedback(
            event_id=event_id,
            feedback=payload.feedback,
            actor=payload.actor,
            notes=payload.notes,
            payload=payload.context,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/audit-logs")
def audit_logs(limit: int = 50) -> dict[str, Any]:
    return {"items": list_audit_logs(limit=limit)}


@app.get("/api/logs/recent")
def recent_logs(system_id: str = DEFAULT_SYSTEM_ID, service_name: str = "", tail: int = 200, since: str = "10m") -> dict[str, Any]:
    try:
        return fetch_recent_logs(system_id=system_id, service_name=service_name, tail=tail, since=since)
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        stdout = (exc.stdout or "").strip()
        raise HTTPException(status_code=502, detail=stderr or stdout or str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/recovery/history")
def recovery_history() -> dict[str, Any]:
    return {"items": list(reversed(RECOVERY_HISTORY[-20:]))}


@app.post("/api/recovery/execute")
def execute_recovery(payload: RecoveryActionRequest) -> dict[str, Any]:
    if payload.action not in {"restart_pod", "scale_service", "alert_only"}:
        raise HTTPException(status_code=400, detail=f"Unsupported recovery action: {payload.action}")

    service_name = payload.service_name or "unknown"
    try:
        action_result = _execute_recovery_action(payload.action, service_name)
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        stdout = (exc.stdout or "").strip()
        raise HTTPException(
            status_code=502,
            detail=(
                f"Recovery command failed for {service_name}: "
                f"{stderr or stdout or str(exc)}"
            ),
        ) from exc

    event = {
        "timestamp": datetime.now(timezone.utc).astimezone().isoformat(),
        "action": payload.action,
        "action_label": action_result["label"],
        "status": action_result["status"],
        "mode": RECOVERY_MODE,
        "namespace": RECOVERY_NAMESPACE,
        "service_name": service_name,
        "severity": payload.severity or "unknown",
        "source": payload.source,
        "notes": action_result["notes"],
        "context": payload.context,
    }
    RECOVERY_HISTORY.append(event)
    write_audit_log(
        action=f"recovery.{payload.action}",
        target_type="service",
        target_id=service_name,
        actor=payload.source,
        payload=event,
    )
    return event


@app.post("/api/live/analyze")
def live_analyze(payload: LiveAnalyzeRequest) -> JSONResponse:
    try:
        live_inputs = collect_live_inputs(
            system_id=payload.system_id,
            source_service=payload.source_service,
            jaeger_url=payload.jaeger_url,
            prometheus_url=payload.prometheus_url,
            lookback_minutes=payload.lookback_minutes,
            query_limit=payload.query_limit,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Live collection failed: {exc}") from exc

    pipeline_payload = {
        "window": live_inputs["window"],
        "graph": live_inputs["graph"],
        "run_rca_on_any_input": payload.run_rca_on_any_input,
    }
    try:
        result = post_json(f"{ORCHESTRATOR_BASE_URL}/analyze/pipeline", pipeline_payload)
    except Exception as exc:
        try:
            result = _run_pipeline_locally(pipeline_payload)
            result["metadata"] = {
                **result.get("metadata", {}),
                "orchestrator_error": str(exc),
            }
        except Exception as fallback_exc:
            raise HTTPException(status_code=502, detail=f"{exc} | Fallback failed: {fallback_exc}") from fallback_exc

    recommendation = recommend_actions(result.get("anomaly", {}), result.get("rca"))
    result["recommendation"] = recommendation
    result["live_context"] = {
        "trace_snapshot": live_inputs["trace_snapshot"],
        "metrics_snapshot": live_inputs["metrics_snapshot"],
        "window_features": live_inputs["window"]["features"],
        "system_id": payload.system_id,
        "source_service": payload.source_service,
        "lookback_minutes": payload.lookback_minutes,
        "jaeger_url": payload.jaeger_url,
        "prometheus_url": payload.prometheus_url,
    }
    event = create_monitoring_event(result)
    result["monitoring_event"] = event
    result["event_id"] = event["id"]
    return JSONResponse(result)


@app.post("/api/demo/analyze")
def demo_analyze(payload: DemoAnalyzeRequest) -> JSONResponse:
    if payload.preset not in WINDOW_PRESETS:
        raise HTTPException(status_code=400, detail=f"Unknown preset: {payload.preset}")

    try:
        pipeline_payload = build_demo_pipeline_payload(
            sample_name=payload.sample_name,
            preset=payload.preset,
            run_rca_on_any_input=payload.run_rca_on_any_input,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        result = post_json(f"{ORCHESTRATOR_BASE_URL}/analyze/pipeline", pipeline_payload)
    except Exception as exc:
        try:
            result = _run_pipeline_locally(pipeline_payload)
            result["metadata"] = {
                **result.get("metadata", {}),
                "orchestrator_error": str(exc),
            }
        except Exception as fallback_exc:
            raise HTTPException(status_code=502, detail=f"{exc} | Fallback failed: {fallback_exc}") from fallback_exc

    recommendation = recommend_actions(result.get("anomaly", {}), result.get("rca"))
    result["recommendation"] = recommendation
    result["demo_context"] = {
        "preset": payload.preset,
        "sample_name": payload.sample_name,
        "run_rca_on_any_input": payload.run_rca_on_any_input,
    }
    return JSONResponse(result)
