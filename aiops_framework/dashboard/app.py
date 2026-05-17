from __future__ import annotations

import json
import os
import sqlite3
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from aiops_framework.inference.orchestrator.clients import get_json, post_json
from aiops_framework.inference.anomaly_service.schemas import WindowPredictResponse
from aiops_framework.inference.common.artifact_registry import (
    get_model_summary,
    promote_model,
)
from aiops_framework.mlops import get_mlops_status
from aiops_framework.inference.rca_service.schemas import GraphPredictResponse, RankedNode
from aiops_framework.core.config import load_system_config
from aiops_framework.registry.system_catalog import get_system, list_systems
from aiops_framework.registry.service_catalog import load_service_catalog

from .auth import (
    AUTH_ENABLED,
    AuthContext,
    ROLE_PERMISSIONS,
    SESSION_TTL_HOURS,
    bootstrap_admin_if_configured,
    clear_session_cookie,
    get_auth_context,
    hash_password,
    make_session_expiry,
    require_permission,
    require_valid_role,
    set_session_cookie,
    verify_password,
)
from .demo_data import DEFAULT_GRAPH_ROOT, WINDOW_PRESETS, build_demo_pipeline_payload, list_graph_samples
from .live_data import DEFAULT_JAEGER_URL, DEFAULT_PROMETHEUS_URL, DEFAULT_SOURCE_SERVICE, DEFAULT_SYSTEM_ID, collect_live_inputs
from .logs import fetch_recent_logs
from .policy import recommend_actions
from .store import (
    count_users,
    create_monitoring_event,
    create_session,
    create_user,
    get_monitoring_event,
    get_user,
    init_store,
    list_audit_logs,
    list_monitoring_events,
    list_users,
    revoke_session,
    save_feedback,
    set_user_active,
    update_user_password,
    update_user_role,
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
ENABLE_K8S_HEALTH_OVERRIDE = os.environ.get("AIOPS_ENABLE_K8S_HEALTH_OVERRIDE", "false").strip().lower() == "true"

app = FastAPI(title="AIOps RCA Dashboard", version="0.1.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class DemoAnalyzeRequest(BaseModel):
    sample_name: str
    preset: str = "healthy"
    run_rca_on_any_input: bool = False
    model_key: str = os.environ.get("AIOPS_DASHBOARD_DEFAULT_RCA_MODEL_KEY", "rf_ml_ranker")


class RecoveryActionRequest(BaseModel):
    system_id: str = DEFAULT_SYSTEM_ID
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


class BootstrapAdminRequest(BaseModel):
    username: str
    password: str
    display_name: str = "Administrator"


class LoginRequest(BaseModel):
    username: str
    password: str


class CreateUserRequest(BaseModel):
    username: str
    password: str
    role: str
    display_name: str = ""
    is_active: bool = True


class UpdateUserRoleRequest(BaseModel):
    role: str


class SetUserActiveRequest(BaseModel):
    is_active: bool


class UpdateUserPasswordRequest(BaseModel):
    password: str


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str


class PromoteModelRequest(BaseModel):
    system_id: str
    model_type: str
    model_name: str
    notes: str = ""


RECOVERY_HISTORY: list[dict[str, Any]] = []


class LiveAnalyzeRequest(BaseModel):
    system_id: str = DEFAULT_SYSTEM_ID
    source_service: str = DEFAULT_SOURCE_SERVICE
    jaeger_url: str = DEFAULT_JAEGER_URL
    prometheus_url: str = DEFAULT_PROMETHEUS_URL
    lookback_minutes: int = 1
    query_limit: int = 150
    run_rca_on_any_input: bool = False
    model_key: str = os.environ.get("AIOPS_DASHBOARD_DEFAULT_RCA_MODEL_KEY", "rf_ml_ranker")


def _read_index_html() -> str:
    html = (TEMPLATES_DIR / "index_v2.html").read_text(encoding="utf-8")
    static_version = os.environ.get("AIOPS_DASHBOARD_STATIC_VERSION", "20260501-multimodel")
    config = {
        "anomalyBaseUrl": ANOMALY_BASE_URL,
        "rcaBaseUrl": RCA_BASE_URL,
        "orchestratorBaseUrl": ORCHESTRATOR_BASE_URL,
        "jaegerUrl": DEFAULT_JAEGER_URL,
        "prometheusUrl": DEFAULT_PROMETHEUS_URL,
        "defaultModelKey": os.environ.get("AIOPS_DASHBOARD_DEFAULT_RCA_MODEL_KEY", "rf_ml_ranker"),
    }
    return (
        html.replace("__DASHBOARD_CONFIG__", json.dumps(config, ensure_ascii=False))
        .replace("__STATIC_VERSION__", static_version)
    )


def _public_user_payload(user: dict[str, Any] | None) -> dict[str, Any] | None:
    if user is None:
        return None
    return {
        "username": user.get("username"),
        "role": user.get("role"),
        "display_name": user.get("display_name", ""),
        "is_active": bool(user.get("is_active", False)),
        "created_at": user.get("created_at"),
        "updated_at": user.get("updated_at"),
        "last_login_at": user.get("last_login_at"),
    }


def _auth_payload(user: dict[str, Any] | None, *, username: str, role: str, permissions: set[str] | None = None) -> dict[str, Any]:
    effective_permissions = permissions if permissions is not None else ROLE_PERMISSIONS.get(role, set())
    return {
        "authenticated": True,
        "auth_enabled": AUTH_ENABLED,
        "user": _public_user_payload(user) or {"username": username, "role": role},
        "permissions": sorted(effective_permissions),
    }


@app.on_event("startup")
def startup() -> None:
    init_store()
    bootstrap_admin_if_configured()


def _best_effort_get(url: str) -> dict[str, Any]:
    try:
        return get_json(url)
    except Exception as exc:
        return {"status": "down", "error": str(exc)}


def _system_namespace(system_id: str) -> str:
    try:
        system = get_system(system_id)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=f"Unknown system_id: {system_id}") from exc

    namespace = str(system.get("namespace") or "").strip()
    if not namespace:
        raise HTTPException(status_code=400, detail=f"System {system_id} has no namespace configured")
    return namespace


def _system_model_root(system_id: str, model_type: str) -> Path:
    cfg = load_system_config(system_id)
    profile = cfg.get("model_profile", {}).get(model_type, {})
    registry_root = profile.get("registry_root")
    if not registry_root:
        raise HTTPException(status_code=404, detail=f"System {system_id} has no {model_type} registry_root configured")

    registry_path = Path(str(registry_root))
    if not registry_path.is_absolute():
        registry_path = (Path(cfg["system_root"]) / registry_path).resolve()
    return registry_path


def _model_registry_payload(system_id: str, model_type: str) -> dict[str, Any]:
    models_root = _system_model_root(system_id, model_type)
    summary = get_model_summary(models_root, system_id=system_id, model_type=model_type)
    summary["models_root"] = str(models_root)
    return summary


def _kubectl(namespace: str, args: list[str]) -> subprocess.CompletedProcess[str]:
    cmd = ["kubectl", "-n", namespace, *args]
    return subprocess.run(cmd, check=True, text=True, capture_output=True)


def _deployment_payload(namespace: str, name: str) -> dict[str, Any]:
    result = _kubectl(namespace, ["get", "deployment", name, "-o", "json"])
    return json.loads(result.stdout)


def _endpoint_payload(namespace: str, name: str) -> dict[str, Any]:
    result = _kubectl(namespace, ["get", "endpoints", name, "-o", "json"])
    return json.loads(result.stdout)


def _workload_payload(namespace: str, kind: str, name: str) -> dict[str, Any] | None:
    try:
        result = _kubectl(namespace, ["get", kind, name, "-o", "json"])
    except subprocess.CalledProcessError:
        return None
    return json.loads(result.stdout)


def _collect_kubernetes_service_health(system_id: str) -> list[dict[str, Any]]:
    namespace = _system_namespace(system_id)
    catalog_df = load_service_catalog(system_id)
    service_names = [str(name).strip() for name in catalog_df["service_name"].astype(str).tolist() if str(name).strip()]
    snapshot: list[dict[str, Any]] = []

    for service_name in service_names:
        entry: dict[str, Any] = {
            "service_name": service_name,
            "namespace": namespace,
            "workload_kind": None,
            "desired_replicas": None,
            "available_replicas": None,
            "endpoint_count": 0,
            "is_degraded": False,
            "reasons": [],
        }

        workload = _workload_payload(namespace, "deployment", service_name)
        if workload is None:
            workload = _workload_payload(namespace, "statefulset", service_name)
            if workload is not None:
                entry["workload_kind"] = "statefulset"
        else:
            entry["workload_kind"] = "deployment"

        if workload is not None:
            entry["desired_replicas"] = int(workload.get("spec", {}).get("replicas", 0) or 0)
            entry["available_replicas"] = int(workload.get("status", {}).get("availableReplicas", 0) or 0)

        endpoints = _workload_payload(namespace, "endpoints", service_name)
        if endpoints is not None:
            ready_addresses = 0
            for subset in endpoints.get("subsets", []) or []:
                ready_addresses += len(subset.get("addresses", []) or [])
            entry["endpoint_count"] = ready_addresses

        reasons: list[str] = []
        desired = entry["desired_replicas"]
        available = entry["available_replicas"]
        endpoint_count = int(entry["endpoint_count"])

        if desired == 0 and entry["workload_kind"] is not None:
            reasons.append("scaled_to_zero")
        elif available == 0 and desired not in (None, 0):
            reasons.append("no_available_replicas")

        if endpoint_count == 0:
            reasons.append("no_ready_endpoints")

        entry["reasons"] = reasons
        entry["is_degraded"] = bool(reasons)
        snapshot.append(entry)

    return snapshot


def _reason_score(reasons: list[str]) -> float:
    score = 0.0
    if "scaled_to_zero" in reasons:
        score += 1.0
    if "no_available_replicas" in reasons:
        score += 0.9
    if "no_ready_endpoints" in reasons:
        score += 0.8
    return score


def _build_kubernetes_override_rca(
    graph_payload: dict[str, Any] | None,
    degraded_services: list[dict[str, Any]],
) -> dict[str, Any]:
    graph_id = None
    node_names: list[str] = []
    if graph_payload:
        graph_id = graph_payload.get("graph_id")
        node_names = [str(item) for item in graph_payload.get("node_names", [])]

    ranked = []
    ordered_services = sorted(
        degraded_services,
        key=lambda svc: (_reason_score(list(svc.get("reasons", []))), str(svc.get("service_name", ""))),
        reverse=True,
    )
    for rank_index, item in enumerate(ordered_services):
        service_name = str(item["service_name"])
        score = max(0.5, _reason_score(list(item.get("reasons", []))) - (rank_index * 0.05))
        node_index = node_names.index(service_name) if service_name in node_names else rank_index
        ranked.append(
            {
                "node_index": int(node_index),
                "service_name": service_name,
                "score": float(score),
            }
        )

    return {
        "graph_id": graph_id,
        "model_key": "kubernetes_health_override",
        "top1": ranked[0],
        "topk": ranked[:3],
        "model_name": "kubernetes_health_override",
        "model_type": "rule_override",
        "metadata": {
            "source": "kubernetes_health",
            "degraded_services": degraded_services,
        },
    }


def _apply_kubernetes_health_override(
    result: dict[str, Any],
    *,
    system_id: str,
    graph_payload: dict[str, Any] | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not ENABLE_K8S_HEALTH_OVERRIDE:
        return result, []

    health_snapshot = _collect_kubernetes_service_health(system_id)
    degraded_services = [item for item in health_snapshot if item.get("is_degraded")]
    if not degraded_services:
        return result, health_snapshot

    anomaly = dict(result.get("anomaly") or {})
    anomaly["is_anomaly"] = True
    anomaly["anomaly_score"] = max(float(anomaly.get("anomaly_score", 0.0) or 0.0), 0.99)
    anomaly_metadata = dict(anomaly.get("metadata") or {})
    anomaly_metadata["kubernetes_health_override"] = True
    anomaly_metadata["degraded_services"] = degraded_services
    anomaly["metadata"] = anomaly_metadata
    result["anomaly"] = anomaly
    result["rca"] = _build_kubernetes_override_rca(graph_payload, degraded_services)
    result["pipeline_state"] = "kubernetes_health_override"
    result["metadata"] = {
        **dict(result.get("metadata") or {}),
        "kubernetes_health_override": True,
        "degraded_services": degraded_services,
    }
    return result, health_snapshot


def _execute_recovery_action(action: str, service_name: str, namespace: str) -> dict[str, str]:
    if RECOVERY_MODE != "real":
        action_map = {
            "restart_pod": {
                "label": "Restart Pod",
                "status": "accepted",
                "notes": f"Demo mode: would call Kubernetes API to restart the predicted service pod in namespace {namespace}.",
            },
            "scale_service": {
                "label": "Scale Service",
                "status": "accepted",
                "notes": f"Demo mode: would increase replicas or trigger HPA guidance for the predicted service in namespace {namespace}.",
            },
            "alert_only": {
                "label": "Alert Only",
                "status": "accepted",
                "notes": f"Demo mode: would notify the operator for namespace {namespace} and wait for manual approval.",
            },
        }
        if action not in action_map:
            raise HTTPException(status_code=400, detail=f"Unsupported recovery action: {action}")
        return action_map[action]

    if action == "alert_only":
        return {
            "label": "Alert Only",
            "status": "accepted",
            "notes": f"Real mode: alert recorded for {service_name} in namespace {namespace}; no Kubernetes mutation was executed.",
        }

    if action == "restart_pod":
        _kubectl(namespace, ["rollout", "restart", f"deployment/{service_name}"])
        _kubectl(namespace, ["rollout", "status", f"deployment/{service_name}", f"--timeout={RECOVERY_TIMEOUT_SECONDS}s"])
        return {
            "label": "Restart Pod",
            "status": "executed",
            "notes": (
                f"Real mode: restarted deployment/{service_name} in namespace {namespace} "
                f"and rollout completed within {RECOVERY_TIMEOUT_SECONDS}s."
            ),
        }

    if action == "scale_service":
        deployment = _deployment_payload(namespace, service_name)
        current_replicas = int(deployment.get("spec", {}).get("replicas", 1) or 1)
        target_replicas = current_replicas + 1
        _kubectl(namespace, ["scale", f"deployment/{service_name}", f"--replicas={target_replicas}"])
        _kubectl(namespace, ["rollout", "status", f"deployment/{service_name}", f"--timeout={RECOVERY_TIMEOUT_SECONDS}s"])
        return {
            "label": "Scale Service",
            "status": "executed",
            "notes": (
                f"Real mode: scaled deployment/{service_name} in namespace {namespace} "
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
        model_key=str(rca_raw.get("model_key") or pipeline_payload["graph"].get("model_key") or ""),
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


def _run_live_rca_only(graph_payload: dict[str, Any] | None, *, model_key: str | None = None) -> dict[str, Any]:
    if graph_payload is None:
        raise ValueError("Live RCA-only fallback requires a graph payload.")

    rca_request = dict(graph_payload)
    if model_key and not rca_request.get("model_key"):
        rca_request["model_key"] = model_key

    rca_raw = post_json(f"{RCA_BASE_URL}/predict/graph", rca_request)
    rca_result = GraphPredictResponse(
        top1=RankedNode(**rca_raw["top1"]),
        topk=[RankedNode(**item) for item in rca_raw.get("topk", [])],
        graph_id=rca_raw.get("graph_id"),
        model_key=str(rca_raw.get("model_key") or rca_request.get("model_key") or ""),
        model_name=rca_raw["model_name"],
        model_type=rca_raw["model_type"],
        metadata=rca_raw.get("metadata", {}),
    ).model_dump()

    anomaly_result = WindowPredictResponse(
        anomaly_score=1.0,
        threshold=0.0,
        is_anomaly=True,
        model_name="live_rca_only_fallback",
        model_kind="fallback",
        optimize_for="rca_only",
        metadata={"reason": "Anomaly/orchestrator unavailable; RCA executed directly from live graph."},
    ).model_dump()

    return {
        "anomaly": anomaly_result,
        "rca": rca_result,
        "pipeline_state": "live_rca_only",
        "metadata": {
            "reason": "Executed live RCA directly because anomaly/orchestrator services were unavailable.",
            "execution_mode": "dashboard_live_rca_only_fallback",
        },
    }


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return _read_index_html()


@app.get("/api/auth/config")
def auth_config() -> dict[str, Any]:
    return {
        "enabled": AUTH_ENABLED,
        "bootstrap_required": count_users() == 0,
        "session_ttl_hours": SESSION_TTL_HOURS,
    }


@app.post("/api/auth/bootstrap")
def bootstrap_admin(payload: BootstrapAdminRequest) -> dict[str, Any]:
    if not AUTH_ENABLED:
        raise HTTPException(status_code=400, detail="Authentication is disabled")
    if count_users() > 0:
        raise HTTPException(status_code=409, detail="Bootstrap is only allowed when no users exist")

    try:
        password_hash = hash_password(payload.password)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    user = create_user(
        username=payload.username,
        password_hash=password_hash,
        role="admin",
        display_name=payload.display_name,
        is_active=True,
    )
    write_audit_log(
        action="auth.bootstrap_admin",
        target_type="user",
        target_id=user["username"],
        actor="bootstrap",
        payload={"display_name": user["display_name"]},
    )
    return {"status": "created", "user": user}


@app.post("/api/auth/login")
def login(payload: LoginRequest, request: Request) -> JSONResponse:
    if not AUTH_ENABLED:
        raise HTTPException(status_code=400, detail="Authentication is disabled")

    username = payload.username.strip()
    user = get_user(username, include_password_hash=True)
    valid = bool(user) and bool(user.get("is_active")) and verify_password(payload.password, str(user.get("password_hash", "")))
    if not valid:
        write_audit_log(
            action="auth.login_failed",
            target_type="user",
            target_id=username or "unknown",
            actor=username or "anonymous",
            payload={"path": str(request.url.path)},
        )
        raise HTTPException(status_code=401, detail="Invalid credentials")

    session_id = create_session(
        username=username,
        expires_at=make_session_expiry(),
        user_agent=request.headers.get("user-agent", ""),
        client_host=request.client.host if request.client else "",
    )
    response = JSONResponse(
        {
            "status": "ok",
            **_auth_payload(
                user,
                username=username,
                role=str(user.get("role", "viewer")),
            ),
        }
    )
    set_session_cookie(response, session_id)
    write_audit_log(
        action="auth.login_success",
        target_type="user",
        target_id=username,
        actor=username,
        payload={"session_id": session_id},
    )
    return response


@app.post("/api/auth/logout")
def logout(request: Request, auth: AuthContext = Depends(get_auth_context)) -> JSONResponse:
    response = JSONResponse({"status": "ok"})
    if auth.session_id:
        revoke_session(auth.session_id)
        write_audit_log(
            action="auth.logout",
            target_type="session",
            target_id=auth.session_id,
            actor=auth.username,
            payload={},
        )
    clear_session_cookie(response)
    return response


@app.get("/api/auth/me")
def auth_me(auth: AuthContext = Depends(get_auth_context)) -> dict[str, Any]:
    user = get_user(auth.username, include_password_hash=False)
    return _auth_payload(user, username=auth.username, role=auth.role, permissions=auth.permissions)


@app.post("/api/auth/change-password")
def change_password(payload: ChangePasswordRequest, auth: AuthContext = Depends(get_auth_context)) -> dict[str, Any]:
    user = get_user(auth.username, include_password_hash=True)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    if not verify_password(payload.current_password, str(user.get("password_hash", ""))):
        raise HTTPException(status_code=401, detail="Current password is incorrect")
    try:
        password_hash = hash_password(payload.new_password)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    update_user_password(auth.username, password_hash)
    write_audit_log(
        action="auth.change_password",
        target_type="user",
        target_id=auth.username,
        actor=auth.username,
        payload={},
    )
    return {"status": "ok"}


@app.get("/api/users")
def users(auth: AuthContext = Depends(require_permission("user_manage"))) -> dict[str, Any]:
    return {"items": list_users(), "actor": auth.username}


@app.post("/api/users")
def create_dashboard_user(
    payload: CreateUserRequest,
    auth: AuthContext = Depends(require_permission("user_manage")),
) -> dict[str, Any]:
    role = require_valid_role(payload.role)
    try:
        password_hash = hash_password(payload.password)
        user = create_user(
            username=payload.username,
            password_hash=password_hash,
            role=role,
            display_name=payload.display_name,
            is_active=payload.is_active,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except sqlite3.IntegrityError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    write_audit_log(
        action="user.create",
        target_type="user",
        target_id=user["username"],
        actor=auth.username,
        payload={"role": user["role"], "is_active": user["is_active"]},
    )
    return {"status": "created", "user": user}


@app.patch("/api/users/{username}/role")
def patch_user_role(
    username: str,
    payload: UpdateUserRoleRequest,
    auth: AuthContext = Depends(require_permission("user_manage")),
) -> dict[str, Any]:
    role = require_valid_role(payload.role)
    try:
        user = update_user_role(username, role)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    write_audit_log(
        action="user.update_role",
        target_type="user",
        target_id=username,
        actor=auth.username,
        payload={"role": role},
    )
    return {"status": "ok", "user": user}


@app.patch("/api/users/{username}/active")
def patch_user_active(
    username: str,
    payload: SetUserActiveRequest,
    auth: AuthContext = Depends(require_permission("user_manage")),
) -> dict[str, Any]:
    try:
        user = set_user_active(username, payload.is_active)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    write_audit_log(
        action="user.set_active",
        target_type="user",
        target_id=username,
        actor=auth.username,
        payload={"is_active": payload.is_active},
    )
    return {"status": "ok", "user": user}


@app.patch("/api/users/{username}/password")
def patch_user_password(
    username: str,
    payload: UpdateUserPasswordRequest,
    auth: AuthContext = Depends(require_permission("user_manage")),
) -> dict[str, Any]:
    try:
        password_hash = hash_password(payload.password)
        update_user_password(username, password_hash)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    write_audit_log(
        action="user.update_password",
        target_type="user",
        target_id=username,
        actor=auth.username,
        payload={},
    )
    return {"status": "ok"}


@app.get("/api/health")
def dashboard_health() -> dict[str, Any]:
    return {
        "dashboard": {
            "status": "ok",
            "graph_root": str(DEFAULT_GRAPH_ROOT),
            "recovery_mode": RECOVERY_MODE,
            "recovery_namespace": RECOVERY_NAMESPACE,
            "auth_enabled": AUTH_ENABLED,
        },
        "anomaly": _best_effort_get(f"{ANOMALY_BASE_URL}/health"),
        "rca": _best_effort_get(f"{RCA_BASE_URL}/health"),
        "orchestrator": _best_effort_get(f"{ORCHESTRATOR_BASE_URL}/health"),
    }


@app.get("/api/metadata")
def dashboard_metadata(auth: AuthContext = Depends(require_permission("read"))) -> dict[str, Any]:
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
        "auth": {
            "enabled": AUTH_ENABLED,
            "bootstrap_required": count_users() == 0,
        },
        "anomaly": _best_effort_get(f"{ANOMALY_BASE_URL}/metadata"),
        "rca": _best_effort_get(f"{RCA_BASE_URL}/metadata"),
        "orchestrator": _best_effort_get(f"{ORCHESTRATOR_BASE_URL}/metadata"),
    }


@app.get("/api/systems")
def systems(auth: AuthContext = Depends(require_permission("read"))) -> dict[str, Any]:
    return {"items": list_systems(), "default_system_id": DEFAULT_SYSTEM_ID}


@app.get("/api/systems/{system_id}")
def system_detail(system_id: str, auth: AuthContext = Depends(require_permission("read"))) -> dict[str, Any]:
    try:
        return get_system(system_id)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/samples")
def samples(auth: AuthContext = Depends(require_permission("read"))) -> dict[str, Any]:
    return {"items": list_graph_samples(), "graph_root": str(DEFAULT_GRAPH_ROOT)}


@app.get("/api/monitoring-events")
def monitoring_events(
    limit: int = 50,
    system_id: str | None = None,
    auth: AuthContext = Depends(require_permission("read")),
) -> dict[str, Any]:
    return {"items": list_monitoring_events(limit=limit, system_id=system_id)}


@app.get("/api/monitoring-events/{event_id}")
def monitoring_event_detail(
    event_id: int,
    auth: AuthContext = Depends(require_permission("read")),
) -> dict[str, Any]:
    event = get_monitoring_event(event_id)
    if event is None:
        raise HTTPException(status_code=404, detail=f"Monitoring event not found: {event_id}")
    return event


@app.post("/api/monitoring-events/{event_id}/feedback")
def record_feedback(
    event_id: int,
    payload: FeedbackRequest,
    auth: AuthContext = Depends(require_permission("feedback_write")),
) -> dict[str, Any]:
    allowed = {"accepted_incident", "rejected_false_positive", "unknown"}
    if payload.feedback not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported feedback: {payload.feedback}")
    try:
        return save_feedback(
            event_id=event_id,
            feedback=payload.feedback,
            actor=auth.username,
            notes=payload.notes,
            payload=payload.context,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/audit-logs")
def audit_logs(
    limit: int = 50,
    actor: str | None = None,
    action: str | None = None,
    system_id: str | None = None,
    auth: AuthContext = Depends(require_permission("audit_view")),
) -> dict[str, Any]:
    return {
        "items": list_audit_logs(limit=limit, actor=actor, action=action, system_id=system_id),
        "filters": {"actor": actor, "action": action, "system_id": system_id},
    }


@app.get("/api/models")
def models(system_id: str, auth: AuthContext = Depends(require_permission("model_select"))) -> dict[str, Any]:
    try:
        system = get_system(system_id)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return {
        "system_id": system_id,
        "display_name": system.get("display_name", system_id),
        "mlops": get_mlops_status(),
        "anomaly": _model_registry_payload(system_id, "anomaly"),
        "rca": _model_registry_payload(system_id, "rca"),
    }


@app.get("/api/mlops/status")
def mlops_status(auth: AuthContext = Depends(require_permission("model_select"))) -> dict[str, Any]:
    return get_mlops_status()


@app.post("/api/models/promote")
def model_promote(
    payload: PromoteModelRequest,
    auth: AuthContext = Depends(require_permission("model_promote")),
) -> dict[str, Any]:
    model_type = payload.model_type.strip().lower()
    if model_type not in {"anomaly", "rca"}:
        raise HTTPException(status_code=400, detail=f"Unsupported model_type: {payload.model_type}")

    models_root = _system_model_root(payload.system_id, model_type)
    try:
        path = promote_model(
            models_root,
            system_id=payload.system_id,
            model_type=model_type,
            model_name=payload.model_name,
            promoted_by=auth.username,
            notes=payload.notes.strip(),
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    write_audit_log(
        action="model.promote",
        target_type=f"{model_type}_model",
        target_id=payload.model_name,
        actor=auth.username,
        payload={
            "system_id": payload.system_id,
            "model_type": model_type,
            "model_name": payload.model_name,
            "notes": payload.notes.strip(),
            "registry_path": str(path),
        },
    )
    return {
        "status": "ok",
        "system_id": payload.system_id,
        "model_type": model_type,
        "model_name": payload.model_name,
        "registry_path": str(path),
    }


@app.get("/api/logs/recent")
def recent_logs(
    system_id: str = DEFAULT_SYSTEM_ID,
    service_name: str = "",
    tail: int = 200,
    since: str = "10m",
    auth: AuthContext = Depends(require_permission("read")),
) -> dict[str, Any]:
    try:
        return fetch_recent_logs(system_id=system_id, service_name=service_name, tail=tail, since=since)
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        stdout = (exc.stdout or "").strip()
        raise HTTPException(status_code=502, detail=stderr or stdout or str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/recovery/history")
def recovery_history(auth: AuthContext = Depends(require_permission("read"))) -> dict[str, Any]:
    return {"items": list(reversed(RECOVERY_HISTORY[-20:]))}


@app.post("/api/recovery/execute")
def execute_recovery(
    payload: RecoveryActionRequest,
    auth: AuthContext = Depends(require_permission("recovery_execute")),
) -> dict[str, Any]:
    if payload.action not in {"restart_pod", "scale_service", "alert_only"}:
        raise HTTPException(status_code=400, detail=f"Unsupported recovery action: {payload.action}")

    namespace = _system_namespace(payload.system_id)
    service_name = payload.service_name or "unknown"
    try:
        action_result = _execute_recovery_action(payload.action, service_name, namespace)
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        stdout = (exc.stdout or "").strip()
        raise HTTPException(
            status_code=502,
            detail=(
                f"Recovery command failed for {service_name} in system {payload.system_id} namespace {namespace}: "
                f"{stderr or stdout or str(exc)}"
            ),
        ) from exc

    event = {
        "timestamp": datetime.now(timezone.utc).astimezone().isoformat(),
        "action": payload.action,
        "action_label": action_result["label"],
        "status": action_result["status"],
        "mode": RECOVERY_MODE,
        "system_id": payload.system_id,
        "namespace": namespace,
        "service_name": service_name,
        "severity": payload.severity or "unknown",
        "source": payload.source,
        "actor": auth.username,
        "notes": action_result["notes"],
        "context": payload.context,
    }
    RECOVERY_HISTORY.append(event)
    write_audit_log(
        action=f"recovery.{payload.action}",
        target_type="service",
        target_id=service_name,
        actor=auth.username,
        payload=event,
    )
    return event


@app.post("/api/live/analyze")
def live_analyze(
    payload: LiveAnalyzeRequest,
    auth: AuthContext = Depends(require_permission("live_analyze")),
) -> JSONResponse:
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

    if live_inputs.get("graph") is not None:
        live_inputs["graph"]["model_key"] = payload.model_key

    pipeline_payload = {
        "window": live_inputs["window"],
        "graph": live_inputs["graph"],
        "run_rca_on_any_input": payload.run_rca_on_any_input,
        "model_key": payload.model_key,
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
            try:
                result = _run_live_rca_only(live_inputs.get("graph"), model_key=payload.model_key)
                result["metadata"] = {
                    **result.get("metadata", {}),
                    "orchestrator_error": str(exc),
                    "fallback_error": str(fallback_exc),
                }
            except Exception as rca_only_exc:
                raise HTTPException(
                    status_code=502,
                    detail=f"{exc} | Fallback failed: {fallback_exc} | RCA-only fallback failed: {rca_only_exc}",
                ) from rca_only_exc

    result, kubernetes_health = _apply_kubernetes_health_override(
        result,
        system_id=payload.system_id,
        graph_payload=live_inputs.get("graph"),
    )
    recommendation = recommend_actions(result.get("anomaly", {}), result.get("rca"))
    result["recommendation"] = recommendation
    result["live_context"] = {
        "trace_snapshot": live_inputs["trace_snapshot"],
        "metrics_snapshot": live_inputs["metrics_snapshot"],
        "window_features": live_inputs["window"]["features"],
        "kubernetes_health": kubernetes_health,
        "system_id": payload.system_id,
        "source_service": payload.source_service,
        "lookback_minutes": payload.lookback_minutes,
        "jaeger_url": payload.jaeger_url,
        "prometheus_url": payload.prometheus_url,
        "model_key": payload.model_key,
    }
    event = create_monitoring_event(result)
    result["monitoring_event"] = event
    result["event_id"] = event["id"]
    write_audit_log(
        action="monitoring.live_analyze",
        target_type="monitoring_event",
        target_id=str(event["id"]),
        actor=auth.username,
        payload={"system_id": payload.system_id, "source_service": payload.source_service},
    )
    return JSONResponse(result)


@app.post("/api/demo/analyze")
def demo_analyze(
    payload: DemoAnalyzeRequest,
    auth: AuthContext = Depends(require_permission("demo_analyze")),
) -> JSONResponse:
    if payload.preset not in WINDOW_PRESETS:
        raise HTTPException(status_code=400, detail=f"Unknown preset: {payload.preset}")

    try:
        pipeline_payload = build_demo_pipeline_payload(
            sample_name=payload.sample_name,
            preset=payload.preset,
            run_rca_on_any_input=payload.run_rca_on_any_input,
        )
        if pipeline_payload.get("graph") is not None:
            pipeline_payload["graph"]["model_key"] = payload.model_key
        pipeline_payload["model_key"] = payload.model_key
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
        "model_key": payload.model_key,
    }
    return JSONResponse(result)
