from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI

from .clients import get_json, post_json
from .executor import execute_action
from .schemas import ExecutionResponse, PipelineAnalyzeRequest, PipelineAnalyzeResponse, PolicyDecision
from aiops_framework.dashboard.policy import recommend_actions
from aiops_framework.inference.anomaly_service.schemas import WindowPredictRequest, WindowPredictResponse
from aiops_framework.inference.rca_service.schemas import GraphPredictResponse, RankedNode


ANOMALY_BASE_URL = os.environ.get("AIOPS_ANOMALY_BASE_URL", "http://127.0.0.1:8000")
RCA_BASE_URL = os.environ.get("AIOPS_RCA_BASE_URL", "http://127.0.0.1:8001")
AUTO_EXECUTION_ENABLED = os.environ.get("AIOPS_ENABLE_AUTO_EXECUTION", "false").lower() == "true"

POLICY_ACTION_MAP = {
    "Restart Pod": "restart_pod",
    "Scale Service": "scale_service",
    "Rollback / Config Update": "rollback_deployment",
}

app = FastAPI(title="AIOps Inference Orchestrator", version="0.2.0")


def _anomaly_predict(payload: WindowPredictRequest) -> WindowPredictResponse:
    result = post_json(f"{ANOMALY_BASE_URL}/predict/window", payload.model_dump())
    return WindowPredictResponse(**result)


def _rca_predict(graph_payload: dict[str, Any]) -> GraphPredictResponse:
    result = post_json(f"{RCA_BASE_URL}/predict/graph", graph_payload)
    result["top1"] = RankedNode(**result["top1"])
    result["topk"] = [RankedNode(**item) for item in result.get("topk", [])]
    return GraphPredictResponse(**result)


def _build_policy(anomaly_result: WindowPredictResponse, rca_result: GraphPredictResponse | None) -> PolicyDecision:
    recommendation = recommend_actions(
        anomaly_result.model_dump(),
        rca_result.model_dump() if rca_result is not None else None,
    )
    return PolicyDecision(
        status=str(recommendation.get("status", "monitor")),
        severity=str(recommendation.get("severity", "low")),
        primary_action=str(recommendation.get("primary_action", "Wait & Observe")),
        secondary_action=recommendation.get("secondary_action"),
        notes=recommendation.get("notes"),
        predicted_service=recommendation.get("predicted_service"),
    )


def _maybe_execute(policy: PolicyDecision) -> ExecutionResponse | None:
    if not AUTO_EXECUTION_ENABLED:
        return None
    if policy.status != "actionable" or not policy.predicted_service:
        return None

    executor_action = POLICY_ACTION_MAP.get(policy.primary_action)
    if executor_action is None:
        return None

    result = execute_action(service=policy.predicted_service, action=executor_action)
    return ExecutionResponse(**result.to_dict())


@app.get("/health")
def health() -> dict[str, Any]:
    own = {"status": "ok", "anomaly_base_url": ANOMALY_BASE_URL, "rca_base_url": RCA_BASE_URL}
    downstream = {}
    for name, base_url in {"anomaly": ANOMALY_BASE_URL, "rca": RCA_BASE_URL}.items():
        try:
            downstream[name] = get_json(f"{base_url}/health")
        except Exception as exc:  # pragma: no cover - best effort status aggregation
            downstream[name] = {"status": "down", "error": str(exc)}
    return {"orchestrator": own, "downstream": downstream}


@app.get("/metadata")
def metadata() -> dict[str, Any]:
    payload = {"anomaly_base_url": ANOMALY_BASE_URL, "rca_base_url": RCA_BASE_URL}
    try:
        payload["anomaly_metadata"] = get_json(f"{ANOMALY_BASE_URL}/metadata")
    except Exception as exc:
        payload["anomaly_metadata_error"] = str(exc)
    try:
        payload["rca_metadata"] = get_json(f"{RCA_BASE_URL}/metadata")
    except Exception as exc:
        payload["rca_metadata_error"] = str(exc)
    return payload


@app.post("/analyze/window", response_model=WindowPredictResponse)
def analyze_window(payload: WindowPredictRequest) -> WindowPredictResponse:
    return _anomaly_predict(payload)


@app.post("/analyze/pipeline", response_model=PipelineAnalyzeResponse)
def analyze_pipeline(payload: PipelineAnalyzeRequest) -> PipelineAnalyzeResponse:
    anomaly_result = _anomaly_predict(payload.window)
    policy = _build_policy(anomaly_result, None)

    should_run_rca = bool(payload.run_rca_on_any_input or anomaly_result.is_anomaly)
    if not should_run_rca or payload.graph is None:
        state = "anomaly_only" if payload.graph is None else "no_anomaly_skip_rca"
        return PipelineAnalyzeResponse(
            anomaly=anomaly_result,
            rca=None,
            pipeline_state=state,
            policy=policy,
            execution=None,
            metadata={
                "reason": "RCA not triggered",
                "auto_execution_enabled": AUTO_EXECUTION_ENABLED,
            },
        )

    graph_payload = payload.graph.model_dump()
    if payload.model_key and not graph_payload.get("model_key"):
        graph_payload["model_key"] = payload.model_key

    rca_result = _rca_predict(graph_payload)
    policy = _build_policy(anomaly_result, rca_result)
    execution = _maybe_execute(policy)
    return PipelineAnalyzeResponse(
        anomaly=anomaly_result,
        rca=rca_result,
        pipeline_state="anomaly_then_rca",
        policy=policy,
        execution=execution,
        metadata={
            "reason": "RCA triggered after anomaly detection",
            "auto_execution_enabled": AUTO_EXECUTION_ENABLED,
            "execution_attempted": execution is not None,
            "model_key": graph_payload.get("model_key") or payload.model_key,
        },
    )
