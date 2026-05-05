from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from aiops_framework.inference.anomaly_service.schemas import WindowPredictRequest, WindowPredictResponse
from aiops_framework.inference.rca_service.schemas import GraphPredictRequest, GraphPredictResponse


class PipelineAnalyzeRequest(BaseModel):
    window: WindowPredictRequest
    graph: GraphPredictRequest | None = None
    run_rca_on_any_input: bool = False
    model_key: str | None = None


class PolicyDecision(BaseModel):
    status: str
    severity: str
    primary_action: str
    secondary_action: str | None = None
    notes: str | None = None
    predicted_service: str | None = None


class ExecutionResponse(BaseModel):
    status: str
    action: str
    service: str
    namespace: str
    command: list[str] = Field(default_factory=list)
    dry_run: bool = False
    returncode: int | None = None
    stdout: str = ""
    stderr: str = ""
    executed_at: str


class PipelineAnalyzeResponse(BaseModel):
    anomaly: WindowPredictResponse
    rca: GraphPredictResponse | None = None
    pipeline_state: str
    policy: PolicyDecision | None = None
    execution: ExecutionResponse | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
