from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class GraphPredictRequest(BaseModel):
    graph_id: str | None = None
    model_key: str | None = None
    node_names: list[str] = Field(default_factory=list)
    node_roles: list[str] = Field(default_factory=list)
    x: list[list[float]] = Field(default_factory=list)
    edge_index: list[list[int]] = Field(default_factory=list)
    top_k: int = 3
    metadata: dict[str, Any] = Field(default_factory=dict)


class RankedNode(BaseModel):
    node_index: int
    service_name: str
    score: float


class GraphPredictResponse(BaseModel):
    graph_id: str | None = None
    model_key: str
    top1: RankedNode
    topk: list[RankedNode] = Field(default_factory=list)
    model_name: str
    model_type: str
    metadata: dict[str, Any] = Field(default_factory=dict)
