from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .model_def import score_rca_model
from .model_loader import LoadedRcaArtifacts


def _score_random_forest(artifacts: LoadedRcaArtifacts, x_rows: list[list[float]]) -> torch.Tensor:
    matrix = np.asarray(x_rows, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("RF RCA inference expects a 2D node-feature matrix.")

    model = artifacts.model
    probabilities = model.predict_proba(matrix)

    classes = list(getattr(model, "classes_", []))
    if 1 in classes:
        positive_index = classes.index(1)
    elif len(classes) == 2:
        positive_index = 1
    else:
        positive_index = 0

    scores = probabilities[:, positive_index]
    return torch.tensor(scores, dtype=torch.float32)


def _score_torch_graph_model(
    artifacts: LoadedRcaArtifacts,
    x_rows: list[list[float]],
    edge_index_rows: list[list[int]],
) -> torch.Tensor:
    x = torch.tensor(x_rows, dtype=torch.float32)
    if edge_index_rows:
        edge_index = torch.tensor(edge_index_rows, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    with torch.no_grad():
        logits = score_rca_model(
            model=artifacts.model,
            x=x,
            edge_index=edge_index,
            model_type=artifacts.model_type,
            device=artifacts.device,
        )
        return torch.softmax(logits, dim=0)


def predict_graph(
    artifacts: LoadedRcaArtifacts,
    x_rows: list[list[float]],
    edge_index_rows: list[list[int]],
    node_names: list[str],
    top_k: int = 3,
    graph_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not node_names:
        raise ValueError("node_names is required for RCA ranking output.")
    if len(node_names) != len(x_rows):
        raise ValueError("node_names length must match number of x rows.")

    if artifacts.model_type == "random_forest_service_ranker":
        probs = _score_random_forest(artifacts, x_rows)
    else:
        probs = _score_torch_graph_model(artifacts, x_rows, edge_index_rows)

    ranked = torch.argsort(probs, descending=True).tolist()
    top_k = max(1, min(int(top_k), len(node_names)))

    ranked_items = []
    for node_index in ranked[:top_k]:
        ranked_items.append(
            {
                "node_index": int(node_index),
                "service_name": str(node_names[node_index]),
                "score": float(probs[node_index].item()),
            }
        )

    return {
        "graph_id": graph_id,
        "top1": ranked_items[0],
        "topk": ranked_items,
        "model_name": str(artifacts.inference_config.get("model_name", artifacts.artifact_dir.name)),
        "model_type": artifacts.model_type,
        "metadata": metadata or {},
    }
