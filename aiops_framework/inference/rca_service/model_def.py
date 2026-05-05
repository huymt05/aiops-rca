from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_adjacency(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.bool)
    if edge_index.numel() > 0:
        adj[edge_index[0], edge_index[1]] = True
    adj.fill_diagonal_(True)
    return adj


def infer_feature_groups(feature_dim: int) -> dict[str, list[int]]:
    if feature_dim != 480:
        raise ValueError(f"Expected 480 benchmark features, received {feature_dim}.")

    return {
        "metrics": list(range(0, 56)),
        "logs": list(range(56, 65)),
        "traces": list(range(65, 80)),
        "relative": list(range(80, 480)),
    }


def build_relation_adjacencies(
    num_nodes: int,
    edge_index: torch.Tensor,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32, device=device)
    if edge_index.numel() > 0:
        src, dst = edge_index
        adj[src, dst] = 1.0
    eye = torch.eye(num_nodes, dtype=torch.float32, device=device)

    def normalize(matrix: torch.Tensor) -> torch.Tensor:
        denom = matrix.sum(dim=-1, keepdim=True).clamp(min=1.0)
        return matrix / denom

    return {
        "incoming": normalize(adj),
        "outgoing": normalize(adj.t()),
        "self": normalize(eye),
        "global": normalize(torch.ones((num_nodes, num_nodes), dtype=torch.float32, device=device)),
    }


class SimpleGraphAttention(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.score_head = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.scale = hidden_dim ** 0.5

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        h = F.relu(h)
        h = self.dropout(h)
        q = self.q_proj(h)
        k = self.k_proj(h)
        v = self.v_proj(h)

        scores = torch.matmul(q, k.transpose(0, 1)) / self.scale
        scores = scores.masked_fill(~adj, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        h2 = torch.matmul(attn, v)
        h = self.norm1(h + self.out_proj(h2))
        h = self.norm2(h + self.ffn(h))
        logits = self.score_head(h).squeeze(-1)
        return logits


class GroupEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RelationMessagePassing(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.relation_proj = nn.ModuleDict(
            {
                "incoming": nn.Linear(hidden_dim, hidden_dim),
                "outgoing": nn.Linear(hidden_dim, hidden_dim),
                "self": nn.Linear(hidden_dim, hidden_dim),
                "global": nn.Linear(hidden_dim, hidden_dim),
            }
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, relation_adj: dict[str, torch.Tensor]) -> torch.Tensor:
        relation_messages = []
        for name, adj in relation_adj.items():
            msg = adj @ h
            msg = self.relation_proj[name](msg)
            relation_messages.append(msg)

        aggregated = sum(relation_messages) / len(relation_messages)
        gate = self.gate(torch.cat([h, aggregated], dim=-1))
        mixed = gate * aggregated + (1.0 - gate) * h
        h = self.norm(h + self.dropout(mixed))
        h = self.norm(h + self.ffn(h))
        return h


class HeteroTelemetryGNN(nn.Module):
    def __init__(
        self,
        feature_groups: dict[str, list[int]],
        hidden_dim: int,
        dropout: float,
        num_layers: int,
    ) -> None:
        super().__init__()
        self.feature_groups = feature_groups
        branch_dim = hidden_dim // 2
        fused_in = branch_dim * len(feature_groups)

        self.encoders = nn.ModuleDict(
            {
                name: GroupEncoder(len(indices), branch_dim, dropout)
                for name, indices in feature_groups.items()
            }
        )
        self.fuse = nn.Sequential(
            nn.Linear(fused_in, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )
        self.layers = nn.ModuleList(
            [RelationMessagePassing(hidden_dim, dropout) for _ in range(num_layers)]
        )
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        encoded_parts = []
        for name, indices in self.feature_groups.items():
            encoded_parts.append(self.encoders[name](x[:, indices]))
        h = self.fuse(torch.cat(encoded_parts, dim=-1))

        relation_adj = build_relation_adjacencies(x.size(0), edge_index, x.device)
        for layer in self.layers:
            h = layer(h, relation_adj)

        return self.score_head(h).squeeze(-1)


def score_rca_model(
    model: nn.Module,
    *,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    model_type: str,
    device: torch.device,
) -> torch.Tensor:
    if model_type == "hetero_telemetry_gnn":
        return model(x.to(device), edge_index.to(device)).cpu()

    if model_type == "simple_graph_attention":
        adj = build_adjacency(edge_index, x.size(0)).to(device)
        return model(x.to(device), adj).cpu()

    raise ValueError(f"Unsupported torch RCA model_type: {model_type}")
