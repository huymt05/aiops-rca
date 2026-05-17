from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

try:
    import dagshub
except Exception:  # pragma: no cover - optional dependency
    dagshub = None

try:
    import mlflow
except Exception:  # pragma: no cover - optional dependency
    mlflow = None

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.rca_data_pipeline.run_manifest import build_run_manifest, collect_package_versions, write_run_manifest
from aiops_framework.inference.common.artifact_registry import set_stage


DEFAULT_DATA_ROOT = Path(r"D:\HOCTAP\2025-2026\HK2\DACN\microservices-demo\data_rca_balanced_v3")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a lightweight graph-attention RCA model on exported graph tensors.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--epochs", type=int, default=140)
    parser.add_argument("--hidden-dim", type=int, default=48)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    parser.add_argument("--mlflow", action="store_true", help="Enable MLflow logging for RCA training.")
    parser.add_argument("--mlflow-experiment", default="aiops-microservices-demo")
    return parser.parse_args()


def read_run_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_adjacency(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.bool)
    if edge_index.numel() > 0:
        adj[edge_index[0], edge_index[1]] = True
    adj.fill_diagonal_(True)
    return adj


@dataclass
class GraphItem:
    run_id: str
    graph_id: str
    x: torch.Tensor
    adj: torch.Tensor
    y: int
    root_cause_service: str


class GraphTensorDataset(Dataset):
    def __init__(self, items: list[GraphItem]) -> None:
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> GraphItem:
        return self.items[index]


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
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        q = self.q_proj(h)
        k = self.k_proj(h)
        v = self.v_proj(h)

        scores = torch.matmul(q, k.transpose(0, 1)) / math.sqrt(q.size(-1))
        mask = ~adj
        scores = scores.masked_fill(mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        h2 = torch.matmul(attn, v)
        h = self.norm1(h + self.out_proj(h2))
        h = self.norm2(h + self.ff(h))
        logits = self.score_head(h).squeeze(-1)
        return logits


def load_graph_items(data_root: Path) -> list[GraphItem]:
    tensor_root = data_root / "processed" / "rca" / "graph_tensors"
    items: list[GraphItem] = []
    for pt_path in sorted(tensor_root.glob("*.pt")):
        obj = torch.load(pt_path, map_location="cpu")
        x = obj["x"].float()
        edge_index = obj["edge_index"].long()
        y_value = int(obj["y"].item() if torch.is_tensor(obj["y"]) else obj["y"][0] if isinstance(obj["y"], (list, tuple)) else obj["y"])
        items.append(
            GraphItem(
                run_id=str(obj["run_id"]),
                graph_id=str(obj["graph_id"]),
                x=x,
                adj=build_adjacency(edge_index, x.size(0)),
                y=y_value,
                root_cause_service=str(obj["root_cause_service"]),
            )
        )
    return items


def filter_by_runs(items: list[GraphItem], run_ids: set[str]) -> list[GraphItem]:
    return [item for item in items if item.run_id in run_ids]


def load_reference_graph_metadata(data_root: Path) -> dict[str, object]:
    tensor_root = data_root / "processed" / "rca" / "graph_tensors"
    first_graph = next(iter(sorted(tensor_root.glob("*.pt"))), None)
    if first_graph is None:
        return {"node_feature_names": [], "sample_graph_id": None}
    obj = torch.load(first_graph, map_location="cpu")
    return {
        "node_feature_names": list(obj.get("node_feature_names", [])),
        "sample_graph_id": str(obj.get("graph_id", first_graph.stem)),
    }


def filter_valid_items(items: list[GraphItem]) -> tuple[list[GraphItem], list[dict[str, object]]]:
    valid: list[GraphItem] = []
    skipped: list[dict[str, object]] = []
    for item in items:
        num_nodes = int(item.x.size(0))
        if 0 <= int(item.y) < num_nodes:
            valid.append(item)
        else:
            skipped.append(
                {
                    "run_id": item.run_id,
                    "graph_id": item.graph_id,
                    "root_cause_service": item.root_cause_service,
                    "target_index": int(item.y),
                    "num_nodes": num_nodes,
                }
            )
    return valid, skipped


def evaluate(model: nn.Module, items: list[GraphItem], device: torch.device) -> dict[str, object]:
    model.eval()
    total = len(items)
    if total == 0:
        return {"graphs": 0, "top1_acc": 0.0, "top3_acc": 0.0, "mrr": 0.0, "per_service": {}}

    correct_top1 = 0
    correct_top3 = 0
    reciprocal_rank_sum = 0.0
    per_service: dict[str, dict[str, float]] = {}

    with torch.no_grad():
        for item in items:
            logits = model(item.x.to(device), item.adj.to(device)).cpu()
            target = item.y
            ranked = torch.argsort(logits, descending=True).tolist()
            top1 = ranked[0]
            top3 = ranked[:3]
            if top1 == target:
                correct_top1 += 1
            if target in top3:
                correct_top3 += 1
            rank = ranked.index(target) + 1
            reciprocal_rank_sum += 1.0 / rank

            svc = item.root_cause_service
            bucket = per_service.setdefault(svc, {"count": 0, "top1_hits": 0, "top3_hits": 0})
            bucket["count"] += 1
            bucket["top1_hits"] += int(top1 == target)
            bucket["top3_hits"] += int(target in top3)

    formatted = {}
    for svc, bucket in per_service.items():
        count = int(bucket["count"])
        formatted[svc] = {
            "count": count,
            "top1_acc": bucket["top1_hits"] / count if count else 0.0,
            "top3_acc": bucket["top3_hits"] / count if count else 0.0,
        }

    return {
        "graphs": total,
        "top1_acc": correct_top1 / total,
        "top3_acc": correct_top3 / total,
        "mrr": reciprocal_rank_sum / total,
        "per_service": formatted,
    }


def _to_mlflow_param(value: object) -> str | int | float | bool:
    if isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value, ensure_ascii=False)


def setup_mlflow_if_requested(args: argparse.Namespace) -> bool:
    should_enable = bool(
        args.mlflow
        or os.environ.get("MLFLOW_TRACKING_URI")
        or (
            os.environ.get("DAGSHUB_USERNAME")
            and os.environ.get("DAGSHUB_REPO")
        )
    )
    if not should_enable:
        return False
    if mlflow is None:
        raise RuntimeError("MLflow logging was requested but mlflow is not installed.")

    dagshub_user = os.environ.get("DAGSHUB_USERNAME")
    dagshub_repo = os.environ.get("DAGSHUB_REPO")
    if dagshub_user and dagshub_repo:
        if dagshub is None:
            raise RuntimeError("DagsHub logging was requested but dagshub is not installed.")
        dagshub.init(repo_owner=dagshub_user, repo_name=dagshub_repo, mlflow=True)

    experiment_name = str(args.mlflow_experiment).strip() or "aiops-microservices-demo"
    mlflow.set_experiment(experiment_name)
    return True


def log_mlflow_run(
    *,
    args: argparse.Namespace,
    data_root: Path,
    output_dir: Path,
    train_items: list[GraphItem],
    val_items: list[GraphItem],
    test_items: list[GraphItem],
    best_epoch: int,
    best_val_metrics: dict[str, object],
    test_metrics: dict[str, object],
    reference_graph: dict[str, object],
    in_dim: int,
    device: torch.device,
) -> dict[str, object]:
    if mlflow is None:
        return {}

    with mlflow.start_run(run_name=f"rca_{output_dir.name}"):
        mlflow.set_tag("task", "root_cause_analysis")
        mlflow.set_tag("dataset", data_root.name)
        mlflow.set_tag("model_name", output_dir.name)

        params = {
            "data_root": str(data_root),
            "output_dir": str(output_dir),
            "epochs": args.epochs,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "patience": args.patience,
            "seed": args.seed,
            "device": str(device),
            "in_dim": in_dim,
            "best_epoch": int(best_epoch),
            "train_graphs": int(len(train_items)),
            "val_graphs": int(len(val_items)),
            "test_graphs": int(len(test_items)),
            "node_feature_count": int(len(reference_graph.get("node_feature_names", []))),
        }
        for key, value in params.items():
            mlflow.log_param(key, _to_mlflow_param(value))

        for prefix, metrics_dict in (("val", best_val_metrics), ("test", test_metrics)):
            for key, value in metrics_dict.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(f"{prefix}_{key}", float(value))

        for artifact_name in [
            "best_model.pt",
            "metrics.json",
            "model_config.json",
            "inference_config.json",
            "run_manifest.json",
        ]:
            artifact_path = output_dir / artifact_name
            if artifact_path.exists():
                mlflow.log_artifact(str(artifact_path))

        run = mlflow.active_run()
        return {
            "run_id": run.info.run_id if run else "",
            "experiment_id": run.info.experiment_id if run else "",
            "artifact_uri": mlflow.get_artifact_uri(),
            "tracking_uri": mlflow.get_tracking_uri(),
            "experiment_name": str(args.mlflow_experiment).strip() or "aiops-microservices-demo",
        }

    return {}


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    data_root = args.data_root
    output_dir = args.output_dir or (data_root / "models" / "rca_gat_like")
    output_dir.mkdir(parents=True, exist_ok=True)

    train_runs = set(read_run_ids(data_root / "splits" / "train_runs.txt"))
    val_runs = set(read_run_ids(data_root / "splits" / "val_runs.txt"))
    test_runs = set(read_run_ids(data_root / "splits" / "test_runs.txt"))

    all_items = load_graph_items(data_root)
    all_items, skipped_items = filter_valid_items(all_items)
    train_items = filter_by_runs(all_items, train_runs)
    val_items = filter_by_runs(all_items, val_runs)
    test_items = filter_by_runs(all_items, test_runs)

    if not train_items:
        raise ValueError("No training graphs found.")

    in_dim = train_items[0].x.size(1)
    reference_graph = load_reference_graph_metadata(data_root)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model = SimpleGraphAttention(in_dim=in_dim, hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_state = None
    best_epoch = 0
    best_ranking = -1.0
    best_val_metrics = None
    epochs_without_improve = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        random.shuffle(train_items)

        for item in train_items:
            optimizer.zero_grad(set_to_none=True)
            logits = model(item.x.to(device), item.adj.to(device))
            target = torch.tensor([item.y], dtype=torch.long, device=device)
            loss = F.cross_entropy(logits.unsqueeze(0), target)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())

        avg_loss = total_loss / max(len(train_items), 1)
        val_metrics = evaluate(model, val_items, device)
        ranking_score = val_metrics["top1_acc"] + 0.5 * val_metrics["mrr"] + 0.25 * val_metrics["top3_acc"]
        history.append(
            {
                "epoch": epoch,
                "loss": avg_loss,
                "val_top1": val_metrics["top1_acc"],
                "val_top3": val_metrics["top3_acc"],
                "val_mrr": val_metrics["mrr"],
                "ranking_score": ranking_score,
            }
        )

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"Epoch {epoch:03d}/{args.epochs} | Loss: {avg_loss:.4f} | "
                f"Val Top-1: {val_metrics['top1_acc']*100:.1f}% | "
                f"Val Top-3: {val_metrics['top3_acc']*100:.1f}% | Val MRR: {val_metrics['mrr']:.3f}"
            )

        if ranking_score > best_ranking:
            best_ranking = ranking_score
            best_epoch = epoch
            best_val_metrics = val_metrics
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            epochs_without_improve = 0
            print(
                f"  [best] Epoch {epoch:03d}: ranking_score={ranking_score:.4f} "
                f"top1={val_metrics['top1_acc']*100:.0f}% mrr={val_metrics['mrr']:.3f}"
            )
        else:
            epochs_without_improve += 1

        if epochs_without_improve >= args.patience:
            print(f"Early stopping at epoch {epoch}.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = evaluate(model, test_items, device)
    payload = {
        "data_root": str(data_root),
        "device": str(device),
        "total_graphs_loaded": len(all_items) + len(skipped_items),
        "skipped_invalid_graphs": len(skipped_items),
        "skipped_examples": skipped_items[:10],
        "train_graphs": len(train_items),
        "val_graphs": len(val_items),
        "test_graphs": len(test_items),
        "best_epoch": best_epoch,
        "best_val_metrics": best_val_metrics,
        "test_metrics": test_metrics,
        "history": history,
        "config": {
            "epochs": args.epochs,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "patience": args.patience,
            "seed": args.seed,
        },
    }

    torch.save(best_state, output_dir / "best_model.pt")
    (output_dir / "metrics.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    model_config = {
        "model_name": output_dir.name,
        "model_type": "simple_graph_attention",
        "in_dim": in_dim,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "node_feature_names": reference_graph["node_feature_names"],
        "state_dict_artifact": "best_model.pt",
        "metrics_artifact": "metrics.json",
    }
    (output_dir / "model_config.json").write_text(json.dumps(model_config, indent=2, ensure_ascii=False), encoding="utf-8")
    inference_config = {
        "model_name": output_dir.name,
        "model_type": "simple_graph_attention",
        "top_k_default": 3,
        "node_feature_names": reference_graph["node_feature_names"],
        "sample_graph_id": reference_graph["sample_graph_id"],
        "model_config_artifact": "model_config.json",
        "state_dict_artifact": "best_model.pt",
        "metrics_artifact": "metrics.json",
        "run_manifest_artifact": "run_manifest.json",
    }
    (output_dir / "inference_config.json").write_text(json.dumps(inference_config, indent=2, ensure_ascii=False), encoding="utf-8")

    package_versions = collect_package_versions(["torch", "mlflow", "dagshub"])
    manifest = build_run_manifest(
        repo_root=REPO_ROOT,
        task="rca_training",
        output_dir=output_dir,
        train_script=Path(__file__).resolve(),
        data_root=data_root,
        args=vars(args),
        artifacts={
            "state_dict": "best_model.pt",
            "metrics": "metrics.json",
            "model_config": "model_config.json",
            "inference_config": "inference_config.json",
        },
        dataset={
            "graph_tensor_root": data_root / "processed" / "rca" / "graph_tensors",
            "train_graphs": len(train_items),
            "val_graphs": len(val_items),
            "test_graphs": len(test_items),
            "skipped_invalid_graphs": len(skipped_items),
            "node_feature_names": reference_graph["node_feature_names"],
            "sample_graph_id": reference_graph["sample_graph_id"],
        },
        package_versions=package_versions,
        extra={
            "best_epoch": best_epoch,
            "device": str(device),
            "best_val_metrics": best_val_metrics,
            "test_metrics": test_metrics,
            "config": payload["config"],
        },
    )
    mlflow_metadata: dict[str, object] = {}
    write_run_manifest(output_dir, manifest)

    if setup_mlflow_if_requested(args):
        mlflow_metadata = log_mlflow_run(
            args=args,
            data_root=data_root,
            output_dir=output_dir,
            train_items=train_items,
            val_items=val_items,
            test_items=test_items,
            best_epoch=best_epoch,
            best_val_metrics=best_val_metrics,
            test_metrics=test_metrics,
            reference_graph=reference_graph,
            in_dim=in_dim,
            device=device,
        )
        if mlflow_metadata:
            manifest["mlflow"] = mlflow_metadata
            write_run_manifest(output_dir, manifest)

    set_stage(
        data_root / "models",
        "rca",
        "candidate",
        output_dir.name,
        notes="Updated after RCA training.",
        metadata={f"mlflow_{key}": value for key, value in mlflow_metadata.items()} if mlflow_metadata else None,
    )

    print("Validation metrics:")
    print(json.dumps(best_val_metrics, indent=2, ensure_ascii=False))
    print("Test metrics:")
    print(json.dumps(test_metrics, indent=2, ensure_ascii=False))
    print(f"Saved artifacts to {output_dir}")
    print(f"Updated candidate registry at {data_root / 'models' / 'model_registry.json'}")


if __name__ == "__main__":
    main()
