from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aiops_framework.inference.common.artifact_registry import get_model_summary
from aiops_framework.inference.rca_service.model_loader import load_artifacts
from aiops_framework.inference.rca_service.predictor import predict_graph


DEFAULT_DATA_ROOT = REPO_ROOT / "data_rca_balanced_v3"
DEFAULT_MODELS_ROOT = DEFAULT_DATA_ROOT / "models"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare RCA models on the real test split graph tensors.")
    parser.add_argument("--system-id", default="online-boutique")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--models-root", type=Path, default=DEFAULT_MODELS_ROOT)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--include-candidates", action="store_true", help="Include registry candidates in addition to production.")
    parser.add_argument("--model-name", action="append", default=[], help="Optional explicit model name(s) to compare.")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/evaluation/rca_model_comparison"))
    return parser.parse_args()


def read_run_ids(path: Path) -> set[str]:
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def load_test_graphs(data_root: Path, allowed_runs: set[str]) -> list[dict[str, Any]]:
    tensor_root = data_root / "processed" / "rca" / "graph_tensors"
    items: list[dict[str, Any]] = []
    for pt_path in sorted(tensor_root.glob("*.pt")):
        payload = torch.load(pt_path, map_location="cpu")
        run_id = str(payload.get("run_id", "")).strip()
        if allowed_runs and run_id not in allowed_runs:
            continue
        y_raw = payload.get("y")
        if torch.is_tensor(y_raw):
            target_index = int(y_raw.flatten()[0].item())
        elif isinstance(y_raw, (list, tuple)):
            target_index = int(y_raw[0])
        else:
            target_index = int(y_raw)
        items.append(
            {
                "graph_id": str(payload.get("graph_id", pt_path.stem)),
                "run_id": run_id,
                "fault_family": str(payload.get("fault_family", "")),
                "root_cause_service": str(payload.get("root_cause_service", "")),
                "target_index": target_index,
                "node_names": list(payload.get("node_names", [])),
                "x": payload["x"].tolist(),
                "edge_index": payload["edge_index"].t().tolist() if payload["edge_index"].numel() > 0 else [],
            }
        )
    return items


def resolve_models(args: argparse.Namespace) -> list[dict[str, str]]:
    summary = get_model_summary(args.models_root, system_id=args.system_id, model_type="rca")
    records: list[dict[str, Any]] = []
    if summary.get("production"):
        records.append(dict(summary["production"]))
    if args.include_candidates:
        records.extend(dict(item) for item in (summary.get("candidates") or []))

    explicit_names = {name.strip() for name in args.model_name if name.strip()}
    models: list[dict[str, str]] = []
    seen: set[str] = set()

    for record in records:
        model_name = str(record.get("model_name") or record.get("model_id") or "").strip()
        if not model_name or model_name in seen:
            continue
        if explicit_names and model_name not in explicit_names:
            continue
        artifact_dir = Path(str(record.get("artifact_dir") or model_name))
        if not artifact_dir.is_absolute():
            artifact_dir = (args.models_root / artifact_dir).resolve()
        models.append(
            {
                "model_name": model_name,
                "artifact_dir": str(artifact_dir),
                "status": str(record.get("status") or ""),
            }
        )
        seen.add(model_name)

    for model_name in sorted(explicit_names - seen):
        artifact_dir = (args.models_root / model_name).resolve()
        models.append({"model_name": model_name, "artifact_dir": str(artifact_dir), "status": "explicit"})

    if not models:
        raise ValueError("No RCA models resolved for comparison.")
    return models


def compare_model(artifact_dir: Path, model_name: str, graphs: list[dict[str, Any]], top_k: int) -> dict[str, Any]:
    artifacts = load_artifacts(artifact_dir=artifact_dir, device="cpu")
    top1_hits = 0
    topk_hits = 0
    reciprocal_rank_sum = 0.0
    per_fault_family: dict[str, dict[str, float]] = defaultdict(lambda: {"graphs": 0, "top1_hits": 0, "topk_hits": 0, "mrr_sum": 0.0})
    examples: list[dict[str, Any]] = []

    for graph in graphs:
        result = predict_graph(
            artifacts=artifacts,
            x_rows=graph["x"],
            edge_index_rows=graph["edge_index"],
            node_names=graph["node_names"],
            top_k=top_k,
            graph_id=graph["graph_id"],
            metadata={"run_id": graph["run_id"], "fault_family": graph["fault_family"]},
        )
        ranked = result["topk"]
        ranked_names = [str(item["service_name"]) for item in ranked]
        true_service = graph["root_cause_service"]
        hit_top1 = bool(ranked_names and ranked_names[0] == true_service)
        hit_topk = true_service in ranked_names

        full_rank_result = predict_graph(
            artifacts=artifacts,
            x_rows=graph["x"],
            edge_index_rows=graph["edge_index"],
            node_names=graph["node_names"],
            top_k=len(graph["node_names"]),
            graph_id=graph["graph_id"],
            metadata={"run_id": graph["run_id"], "fault_family": graph["fault_family"]},
        )
        full_ranked_names = [str(item["service_name"]) for item in full_rank_result["topk"]]
        rank = full_ranked_names.index(true_service) + 1 if true_service in full_ranked_names else len(full_ranked_names) + 1

        top1_hits += int(hit_top1)
        topk_hits += int(hit_topk)
        reciprocal_rank_sum += 1.0 / rank

        bucket = per_fault_family[graph["fault_family"] or "unknown"]
        bucket["graphs"] += 1
        bucket["top1_hits"] += int(hit_top1)
        bucket["topk_hits"] += int(hit_topk)
        bucket["mrr_sum"] += 1.0 / rank

        if len(examples) < 12:
            examples.append(
                {
                    "graph_id": graph["graph_id"],
                    "fault_family": graph["fault_family"],
                    "true_root_cause": true_service,
                    "predicted_top1": ranked_names[0] if ranked_names else "",
                    "topk": ranked,
                }
            )

    total = len(graphs)
    fault_family_summary = {}
    for fault_family, bucket in per_fault_family.items():
        count = int(bucket["graphs"])
        fault_family_summary[fault_family] = {
            "graphs": count,
            "top1_acc": bucket["top1_hits"] / count if count else None,
            "topk_acc": bucket["topk_hits"] / count if count else None,
            "mrr": bucket["mrr_sum"] / count if count else None,
        }

    return {
        "model_name": model_name,
        "artifact_dir": str(artifact_dir),
        "comparison_mode": "live_test_split",
        "graphs": total,
        "top1_acc": top1_hits / total if total else None,
        "topk_acc": topk_hits / total if total else None,
        "mrr": reciprocal_rank_sum / total if total else None,
        "fault_family_summary": fault_family_summary,
        "examples": examples,
    }


def compare_from_artifact_metrics(artifact_dir: Path, model_name: str, top_k: int, error: str) -> dict[str, Any]:
    metrics_path = artifact_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"No metrics.json found for fallback comparison: {metrics_path}")
    payload = json.loads(metrics_path.read_text(encoding="utf-8-sig"))
    spaces = [payload]
    for key in ("test_metrics", "best_val_metrics", "val_metrics"):
        value = payload.get(key)
        if isinstance(value, dict):
            spaces.append(value)

    def _find(*keys: str) -> float | None:
        for space in spaces:
            for key in keys:
                if key in space and space[key] is not None:
                    return float(space[key])
        return None

    metrics_note = payload.get("metrics_note")
    if isinstance(metrics_note, dict):
        spaces.append(metrics_note)

    num_graphs = None
    for key in ("graphs", "num_graphs", "test_graphs"):
        for space in spaces:
            if key in space and space[key] is not None:
                num_graphs = int(space[key])
                break
        if num_graphs is not None:
            break

    graphs = num_graphs or 0
    top1_acc = _find("top1_acc", "top1_accuracy", "case_cv_top1", "leave_fault_top1", "leave_service_top1")
    topk_acc = _find(
        f"top{top_k}_acc",
        "top3_acc",
        "top3_accuracy",
        "recall_at_3",
        "case_cv_top3",
        "leave_fault_top3",
        "leave_service_top3",
    )
    mrr = _find("mrr", "case_cv_mrr", "leave_fault_mrr", "leave_service_mrr")

    return {
        "model_name": model_name,
        "artifact_dir": str(artifact_dir),
        "comparison_mode": "artifact_metrics_fallback",
        "graphs": graphs,
        "top1_acc": top1_acc,
        "topk_acc": topk_acc,
        "mrr": mrr,
        "fault_family_summary": {},
        "examples": [],
        "error": error,
    }


def write_outputs(output_dir: Path, comparisons: list[dict[str, Any]], system_id: str, top_k: int) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{system_id}_rca_model_comparison.json"
    md_path = output_dir / f"{system_id}_rca_model_comparison.md"
    csv_path = output_dir / f"{system_id}_rca_model_comparison.csv"

    json_path.write_text(json.dumps(comparisons, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("model_name,graphs,top1_acc,top{}_acc,mrr\n".format(top_k))
        for item in comparisons:
            handle.write(
                f"{item['model_name']},{item['graphs']},{item['top1_acc']},{item['topk_acc']},{item['mrr']}\n"
            )

    lines = [
        f"# RCA Model Comparison for {system_id}",
        "",
        f"Top-k threshold: **{top_k}**",
        "",
        f"| Model | Graphs | Top-1 Acc | Top-{top_k} Acc | MRR |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in comparisons:
        if item.get("error"):
            lines.append(f"| {item['model_name']} | 0 | n/a | n/a | n/a |")
        else:
            lines.append(
                f"| {item['model_name']} | {item['graphs']} | {item['top1_acc']:.4f} | {item['topk_acc']:.4f} | {item['mrr']:.4f} |"
            )
    md_path.write_text("\n".join(lines), encoding="utf-8")

    return {"json": str(json_path), "csv": str(csv_path), "markdown": str(md_path)}


def main() -> None:
    args = parse_args()
    test_runs = read_run_ids(args.data_root / "splits" / "test_runs.txt")
    graphs = load_test_graphs(args.data_root, test_runs)
    models = resolve_models(args)
    comparisons = []
    for item in models:
        try:
            comparisons.append(
                compare_model(Path(item["artifact_dir"]), item["model_name"], graphs, top_k=max(1, int(args.top_k)))
            )
        except Exception as exc:
            comparisons.append(compare_from_artifact_metrics(Path(item["artifact_dir"]), item["model_name"], max(1, int(args.top_k)), str(exc)))
    comparisons = sorted(
        comparisons,
        key=lambda item: (item.get("error") is not None, -float(item["mrr"] or 0.0)),
    )
    outputs = write_outputs(args.output_dir, comparisons, args.system_id, max(1, int(args.top_k)))
    print(json.dumps({"models": [item["model_name"] for item in comparisons], "outputs": outputs}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
