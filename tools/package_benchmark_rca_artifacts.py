from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import joblib
import torch


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def package_rf(source_dir: Path, dest_dir: Path, model_name: str) -> None:
    source_model = source_dir / "model.joblib"
    source_metrics = source_dir / "metrics.json"
    payload = joblib.load(source_model)

    shutil.copy2(source_model, dest_dir / "model.joblib")
    if source_metrics.exists():
        shutil.copy2(source_metrics, dest_dir / "metrics.json")

    model_config = {
        "model_type": "random_forest_service_ranker",
        "joblib_artifact": "model.joblib",
        "feature_cols": list(payload.get("feature_cols", [])),
        "num_features": int(payload.get("num_features", len(payload.get("feature_cols", [])))),
    }
    inference_config = {
        "model_name": model_name,
        "model_type": "random_forest_service_ranker",
        "dataset": payload.get("dataset", "RCAEval RE2-OB"),
        "top_k_default": 3,
    }
    write_json(dest_dir / "model_config.json", model_config)
    write_json(dest_dir / "inference_config.json", inference_config)


def package_gat(source_dir: Path, dest_dir: Path, model_name: str) -> None:
    source_model = source_dir / "best_model.pt"
    source_metrics = source_dir / "metrics.json"
    payload = torch.load(source_model, map_location="cpu")
    model_cfg = payload.get("model_config", {})

    shutil.copy2(source_model, dest_dir / "best_model.pt")
    if source_metrics.exists():
        shutil.copy2(source_metrics, dest_dir / "metrics.json")

    model_config = {
        "model_type": "simple_graph_attention",
        "state_dict_artifact": "best_model.pt",
        "in_dim": int(model_cfg["in_dim"]),
        "hidden_dim": int(model_cfg["hidden_dim"]),
        "dropout": float(model_cfg["dropout"]),
    }
    inference_config = {
        "model_name": model_name,
        "model_type": "simple_graph_attention",
        "dataset": "RCAEval RE2-OB",
        "top_k_default": 3,
    }
    write_json(dest_dir / "model_config.json", model_config)
    write_json(dest_dir / "inference_config.json", inference_config)


def package_hgnn(source_dir: Path, dest_dir: Path, model_name: str) -> None:
    source_model = source_dir / "best_model.pt"
    source_metrics = source_dir / "metrics.json"
    payload = torch.load(source_model, map_location="cpu")
    model_cfg = payload.get("model_config", {})
    feature_groups = model_cfg.get("feature_groups", {})
    inferred_dim = sum(len(v) for v in feature_groups.values())

    shutil.copy2(source_model, dest_dir / "best_model.pt")
    if source_metrics.exists():
        shutil.copy2(source_metrics, dest_dir / "metrics.json")

    model_config = {
        "model_type": "hetero_telemetry_gnn",
        "state_dict_artifact": "best_model.pt",
        "in_dim": int(inferred_dim),
        "hidden_dim": int(model_cfg["hidden_dim"]),
        "dropout": float(model_cfg["dropout"]),
        "num_layers": int(model_cfg["num_layers"]),
        "feature_groups": feature_groups,
    }
    inference_config = {
        "model_name": model_name,
        "model_type": "hetero_telemetry_gnn",
        "dataset": "RCAEval RE2-OB",
        "top_k_default": 3,
    }
    write_json(dest_dir / "model_config.json", model_config)
    write_json(dest_dir / "inference_config.json", inference_config)


def main() -> None:
    parser = argparse.ArgumentParser(description="Package RCA benchmark artifacts for aiops rca_service")
    parser.add_argument("--model-family", choices=["rf", "gat", "hgnn"], required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--dest-dir", type=Path, required=True)
    parser.add_argument("--model-name", type=str, default="")
    args = parser.parse_args()

    dest_dir = ensure_dir(args.dest_dir)
    model_name = args.model_name.strip() or dest_dir.name

    if args.model_family == "rf":
        package_rf(args.source_dir, dest_dir, model_name)
    elif args.model_family == "gat":
        package_gat(args.source_dir, dest_dir, model_name)
    else:
        package_hgnn(args.source_dir, dest_dir, model_name)

    print(f"Packaged {args.model_family} artifact to: {dest_dir}")


if __name__ == "__main__":
    main()
