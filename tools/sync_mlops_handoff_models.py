from __future__ import annotations

import argparse
import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HANDOFF_ROOT = Path(r"E:\mlops_train_handoff")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sync trained anomaly/RCA model artifacts from a handoff directory into runtime data roots."
    )
    parser.add_argument("--handoff-root", type=Path, default=DEFAULT_HANDOFF_ROOT)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def copy_tree(source: Path, dest: Path, *, dry_run: bool) -> None:
    if not source.exists():
        raise FileNotFoundError(f"Source path not found: {source}")
    if dry_run:
        print(f"[dry-run] copy {source} -> {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, dest, dirs_exist_ok=True)
    print(f"Copied {source} -> {dest}")


def main() -> None:
    args = parse_args()
    handoff_root = args.handoff_root.resolve()
    repo_root = args.repo_root.resolve()

    anomaly_src = handoff_root / "datasets" / "anomaly_train_dataset" / "models"
    rca_src = handoff_root / "datasets" / "rca_train_dataset" / "models"
    anomaly_dest = repo_root / "data_anomaly_balanced_v3" / "models"
    rca_dest = repo_root / "data_rca_balanced_v3" / "models"

    copy_tree(anomaly_src, anomaly_dest, dry_run=args.dry_run)
    copy_tree(rca_src, rca_dest, dry_run=args.dry_run)

    anomaly_registry = anomaly_dest / "model_registry.json"
    rca_registry = rca_dest / "model_registry.json"
    print(f"Anomaly registry: {anomaly_registry}")
    print(f"RCA registry: {rca_registry}")


if __name__ == "__main__":
    main()
