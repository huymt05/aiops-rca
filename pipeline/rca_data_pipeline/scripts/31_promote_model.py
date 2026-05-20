from __future__ import annotations

import argparse
from pathlib import Path

import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aiops_framework.inference.common.artifact_registry import DEFAULT_SYSTEM_ID, promote_stage, registry_path, set_stage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Promote a candidate model to production in the local model registry.")
    parser.add_argument("--task", choices=["anomaly", "rca"], required=True)
    parser.add_argument("--models-root", type=Path, required=True)
    parser.add_argument("--source-stage", default="candidate")
    parser.add_argument("--target-stage", default="production")
    parser.add_argument("--previous-stage", default="previous")
    parser.add_argument("--system-id", default=DEFAULT_SYSTEM_ID, help="Registry system id to promote within.")
    parser.add_argument("--model-name", default=None, help="Optional direct model name to assign before promotion.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.model_name:
        set_stage(
            args.models_root,
            args.task,
            args.source_stage,
            args.model_name,
            notes="Manually selected before promotion.",
            system_id=args.system_id,
        )
    promote_stage(
        args.models_root,
        args.task,
        source_stage=args.source_stage,
        target_stage=args.target_stage,
        previous_stage=args.previous_stage,
        system_id=args.system_id,
    )
    print(f"Updated registry: {registry_path(args.models_root)}")


if __name__ == "__main__":
    main()
