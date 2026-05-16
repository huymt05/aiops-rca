from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update image tags in deploy/aiops kustomize overlays.")
    parser.add_argument("--env", choices=["dev", "prod"], required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument(
        "--image",
        action="append",
        dest="images",
        default=[],
        help="Specific fully-qualified image name to update. Repeat for multiple images.",
    )
    return parser.parse_args()


def update_tag_block(text: str, image_name: str, tag: str) -> str:
    lines = text.splitlines()
    for idx, line in enumerate(lines):
        if f"name: {image_name}" in line:
            for inner in range(idx + 1, min(idx + 6, len(lines))):
                if "newTag:" in lines[inner]:
                    prefix = lines[inner].split("newTag:")[0]
                    lines[inner] = f"{prefix}newTag: {tag}"
                    break
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    kustomization = repo_root / "deploy" / "aiops" / "environments" / args.env / "kustomization.yaml"
    text = kustomization.read_text(encoding="utf-8")

    images = args.images or [
        "ghcr.io/huymt05/aiops-anomaly-service",
        "ghcr.io/huymt05/aiops-rca-service",
        "ghcr.io/huymt05/aiops-orchestrator",
        "ghcr.io/huymt05/aiops-dashboard",
    ]

    for image in images:
        text = update_tag_block(text, image, args.tag)

    kustomization.write_text(text, encoding="utf-8")
    print(f"Updated {kustomization} to tag={args.tag} for {', '.join(images)}")


if __name__ == "__main__":
    main()
