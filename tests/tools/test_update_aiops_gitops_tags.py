from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[2] / "tools" / "update_aiops_gitops_tags.py"
SPEC = spec_from_file_location("update_aiops_gitops_tags", MODULE_PATH)
MODULE = module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_update_tag_block_replaces_matching_image_only() -> None:
    original = """images:
  - name: ghcr.io/huymt05/aiops-anomaly-service
    newTag: old-tag
  - name: ghcr.io/huymt05/aiops-rca-service
    newTag: keep-me
"""

    updated = MODULE.update_tag_block(
        original,
        "ghcr.io/huymt05/aiops-anomaly-service",
        "sha-1234567",
    )

    assert "newTag: sha-1234567" in updated
    assert "newTag: keep-me" in updated


def test_update_tag_block_leaves_text_unchanged_when_image_missing() -> None:
    original = """images:
  - name: ghcr.io/huymt05/aiops-dashboard
    newTag: current
"""

    updated = MODULE.update_tag_block(
        original,
        "ghcr.io/huymt05/aiops-orchestrator",
        "sha-abcdef0",
    )

    assert updated == original
