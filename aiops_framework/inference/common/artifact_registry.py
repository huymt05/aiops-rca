from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_STAGE = "production"
DEFAULT_SYSTEM_ID = "online-boutique"
REGISTRY_FILENAME = "model_registry.json"
SCHEMA_VERSION_V1 = "1.0"
SCHEMA_VERSION_V2 = "2.0"
MAX_CANDIDATES = 3


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def registry_path(models_root: Path) -> Path:
    return Path(models_root) / REGISTRY_FILENAME


def _json_read(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8")) or {}


def write_registry(models_root: Path, payload: dict[str, Any]) -> Path:
    path = registry_path(models_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload["updated_at"] = now_iso()
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return path


def _empty_v1(task: str | None = None) -> dict[str, Any]:
    return {"schema_version": SCHEMA_VERSION_V1, "task": task or "unknown", "stages": {}}


def _empty_task_block() -> dict[str, Any]:
    return {"production": None, "previous": [], "candidates": []}


def _empty_v2(task: str | None = None) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION_V2,
        "task": task or "unknown",
        "updated_at": now_iso(),
        "systems": {},
        "stages": {},
    }


def _normalize_system_id(system_id: str | None) -> str:
    value = str(system_id or DEFAULT_SYSTEM_ID).strip()
    return value or DEFAULT_SYSTEM_ID


def _infer_model_type(task: str | None, models_root: Path | None = None) -> str:
    if task:
        value = str(task).strip().lower()
        if "rca" in value:
            return "rca"
        if "anomaly" in value:
            return "anomaly"
        return value
    if models_root:
        value = str(models_root).lower()
        if "rca" in value:
            return "rca"
        if "anomaly" in value:
            return "anomaly"
    return "unknown"


def _ensure_system_task(payload: dict[str, Any], system_id: str, model_type: str) -> dict[str, Any]:
    systems = payload.setdefault("systems", {})
    system_block = systems.setdefault(system_id, {})
    task_block = system_block.setdefault(model_type, _empty_task_block())
    task_block.setdefault("production", None)
    task_block.setdefault("previous", [])
    task_block.setdefault("candidates", [])
    if task_block["previous"] is None:
        task_block["previous"] = []
    if task_block["candidates"] is None:
        task_block["candidates"] = []
    return task_block


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _default_rank_score(model_type: str, metrics: dict[str, Any]) -> float:
    if not metrics:
        return 0.0

    def num(*keys: str, default: float = 0.0) -> float:
        for key in keys:
            if key in metrics:
                return _to_float(metrics.get(key), default)
        return default

    if model_type == "rca":
        top1 = num("top1_acc", "top1_accuracy", "accuracy", default=0.0)
        top3 = num("top3_acc", "top3_accuracy", "recall_at_3", default=0.0)
        mrr = num("mrr", default=0.0)
        return 0.50 * top3 + 0.30 * mrr + 0.20 * top1

    if model_type == "anomaly":
        f1 = num("f1_anomaly", "f1", "f1_score", default=0.0)
        balanced_accuracy = num("balanced_accuracy", default=0.0)
        roc_auc = num("roc_auc", "auc", default=0.0)
        pr_auc = num("average_precision", "pr_auc", default=0.0)
        if pr_auc > 0:
            return 0.45 * f1 + 0.25 * balanced_accuracy + 0.20 * pr_auc + 0.10 * roc_auc
        return 0.55 * f1 + 0.30 * balanced_accuracy + 0.15 * roc_auc

    return num("rank_score", "score", "accuracy", default=0.0)


def _read_metrics_bundle(artifact_dir: Path) -> tuple[dict[str, Any], str | None, str | None]:
    metrics: dict[str, Any] = {}
    trained_at: str | None = None
    model_version: str | None = None

    run_manifest_path = artifact_dir / "run_manifest.json"
    if run_manifest_path.exists():
        try:
            manifest = _json_read(run_manifest_path)
            manifest_metrics = manifest.get("metrics") or manifest.get("evaluation") or manifest.get("eval_metrics") or {}
            if isinstance(manifest_metrics, dict):
                metrics.update(manifest_metrics)
            trained_at = manifest.get("trained_at") or manifest.get("created_at") or manifest.get("timestamp") or trained_at
            model_version = manifest.get("model_version") or manifest.get("version") or model_version
        except Exception:
            pass

    metrics_path = artifact_dir / "metrics.json"
    if metrics_path.exists():
        try:
            raw_metrics = _json_read(metrics_path)
            if isinstance(raw_metrics, dict):
                metrics.update(raw_metrics)
        except Exception:
            pass

    return metrics, str(trained_at) if trained_at else None, str(model_version) if model_version else None


def _stage_entry_to_model_record(
    *,
    model_name: str,
    models_root: Path,
    model_type: str,
    stage: str,
    notes: str = "",
    updated_at: str | None = None,
) -> dict[str, Any]:
    artifact_dir = Path(models_root) / model_name
    metrics, trained_at, model_version = _read_metrics_bundle(artifact_dir)
    run_manifest_path = artifact_dir / "run_manifest.json"

    return {
        "model_id": model_name,
        "model_name": model_name,
        "model_version": model_version or model_name,
        "model_type": model_type,
        "artifact_dir": model_name,
        "run_manifest_path": f"{model_name}/run_manifest.json" if run_manifest_path.exists() else "",
        "metrics": metrics,
        "rank_score": float(_default_rank_score(model_type, metrics)),
        "trained_at": trained_at or updated_at or now_iso(),
        "updated_at": updated_at or now_iso(),
        "status": stage,
        "notes": notes,
    }


def _normalize_candidate(candidate: dict[str, Any], model_type: str) -> dict[str, Any]:
    item = dict(candidate)
    model_name = str(
        item.get("model_name")
        or item.get("model_id")
        or Path(str(item.get("artifact_dir", ""))).name
        or "unknown_model"
    )
    metrics = item.get("metrics") or {}
    if not isinstance(metrics, dict):
        metrics = {}

    item.setdefault("model_id", model_name)
    item["model_name"] = model_name
    item.setdefault("model_version", str(item.get("version") or model_name))
    item["model_type"] = model_type
    item["artifact_dir"] = str(item.get("artifact_dir") or model_name)
    item["run_manifest_path"] = str(item.get("run_manifest_path") or "")
    item["metrics"] = metrics
    item["rank_score"] = float(item.get("rank_score", _default_rank_score(model_type, metrics)) or 0.0)
    item.setdefault("trained_at", now_iso())
    item["updated_at"] = now_iso()
    item["status"] = "candidate"
    return item


def _sync_legacy_stages(payload: dict[str, Any], *, system_id: str, model_type: str) -> dict[str, Any]:
    task_block = _ensure_system_task(payload, system_id, model_type)
    stages: dict[str, Any] = {}

    production = task_block.get("production")
    if isinstance(production, dict) and production.get("model_name"):
        stages["production"] = {
            "model_name": production["model_name"],
            "updated_at": production.get("updated_at", now_iso()),
            "notes": production.get("notes", ""),
        }

    candidates = task_block.get("candidates") or []
    if candidates:
        first = candidates[0]
        stages["candidate"] = {
            "model_name": first.get("model_name"),
            "updated_at": first.get("updated_at", now_iso()),
            "notes": first.get("notes", ""),
        }

    previous_items = task_block.get("previous") or []
    if previous_items:
        first = previous_items[0]
        stages["previous"] = {
            "model_name": first.get("model_name"),
            "updated_at": first.get("updated_at", now_iso()),
            "notes": first.get("notes", ""),
        }

    payload["stages"] = stages
    return payload


def migrate_v1_to_v2(
    payload: dict[str, Any],
    *,
    models_root: Path,
    task: str | None = None,
    default_system_id: str = DEFAULT_SYSTEM_ID,
) -> dict[str, Any]:
    if payload.get("schema_version") == SCHEMA_VERSION_V2:
        payload.setdefault("systems", {})
        payload.setdefault("stages", {})
        payload.setdefault("updated_at", now_iso())
        if task and not payload.get("task"):
            payload["task"] = task
        return payload

    if not payload:
        return _empty_v2(task=task)

    model_type = _infer_model_type(task or payload.get("task"), models_root)
    migrated = _empty_v2(task=task or payload.get("task") or model_type)
    system_id = _normalize_system_id(default_system_id)
    task_block = _ensure_system_task(migrated, system_id, model_type)

    stages = payload.get("stages") or {}
    if not isinstance(stages, dict):
        stages = {}

    for stage, entry in stages.items():
        if not isinstance(entry, dict):
            continue
        model_name = entry.get("model_name")
        if not model_name:
            continue
        record = _stage_entry_to_model_record(
            model_name=str(model_name),
            models_root=models_root,
            model_type=model_type,
            stage=stage,
            notes=str(entry.get("notes") or ""),
            updated_at=entry.get("updated_at"),
        )
        if stage == "production":
            record["status"] = "production"
            task_block["production"] = record
        elif stage == "previous":
            record["status"] = "previous"
            task_block["previous"].append(record)
        else:
            record["status"] = "candidate"
            task_block["candidates"].append(record)

    task_block["candidates"] = sorted(
        task_block["candidates"],
        key=lambda item: float(item.get("rank_score", 0.0) or 0.0),
        reverse=True,
    )[:MAX_CANDIDATES]

    return _sync_legacy_stages(migrated, system_id=system_id, model_type=model_type)


def load_registry(
    models_root: Path,
    task: str | None = None,
    *,
    system_id: str | None = None,
    migrate: bool = True,
    write_back: bool = False,
) -> dict[str, Any]:
    payload = _json_read(registry_path(models_root))

    if not payload:
        payload = _empty_v2(task=task) if migrate else _empty_v1(task=task)
    elif migrate:
        payload = migrate_v1_to_v2(
            payload,
            models_root=Path(models_root),
            task=task,
            default_system_id=_normalize_system_id(system_id),
        )
    else:
        if "stages" not in payload or not isinstance(payload["stages"], dict):
            payload["stages"] = {}
        if task and not payload.get("task"):
            payload["task"] = task

    if migrate:
        payload["schema_version"] = SCHEMA_VERSION_V2
        payload.setdefault("systems", {})
        payload.setdefault("stages", {})
        if task and not payload.get("task"):
            payload["task"] = task
        if write_back:
            write_registry(models_root, payload)

    return payload


def load_registry_v1(models_root: Path, task: str | None = None) -> dict[str, Any]:
    payload = _json_read(registry_path(models_root))
    if not payload:
        return _empty_v1(task=task)
    if payload.get("schema_version") == SCHEMA_VERSION_V2:
        payload.setdefault("stages", {})
    if "stages" not in payload or not isinstance(payload["stages"], dict):
        payload["stages"] = {}
    if task and not payload.get("task"):
        payload["task"] = task
    return {"schema_version": SCHEMA_VERSION_V1, "task": payload.get("task") or task or "unknown", "stages": payload["stages"]}


def set_stage(
    models_root: Path,
    task: str,
    stage: str,
    model_name: str,
    notes: str | None = None,
    *,
    system_id: str = DEFAULT_SYSTEM_ID,
    metadata: dict[str, Any] | None = None,
) -> Path:
    models_root = Path(models_root)
    model_type = _infer_model_type(task, models_root)
    system_id = _normalize_system_id(system_id)
    payload = load_registry(
        models_root,
        task=task,
        system_id=system_id,
        migrate=True,
        write_back=False,
    )
    task_block = _ensure_system_task(payload, system_id, model_type)
    record = _stage_entry_to_model_record(
        model_name=model_name,
        models_root=models_root,
        model_type=model_type,
        stage=stage,
        notes=notes or "",
        updated_at=now_iso(),
    )

    if stage == "production":
        record["status"] = "production"
        task_block["production"] = record
    elif stage == "previous":
        record["status"] = "previous"
        task_block["previous"] = [record, *task_block.get("previous", [])][:MAX_CANDIDATES]
    else:
        record["status"] = "candidate"
        existing = [
            item
            for item in task_block.get("candidates", [])
            if item.get("model_name") != model_name and item.get("model_id") != model_name
        ]
        existing.append(record)
        task_block["candidates"] = sorted(
            existing,
            key=lambda item: float(item.get("rank_score", 0.0) or 0.0),
            reverse=True,
        )[:MAX_CANDIDATES]

    if metadata:
        record.update({str(key): value for key, value in metadata.items()})

    payload["task"] = task
    payload["schema_version"] = SCHEMA_VERSION_V2
    _sync_legacy_stages(payload, system_id=system_id, model_type=model_type)
    return write_registry(models_root, payload)


def register_candidate(
    models_root: Path,
    *,
    system_id: str,
    model_type: str,
    candidate: dict[str, Any],
    task: str | None = None,
    max_candidates: int = MAX_CANDIDATES,
) -> Path:
    system_id = _normalize_system_id(system_id)
    payload = load_registry(
        models_root,
        task=task or model_type,
        system_id=system_id,
        migrate=True,
        write_back=False,
    )
    payload["schema_version"] = SCHEMA_VERSION_V2
    payload["task"] = task or payload.get("task") or model_type

    task_block = _ensure_system_task(payload, system_id, model_type)
    normalized = _normalize_candidate(candidate, model_type)

    existing = [
        item
        for item in task_block.get("candidates", [])
        if item.get("model_id") != normalized["model_id"]
        and item.get("model_name") != normalized["model_name"]
    ]
    existing.append(normalized)
    task_block["candidates"] = sorted(
        existing,
        key=lambda item: float(item.get("rank_score", 0.0) or 0.0),
        reverse=True,
    )[:max_candidates]

    _sync_legacy_stages(payload, system_id=system_id, model_type=model_type)
    return write_registry(models_root, payload)


def list_candidates(
    models_root: Path,
    *,
    system_id: str,
    model_type: str,
    task: str | None = None,
) -> list[dict[str, Any]]:
    payload = load_registry(models_root, task=task or model_type, system_id=system_id)
    task_block = _ensure_system_task(payload, _normalize_system_id(system_id), model_type)
    return list(task_block.get("candidates", []))


def get_production_model(
    models_root: Path,
    *,
    system_id: str,
    model_type: str,
    task: str | None = None,
) -> dict[str, Any] | None:
    payload = load_registry(models_root, task=task or model_type, system_id=system_id)
    task_block = _ensure_system_task(payload, _normalize_system_id(system_id), model_type)
    production = task_block.get("production")
    return dict(production) if production else None


def get_model_summary(
    models_root: Path,
    *,
    system_id: str,
    model_type: str,
    task: str | None = None,
) -> dict[str, Any]:
    payload = load_registry(models_root, task=task or model_type, system_id=system_id)
    task_block = _ensure_system_task(payload, _normalize_system_id(system_id), model_type)
    return {
        "system_id": _normalize_system_id(system_id),
        "model_type": model_type,
        "production": dict(task_block.get("production")) if task_block.get("production") else None,
        "previous": [dict(item) for item in (task_block.get("previous") or [])],
        "candidates": [dict(item) for item in (task_block.get("candidates") or [])],
    }


def promote_model(
    models_root: Path,
    *,
    system_id: str,
    model_type: str,
    model_id: str | None = None,
    model_name: str | None = None,
    promoted_by: str = "system",
    notes: str = "",
    task: str | None = None,
    keep_previous: int = 5,
) -> Path:
    system_id = _normalize_system_id(system_id)
    payload = load_registry(
        models_root,
        task=task or model_type,
        system_id=system_id,
        migrate=True,
        write_back=False,
    )
    task_block = _ensure_system_task(payload, system_id, model_type)

    candidates = list(task_block.get("candidates", []))
    selected: dict[str, Any] | None = None
    for item in candidates:
        if model_id and item.get("model_id") == model_id:
            selected = item
            break
        if model_name and item.get("model_name") == model_name:
            selected = item
            break

    if selected is None and not model_id and not model_name and candidates:
        selected = candidates[0]

    if selected is None:
        target = model_id or model_name or "<top-candidate>"
        raise ValueError(f"Candidate not found for {system_id}/{model_type}: {target}")

    current = task_block.get("production")
    if current:
        previous = dict(current)
        previous["status"] = "previous"
        previous["archived_at"] = now_iso()
        previous["notes"] = previous.get("notes") or "Auto-copied from production during promotion."
        task_block["previous"] = [previous, *task_block.get("previous", [])][:keep_previous]

    production = dict(selected)
    production["status"] = "production"
    production["promoted_at"] = now_iso()
    production["promoted_by"] = promoted_by
    production["notes"] = notes or f"Promoted from candidate for {system_id}/{model_type}."
    production["updated_at"] = now_iso()
    task_block["production"] = production

    task_block["candidates"] = [
        item
        for item in candidates
        if item.get("model_id") != selected.get("model_id")
        and item.get("model_name") != selected.get("model_name")
    ][:MAX_CANDIDATES]

    payload["schema_version"] = SCHEMA_VERSION_V2
    payload["task"] = task or payload.get("task") or model_type
    _sync_legacy_stages(payload, system_id=system_id, model_type=model_type)
    return write_registry(models_root, payload)


def promote_stage(
    models_root: Path,
    task: str,
    *,
    source_stage: str = "candidate",
    target_stage: str = "production",
    previous_stage: str = "previous",
    system_id: str = DEFAULT_SYSTEM_ID,
) -> Path:
    del target_stage, previous_stage  # v2 uses fixed production/previous semantics.
    model_type = _infer_model_type(task, Path(models_root))
    if source_stage == "candidate":
        return promote_model(
            models_root,
            system_id=system_id,
            model_type=model_type,
            promoted_by="legacy_promote_stage",
            task=task,
        )

    payload = load_registry(Path(models_root), task=task, system_id=system_id, migrate=True, write_back=False)
    task_block = _ensure_system_task(payload, _normalize_system_id(system_id), model_type)
    if source_stage == "production":
        if not task_block.get("production"):
            raise ValueError(f"Stage '{source_stage}' is not defined in {registry_path(Path(models_root))}")
        _sync_legacy_stages(payload, system_id=_normalize_system_id(system_id), model_type=model_type)
        return write_registry(Path(models_root), payload)

    raise ValueError(f"Unsupported legacy source stage for v2 registry: {source_stage}")


def resolve_artifact_dir(
    models_root: Path,
    stage: str = DEFAULT_STAGE,
    *,
    system_id: str | None = None,
    model_type: str | None = None,
    task: str | None = None,
) -> Path:
    models_root = Path(models_root)

    if system_id is not None or model_type is not None:
        if not system_id or not model_type:
            raise ValueError("Both system_id and model_type are required for per-system artifact resolution.")

        registry = load_registry(models_root, task=task or model_type, system_id=system_id)
        task_block = _ensure_system_task(registry, _normalize_system_id(system_id), model_type)

        if stage == "production":
            entry = task_block.get("production")
        elif stage == "candidate":
            candidates = task_block.get("candidates") or []
            entry = candidates[0] if candidates else None
        elif stage == "previous":
            previous = task_block.get("previous") or []
            entry = previous[0] if previous else None
        else:
            entry = None

        if not entry:
            raise FileNotFoundError(
                f"No registry entry found for stage '{stage}' under system '{system_id}' and model_type '{model_type}'."
            )

        artifact_dir = Path(str(entry.get("artifact_dir") or ""))
        if not artifact_dir.is_absolute():
            artifact_dir = models_root / artifact_dir
        if artifact_dir.exists():
            return artifact_dir
        raise FileNotFoundError(
            f"Registry stage '{stage}' for {system_id}/{model_type} points to missing model directory: {artifact_dir}"
        )

    registry = load_registry_v1(models_root)
    stage_entry = registry.get("stages", {}).get(stage, {})
    model_name = stage_entry.get("model_name")
    if model_name:
        artifact_dir = models_root / str(model_name)
        if artifact_dir.exists():
            return artifact_dir
        raise FileNotFoundError(f"Stage '{stage}' points to missing model directory: {artifact_dir}")

    fallback_dir = models_root / stage
    if fallback_dir.exists():
        return fallback_dir

    raise FileNotFoundError(f"Unable to resolve artifact dir for stage '{stage}' under {models_root}")


def resolve_production_artifact_dir(
    models_root: Path,
    *,
    system_id: str,
    model_type: str,
    task: str | None = None,
) -> Path:
    return resolve_artifact_dir(
        models_root,
        stage="production",
        system_id=system_id,
        model_type=model_type,
        task=task or model_type,
    )
