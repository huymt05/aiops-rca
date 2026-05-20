from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aiops_framework.dashboard.store import DB_PATH, JSON_PATH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a paper-ready incident evaluation table from dashboard monitoring events and feedback."
    )
    parser.add_argument("--db-path", type=Path, default=DB_PATH)
    parser.add_argument("--json-path", type=Path, default=JSON_PATH)
    parser.add_argument("--system-id", default="", help="Optional system filter.")
    parser.add_argument(
        "--annotations",
        type=Path,
        default=None,
        help="Optional JSON/JSONL file with ground-truth annotations keyed by event_id.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/evaluation"),
        help="Directory for CSV/JSON/Markdown outputs.",
    )
    return parser.parse_args()


def _json_load(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def _normalize_text(value: Any) -> str:
    return str(value or "").strip()


def _normalize_bool(value: Any) -> bool | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "anomaly"}:
        return True
    if text in {"0", "false", "no", "n", "normal"}:
        return False
    return None


def load_annotations(path: Path | None) -> dict[int, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return {}
    records: list[dict[str, Any]]
    if path.suffix.lower() == ".jsonl":
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        payload = json.loads(text)
        if isinstance(payload, dict) and "items" in payload:
            payload = payload["items"]
        if isinstance(payload, dict):
            records = [dict({"event_id": key}, **value) for key, value in payload.items()]
        else:
            records = list(payload)
    annotations: dict[int, dict[str, Any]] = {}
    for item in records:
        try:
            event_id = int(item["event_id"])
        except (KeyError, TypeError, ValueError):
            continue
        annotations[event_id] = dict(item)
    return annotations


def load_event_rows(db_path: Path, json_path: Path, system_id: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if db_path.exists():
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        try:
            query = """
                SELECT
                  e.id,
                  e.created_at,
                  e.system_id,
                  e.source_service,
                  e.status,
                  e.is_anomaly,
                  e.anomaly_score,
                  e.anomaly_model,
                  e.rca_top1_service,
                  e.rca_top1_score,
                  e.recommendation_action,
                  e.payload_json,
                  f.feedback,
                  f.actor AS feedback_actor,
                  f.notes AS feedback_notes,
                  f.payload_json AS feedback_payload_json
                FROM monitoring_events e
                LEFT JOIN feedback f ON f.event_id = e.id
            """
            params: list[Any] = []
            if system_id:
                query += " WHERE e.system_id = ?"
                params.append(system_id)
            query += " ORDER BY e.id ASC"
            rows = [dict(row) for row in conn.execute(query, tuple(params)).fetchall()]
        finally:
            conn.close()
        return rows

    if json_path.exists():
        state = _json_load(json_path.read_text(encoding="utf-8"), {})
        feedback_by_event: dict[int, dict[str, Any]] = {}
        for item in state.get("feedback", []):
            feedback_by_event[int(item["event_id"])] = item
        for event in state.get("monitoring_events", []):
            if system_id and event.get("system_id") != system_id:
                continue
            feedback = feedback_by_event.get(int(event["id"]), {})
            rows.append(
                {
                    "id": event["id"],
                    "created_at": event.get("created_at", ""),
                    "system_id": event.get("system_id", ""),
                    "source_service": event.get("source_service", ""),
                    "status": event.get("status", ""),
                    "is_anomaly": int(bool(event.get("is_anomaly"))),
                    "anomaly_score": event.get("anomaly_score"),
                    "anomaly_model": event.get("anomaly_model"),
                    "rca_top1_service": event.get("rca_top1_service"),
                    "rca_top1_score": event.get("rca_top1_score"),
                    "recommendation_action": event.get("recommendation_action"),
                    "payload_json": json.dumps(event.get("payload", {}), ensure_ascii=False),
                    "feedback": feedback.get("feedback", ""),
                    "feedback_actor": feedback.get("actor", ""),
                    "feedback_notes": feedback.get("notes", ""),
                    "feedback_payload_json": json.dumps(feedback.get("payload", {}), ensure_ascii=False),
                }
            )
        return rows

    raise FileNotFoundError(f"Neither dashboard DB nor JSON store found. Checked: {db_path} and {json_path}")


def _infer_expected_root_cause(annotation: dict[str, Any], event_payload: dict[str, Any]) -> str:
    explicit = _normalize_text(annotation.get("expected_root_cause") or annotation.get("true_root_cause"))
    if explicit:
        return explicit
    for key in ("true_root_cause", "root_cause_service", "fault_target_service"):
        value = _normalize_text(event_payload.get(key))
        if value:
            return value
    live = event_payload.get("live_context") or {}
    for key in ("true_root_cause", "root_cause_service", "fault_target_service"):
        value = _normalize_text(live.get(key))
        if value:
            return value
    return ""


def _infer_expected_anomaly(annotation: dict[str, Any], feedback: str, event_payload: dict[str, Any]) -> bool | None:
    explicit = _normalize_bool(annotation.get("expected_is_anomaly"))
    if explicit is not None:
        return explicit
    feedback_norm = _normalize_text(feedback).lower()
    if feedback_norm == "accepted_incident":
        return True
    if feedback_norm == "rejected_false_positive":
        return False
    for key in ("expected_is_anomaly", "is_anomaly"):
        inferred = _normalize_bool(event_payload.get(key))
        if inferred is not None:
            return inferred
    return None


def build_rows(rows: list[dict[str, Any]], annotations: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for row in rows:
        event_id = int(row["id"])
        event_payload = _json_load(row.get("payload_json"), {})
        feedback_payload = _json_load(row.get("feedback_payload_json"), {})
        annotation = annotations.get(event_id, {})

        expected_root_cause = _infer_expected_root_cause(annotation, event_payload)
        expected_is_anomaly = _infer_expected_anomaly(annotation, row.get("feedback", ""), event_payload)
        predicted_root_cause = _normalize_text(row.get("rca_top1_service"))
        predicted_is_anomaly = bool(row.get("is_anomaly"))

        topk_services: list[str] = []
        try:
            topk = (((event_payload.get("rca") or {}).get("topk")) or [])
            topk_services = [_normalize_text(item.get("service_name")) for item in topk if _normalize_text(item.get("service_name"))]
        except Exception:
            topk_services = []

        root_cause_top1_correct = None
        root_cause_top3_hit = None
        if expected_root_cause:
            root_cause_top1_correct = predicted_root_cause == expected_root_cause
            root_cause_top3_hit = expected_root_cause in topk_services if topk_services else predicted_root_cause == expected_root_cause

        anomaly_correct = None if expected_is_anomaly is None else bool(predicted_is_anomaly == expected_is_anomaly)

        items.append(
            {
                "event_id": event_id,
                "created_at": row.get("created_at", ""),
                "system_id": _normalize_text(row.get("system_id")),
                "source_service": _normalize_text(row.get("source_service")),
                "incident_label": _normalize_text(annotation.get("incident_label") or event_payload.get("fault_family") or ""),
                "expected_is_anomaly": expected_is_anomaly,
                "predicted_is_anomaly": predicted_is_anomaly,
                "anomaly_correct": anomaly_correct,
                "anomaly_score": row.get("anomaly_score"),
                "anomaly_model": _normalize_text(row.get("anomaly_model")),
                "expected_root_cause": expected_root_cause,
                "predicted_root_cause": predicted_root_cause,
                "root_cause_top1_correct": root_cause_top1_correct,
                "root_cause_top3_hit": root_cause_top3_hit,
                "rca_top1_score": row.get("rca_top1_score"),
                "top3_services": topk_services,
                "recommendation_action": _normalize_text(row.get("recommendation_action")),
                "feedback": _normalize_text(row.get("feedback")),
                "feedback_actor": _normalize_text(row.get("feedback_actor")),
                "feedback_notes": _normalize_text(row.get("feedback_notes")),
                "annotation_payload": annotation,
                "event_payload": event_payload,
                "feedback_payload": feedback_payload,
            }
        )
    return items


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator


def build_summary(items: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(items)
    anomaly_labeled = [item for item in items if item["anomaly_correct"] is not None]
    rca_labeled = [item for item in items if item["root_cause_top1_correct"] is not None]
    top3_labeled = [item for item in items if item["root_cause_top3_hit"] is not None]

    summary = {
        "records": total,
        "systems": dict(Counter(item["system_id"] for item in items)),
        "feedback": dict(Counter(item["feedback"] or "none" for item in items)),
        "anomaly_accuracy": _rate(sum(1 for item in anomaly_labeled if item["anomaly_correct"]), len(anomaly_labeled)),
        "rca_top1_accuracy": _rate(sum(1 for item in rca_labeled if item["root_cause_top1_correct"]), len(rca_labeled)),
        "rca_top3_hit_rate": _rate(sum(1 for item in top3_labeled if item["root_cause_top3_hit"]), len(top3_labeled)),
        "annotated_for_anomaly": len(anomaly_labeled),
        "annotated_for_rca_top1": len(rca_labeled),
        "annotated_for_rca_top3": len(top3_labeled),
    }
    return summary


def write_outputs(output_dir: Path, items: list[dict[str, Any]], summary: dict[str, Any]) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "incident_evaluation_table.csv"
    json_path = output_dir / "incident_evaluation_table.json"
    summary_path = output_dir / "incident_evaluation_summary.json"
    markdown_path = output_dir / "incident_evaluation_summary.md"

    csv_fields = [
        "event_id",
        "created_at",
        "system_id",
        "source_service",
        "incident_label",
        "expected_is_anomaly",
        "predicted_is_anomaly",
        "anomaly_correct",
        "anomaly_score",
        "anomaly_model",
        "expected_root_cause",
        "predicted_root_cause",
        "root_cause_top1_correct",
        "root_cause_top3_hit",
        "rca_top1_score",
        "top3_services",
        "recommendation_action",
        "feedback",
        "feedback_actor",
        "feedback_notes",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for item in items:
            row = dict(item)
            row["top3_services"] = ", ".join(item["top3_services"])
            writer.writerow({key: row.get(key) for key in csv_fields})

    json_path.write_text(json.dumps(items, indent=2, ensure_ascii=False, sort_keys=True, default=str), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")

    markdown_lines = [
        "# Incident Evaluation Summary",
        "",
        f"- Records: **{summary['records']}**",
        f"- Anomaly accuracy: **{summary['anomaly_accuracy'] if summary['anomaly_accuracy'] is not None else 'n/a'}**",
        f"- RCA Top-1 accuracy: **{summary['rca_top1_accuracy'] if summary['rca_top1_accuracy'] is not None else 'n/a'}**",
        f"- RCA Top-3 hit rate: **{summary['rca_top3_hit_rate'] if summary['rca_top3_hit_rate'] is not None else 'n/a'}**",
        "",
        "| Event | System | Source | Predicted anomaly | Predicted RCA | Feedback |",
        "|---|---|---|---:|---|---|",
    ]
    for item in items[:20]:
        markdown_lines.append(
            f"| {item['event_id']} | {item['system_id']} | {item['source_service']} | "
            f"{item['predicted_is_anomaly']} | {item['predicted_root_cause'] or '-'} | {item['feedback'] or '-'} |"
        )
    markdown_path.write_text("\n".join(markdown_lines), encoding="utf-8")

    return {
        "csv": str(csv_path),
        "json": str(json_path),
        "summary_json": str(summary_path),
        "summary_md": str(markdown_path),
    }


def main() -> None:
    args = parse_args()
    rows = load_event_rows(args.db_path, args.json_path, args.system_id.strip())
    annotations = load_annotations(args.annotations)
    items = build_rows(rows, annotations)
    summary = build_summary(items)
    outputs = write_outputs(args.output_dir, items, summary)
    print(json.dumps({"summary": summary, "outputs": outputs}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
