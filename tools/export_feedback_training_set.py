from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aiops_framework.dashboard.store import DB_PATH, JSON_PATH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export dashboard feedback and monitoring events into a retraining snapshot."
    )
    parser.add_argument("--db-path", type=Path, default=DB_PATH)
    parser.add_argument("--json-path", type=Path, default=JSON_PATH)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/feedback/feedback_training_snapshot.jsonl"),
    )
    parser.add_argument("--system-id", default="", help="Optional system filter, e.g. online-boutique")
    parser.add_argument("--include-unknown", action="store_true")
    return parser.parse_args()


def _json_load(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def _normalize_feedback_label(value: str) -> str:
    return str(value or "").strip().lower()


def _snapshot_record(row: dict[str, Any]) -> dict[str, Any] | None:
    feedback = _normalize_feedback_label(row["feedback"])
    anomaly_relabel: int | None
    rca_relabel_service: str | None
    needs_human_review = False

    if feedback == "accepted_incident":
        anomaly_relabel = 1
        rca_relabel_service = row.get("rca_top1_service")
    elif feedback == "rejected_false_positive":
        anomaly_relabel = 0
        rca_relabel_service = None
    elif feedback == "unknown":
        anomaly_relabel = None
        rca_relabel_service = None
        needs_human_review = True
    else:
        return None

    event_payload = _json_load(row.get("event_payload_json"), {})
    feedback_payload = _json_load(row.get("feedback_payload_json"), {})
    return {
        "feedback_id": int(row["feedback_id"]),
        "event_id": int(row["event_id"]),
        "system_id": row["system_id"],
        "source_service": row["source_service"],
        "event_created_at": row["event_created_at"],
        "feedback_created_at": row["feedback_created_at"],
        "feedback": feedback,
        "actor": row["actor"],
        "notes": row["notes"] or "",
        "status": row["status"],
        "is_anomaly": bool(row["is_anomaly"]),
        "anomaly_score": row["anomaly_score"],
        "anomaly_model": row["anomaly_model"],
        "rca_top1_service": row["rca_top1_service"],
        "rca_top1_score": row["rca_top1_score"],
        "recommendation_action": row["recommendation_action"],
        "anomaly_relabel": anomaly_relabel,
        "rca_relabel_service": rca_relabel_service,
        "needs_human_review": needs_human_review,
        "event_payload": event_payload,
        "feedback_payload": feedback_payload,
    }


def load_from_sqlite(db_path: Path, system_id: str, include_unknown: bool) -> list[dict[str, Any]]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        query = """
            SELECT
              f.id AS feedback_id,
              f.event_id AS event_id,
              f.created_at AS feedback_created_at,
              f.feedback AS feedback,
              f.actor AS actor,
              f.notes AS notes,
              f.payload_json AS feedback_payload_json,
              e.created_at AS event_created_at,
              e.system_id AS system_id,
              e.source_service AS source_service,
              e.status AS status,
              e.is_anomaly AS is_anomaly,
              e.anomaly_score AS anomaly_score,
              e.anomaly_model AS anomaly_model,
              e.rca_top1_service AS rca_top1_service,
              e.rca_top1_score AS rca_top1_score,
              e.recommendation_action AS recommendation_action,
              e.payload_json AS event_payload_json
            FROM feedback f
            JOIN monitoring_events e ON e.id = f.event_id
        """
        params: list[Any] = []
        clauses: list[str] = []
        if system_id:
            clauses.append("e.system_id = ?")
            params.append(system_id)
        if not include_unknown:
            clauses.append("f.feedback != ?")
            params.append("unknown")
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY f.id ASC"
        rows = conn.execute(query, tuple(params)).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def load_from_json(json_path: Path, system_id: str, include_unknown: bool) -> list[dict[str, Any]]:
    state = _json_load(json_path.read_text(encoding="utf-8"), {}) if json_path.exists() else {}
    events = {int(item["id"]): item for item in state.get("monitoring_events", [])}
    rows: list[dict[str, Any]] = []
    for item in state.get("feedback", []):
        if not include_unknown and _normalize_feedback_label(item.get("feedback")) == "unknown":
            continue
        event = events.get(int(item["event_id"]))
        if event is None:
            continue
        if system_id and event.get("system_id") != system_id:
            continue
        rows.append(
            {
                "feedback_id": item["id"],
                "event_id": item["event_id"],
                "feedback_created_at": item["created_at"],
                "feedback": item["feedback"],
                "actor": item.get("actor", "operator"),
                "notes": item.get("notes", ""),
                "feedback_payload_json": json.dumps(item.get("payload", {}), ensure_ascii=False),
                "event_created_at": event.get("created_at"),
                "system_id": event.get("system_id"),
                "source_service": event.get("source_service"),
                "status": event.get("status"),
                "is_anomaly": int(bool(event.get("is_anomaly"))),
                "anomaly_score": event.get("anomaly_score"),
                "anomaly_model": event.get("anomaly_model"),
                "rca_top1_service": event.get("rca_top1_service"),
                "rca_top1_score": event.get("rca_top1_score"),
                "recommendation_action": event.get("recommendation_action"),
                "event_payload_json": json.dumps(event.get("payload", {}), ensure_ascii=False),
            }
        )
    return rows


def build_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_feedback = Counter(record["feedback"] for record in records)
    by_system: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        by_system[str(record["system_id"])][str(record["feedback"])] += 1
    return {
        "records": len(records),
        "by_feedback": dict(by_feedback),
        "by_system": {system_id: dict(counter) for system_id, counter in by_system.items()},
    }


def main() -> None:
    args = parse_args()
    system_id = args.system_id.strip()

    if args.db_path.exists():
        rows = load_from_sqlite(args.db_path, system_id, args.include_unknown)
        source = str(args.db_path)
    elif args.json_path.exists():
        rows = load_from_json(args.json_path, system_id, args.include_unknown)
        source = str(args.json_path)
    else:
        raise FileNotFoundError(
            f"Neither dashboard DB nor JSON fallback store was found. Checked: {args.db_path} and {args.json_path}"
        )

    records = [record for row in rows if (record := _snapshot_record(row)) is not None]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")

    summary = build_summary(records)
    summary["source"] = source
    summary["output"] = str(args.output)
    summary_path = args.output.with_name(args.output.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")

    print(f"Exported {len(records)} feedback snapshot rows to {args.output}")
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
