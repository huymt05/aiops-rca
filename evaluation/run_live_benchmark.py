#!/usr/bin/env python3
import argparse
import csv
import json
import statistics
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


CSV_FIELDS = [
    "timestamp_utc",
    "system_id",
    "run_index",
    "success",
    "http_status",
    "duration_ms",
    "pipeline_state",
    "event_id",
    "anomaly_score",
    "threshold",
    "is_anomaly",
    "anomaly_model",
    "rca_model",
    "top1_service",
    "top1_score",
    "top3_services",
    "error",
]


def percentile(values, quantile):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def rounded(value):
    return None if value is None else round(float(value), 3)


def post_json(url, payload, timeout):
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
            status = response.status
        error = ""
    except urllib.error.HTTPError as exc:
        status = exc.code
        data = None
        error = exc.read().decode("utf-8", errors="replace")
    except Exception as exc:
        status = None
        data = None
        error = str(exc)

    duration_ms = (time.perf_counter() - started) * 1000
    return status, data, error, duration_ms


def build_record(system_id, run_index, status, data, error, duration_ms):
    anomaly = (data or {}).get("anomaly") or {}
    rca = (data or {}).get("rca") or {}
    top1 = rca.get("top1") or {}
    topk = rca.get("topk") or []

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "system_id": system_id,
        "run_index": run_index,
        "success": bool(data is not None and status and 200 <= status < 300),
        "http_status": status,
        "duration_ms": rounded(duration_ms),
        "pipeline_state": (data or {}).get("pipeline_state", ""),
        "event_id": (data or {}).get("event_id", ""),
        "anomaly_score": anomaly.get("anomaly_score"),
        "threshold": anomaly.get("threshold"),
        "is_anomaly": anomaly.get("is_anomaly"),
        "anomaly_model": anomaly.get("model_name", ""),
        "rca_model": rca.get("model_name", ""),
        "top1_service": top1.get("service_name", ""),
        "top1_score": top1.get("score"),
        "top3_services": "|".join(
            str(item.get("service_name", ""))
            for item in topk[:3]
            if item.get("service_name")
        ),
        "error": error,
        "response": data,
    }


def summarize(records, systems):
    summaries = []

    for system_id in systems:
        rows = [row for row in records if row["system_id"] == system_id]
        successful = [row for row in rows if row["success"]]
        durations = [float(row["duration_ms"]) for row in successful]
        anomaly_runs = [row for row in successful if row["is_anomaly"] is True]
        rca_runs = [row for row in successful if row["top1_service"]]

        summaries.append(
            {
                "system_id": system_id,
                "runs": len(rows),
                "successful_runs": len(successful),
                "failed_runs": len(rows) - len(successful),
                "success_rate": rounded(
                    len(successful) / len(rows) if rows else 0
                ),
                "mean_ms": rounded(
                    statistics.fmean(durations) if durations else None
                ),
                "p50_ms": rounded(percentile(durations, 0.50)),
                "p95_ms": rounded(percentile(durations, 0.95)),
                "anomaly_runs": len(anomaly_runs),
                "rca_runs": len(rca_runs),
            }
        )

    return summaries


def write_outputs(output_dir, records, summaries):
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    csv_path = output_dir / f"live_benchmark_{stamp}.csv"
    json_path = output_dir / f"live_benchmark_{stamp}.json"
    summary_path = output_dir / f"live_benchmark_summary_{stamp}.json"
    markdown_path = output_dir / f"live_benchmark_summary_{stamp}.md"

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key) for key in CSV_FIELDS})

    json_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary_path.write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# Live Evaluation Benchmark",
        "",
        "| System | Runs | Success | Success rate | Mean ms | P50 ms | P95 ms | Anomaly runs | RCA runs |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summaries:
        lines.append(
            f"| {item['system_id']} | {item['runs']} | "
            f"{item['successful_runs']} | {item['success_rate']:.3f} | "
            f"{item['mean_ms']} | {item['p50_ms']} | {item['p95_ms']} | "
            f"{item['anomaly_runs']} | {item['rca_runs']} |"
        )

    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return csv_path, json_path, summary_path, markdown_path


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark repeated calls to the dashboard Live Analyze API."
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8010",
        help="Dashboard base URL.",
    )
    parser.add_argument(
        "--system",
        action="append",
        required=True,
        help="System ID. Repeat this option for multiple systems.",
    )
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--lookback-minutes", type=int, default=1)
    parser.add_argument("--query-limit", type=int, default=150)
    parser.add_argument("--source-service", default="all")
    parser.add_argument("--model-key", default="rf_ml_ranker")
    parser.add_argument(
        "--force-rca",
        action="store_true",
        help="Run RCA even when anomaly score is below threshold.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluation/results"),
    )
    args = parser.parse_args()

    if args.runs < 1:
        parser.error("--runs must be at least 1")

    endpoint = args.base_url.rstrip("/") + "/api/live/analyze"
    records = []

    for system_id in args.system:
        for run_index in range(1, args.runs + 1):
            payload = {
                "system_id": system_id,
                "source_service": args.source_service,
                "lookback_minutes": args.lookback_minutes,
                "query_limit": args.query_limit,
                "run_rca_on_any_input": args.force_rca,
                "model_key": args.model_key,
            }

            status, data, error, duration_ms = post_json(
                endpoint,
                payload,
                args.timeout,
            )
            record = build_record(
                system_id,
                run_index,
                status,
                data,
                error,
                duration_ms,
            )
            records.append(record)

            state = record["pipeline_state"] or "error"
            print(
                f"[{system_id}] {run_index}/{args.runs} "
                f"status={status} duration_ms={record['duration_ms']} "
                f"state={state} top1={record['top1_service'] or '-'}"
            )

            if args.interval > 0:
                time.sleep(args.interval)

    summaries = summarize(records, args.system)
    paths = write_outputs(args.output_dir, records, summaries)

    print("\nOutputs:")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
