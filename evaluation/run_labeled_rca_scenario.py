#!/usr/bin/env python3
"""Run a controlled Kubernetes fault and evaluate live RCA against ground truth."""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


CSV_FIELDS = [
    "timestamp_utc",
    "scenario_id",
    "system_id",
    "namespace",
    "deployment",
    "expected_root_cause",
    "run_index",
    "success",
    "http_status",
    "duration_ms",
    "pipeline_state",
    "anomaly_score",
    "threshold",
    "predicted_is_anomaly",
    "anomaly_correct",
    "top1_service",
    "top1_score",
    "top3_services",
    "top1_correct",
    "top3_hit",
    "error",
]


def rounded(value: Any) -> float | None:
    return None if value is None else round(float(value), 3)


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def normalize_service(value: Any) -> str:
    return str(value or "").strip().lower().replace("_", "-")


def post_json(url: str, payload: dict[str, Any], timeout: float) -> tuple[int | None, dict[str, Any] | None, str, float]:
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
    except Exception as exc:  # Network errors are recorded in the evaluation output.
        status = None
        data = None
        error = str(exc)
    return status, data, error, (time.perf_counter() - started) * 1000


def run_command(command: list[str], timeout: float | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


class ScaleToZeroFault:
    """Scale one Deployment to zero and always restore its original replica count."""

    def __init__(
        self,
        kubectl_command: list[str],
        namespace: str,
        deployment: str,
        rollout_timeout: int,
        runner: Callable[..., subprocess.CompletedProcess[str]] = run_command,
    ) -> None:
        self.kubectl_command = kubectl_command
        self.namespace = namespace
        self.deployment = deployment
        self.rollout_timeout = rollout_timeout
        self.runner = runner
        self.original_replicas: int | None = None

    def _kubectl(self, *args: str, timeout: float | None = None) -> subprocess.CompletedProcess[str]:
        return self.runner(
            [*self.kubectl_command, "-n", self.namespace, *args],
            timeout=timeout,
        )

    def __enter__(self) -> "ScaleToZeroFault":
        result = self._kubectl(
            "get",
            "deployment",
            self.deployment,
            "-o",
            "jsonpath={.spec.replicas}",
        )
        try:
            self.original_replicas = int(result.stdout.strip())
        except ValueError as exc:
            raise RuntimeError(
                f"Could not read replicas for deployment/{self.deployment}: {result.stdout!r}"
            ) from exc
        if self.original_replicas < 1:
            raise RuntimeError(
                f"deployment/{self.deployment} already has {self.original_replicas} replicas; refusing to inject the fault"
            )
        self._kubectl("scale", f"deployment/{self.deployment}", "--replicas=0")
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        if self.original_replicas is None:
            return False
        print(
            f"Restoring deployment/{self.deployment} to {self.original_replicas} replica(s)...",
            file=sys.stderr,
        )
        self._kubectl(
            "scale",
            f"deployment/{self.deployment}",
            f"--replicas={self.original_replicas}",
        )
        self._kubectl(
            "rollout",
            "status",
            f"deployment/{self.deployment}",
            f"--timeout={self.rollout_timeout}s",
            timeout=self.rollout_timeout + 15,
        )
        return False


def generate_get_traffic(url: str, requests: int, interval: float, timeout: float) -> dict[str, int]:
    counts = {"attempted": 0, "successful": 0, "failed": 0}
    if not url or requests <= 0:
        return counts
    for _ in range(requests):
        counts["attempted"] += 1
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                response.read(1)
                if 200 <= response.status < 400:
                    counts["successful"] += 1
                else:
                    counts["failed"] += 1
        except Exception:
            counts["failed"] += 1
        if interval > 0:
            time.sleep(interval)
    return counts


def build_record(
    args: argparse.Namespace,
    run_index: int,
    status: int | None,
    data: dict[str, Any] | None,
    error: str,
    duration_ms: float,
) -> dict[str, Any]:
    anomaly = (data or {}).get("anomaly") or {}
    rca = (data or {}).get("rca") or {}
    top1 = rca.get("top1") or {}
    topk = rca.get("topk") or []
    top3 = [str(item.get("service_name", "")).strip() for item in topk[:3] if item.get("service_name")]
    expected = normalize_service(args.expected_root_cause)
    predicted = normalize_service(top1.get("service_name"))
    normalized_top3 = [normalize_service(item) for item in top3]
    predicted_is_anomaly = anomaly.get("is_anomaly")

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "scenario_id": args.scenario_id,
        "system_id": args.system_id,
        "namespace": args.namespace,
        "deployment": args.deployment,
        "expected_root_cause": args.expected_root_cause,
        "run_index": run_index,
        "success": bool(data is not None and status and 200 <= status < 300),
        "http_status": status,
        "duration_ms": rounded(duration_ms),
        "pipeline_state": (data or {}).get("pipeline_state", ""),
        "anomaly_score": anomaly.get("anomaly_score"),
        "threshold": anomaly.get("threshold"),
        "predicted_is_anomaly": predicted_is_anomaly,
        "anomaly_correct": predicted_is_anomaly is True,
        "top1_service": top1.get("service_name", ""),
        "top1_score": top1.get("score"),
        "top3_services": "|".join(top3),
        "top1_correct": bool(predicted and predicted == expected),
        "top3_hit": expected in normalized_top3,
        "error": error,
        "response": data,
    }


def summarize(records: list[dict[str, Any]], traffic: dict[str, int]) -> dict[str, Any]:
    successful = [row for row in records if row["success"]]
    durations = [float(row["duration_ms"]) for row in successful]
    count = len(records)
    return {
        "scenario_id": records[0]["scenario_id"] if records else "",
        "system_id": records[0]["system_id"] if records else "",
        "expected_root_cause": records[0]["expected_root_cause"] if records else "",
        "runs": count,
        "successful_runs": len(successful),
        "success_rate": rounded(len(successful) / count if count else 0),
        "anomaly_detection_rate": rounded(
            sum(row["predicted_is_anomaly"] is True for row in records) / count if count else 0
        ),
        "rca_top1_accuracy": rounded(sum(row["top1_correct"] for row in records) / count if count else 0),
        "rca_top3_hit_rate": rounded(sum(row["top3_hit"] for row in records) / count if count else 0),
        "mean_ms": rounded(statistics.fmean(durations) if durations else None),
        "p50_ms": rounded(percentile(durations, 0.50)),
        "p95_ms": rounded(percentile(durations, 0.95)),
        "traffic": traffic,
    }


def write_outputs(output_dir: Path, records: list[dict[str, Any]], summary: dict[str, Any]) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"labeled_rca_{summary['scenario_id']}_{stamp}"
    csv_path = output_dir / f"{prefix}.csv"
    json_path = output_dir / f"{prefix}.json"
    summary_json = output_dir / f"{prefix}_summary.json"
    summary_md = output_dir / f"{prefix}_summary.md"

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key) for key in CSV_FIELDS})
    json_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Labeled RCA Scenario",
        "",
        f"- Scenario: **{summary['scenario_id']}**",
        f"- System: **{summary['system_id']}**",
        f"- Ground-truth root cause: **{summary['expected_root_cause']}**",
        f"- Successful runs: **{summary['successful_runs']}/{summary['runs']}**",
        f"- Anomaly detection rate: **{summary['anomaly_detection_rate']}**",
        f"- RCA Top-1 accuracy: **{summary['rca_top1_accuracy']}**",
        f"- RCA Top-3 hit rate: **{summary['rca_top3_hit_rate']}**",
        f"- Mean latency: **{summary['mean_ms']} ms**",
        f"- P95 latency: **{summary['p95_ms']} ms**",
        "",
        "| Run | Anomaly | Top-1 | Top-1 correct | Top-3 | Top-3 hit | Duration ms |",
        "|---:|---|---|---|---|---|---:|",
    ]
    for row in records:
        lines.append(
            f"| {row['run_index']} | {row['predicted_is_anomaly']} | {row['top1_service'] or '-'} | "
            f"{row['top1_correct']} | {row['top3_services'] or '-'} | {row['top3_hit']} | {row['duration_ms']} |"
        )
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return [csv_path, json_path, summary_json, summary_md]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inject a reversible scale-to-zero fault and evaluate live RCA against its known root cause."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8010")
    parser.add_argument("--system-id", required=True)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--deployment", required=True)
    parser.add_argument("--expected-root-cause", required=True)
    parser.add_argument("--scenario-id", required=True)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--fault-settle-seconds", type=float, default=20.0)
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--rollout-timeout", type=int, default=180)
    parser.add_argument("--lookback-minutes", type=int, default=2)
    parser.add_argument("--query-limit", type=int, default=300)
    parser.add_argument("--source-service", default="all")
    parser.add_argument("--model-key", default="rf_ml_ranker")
    parser.add_argument("--traffic-url", default="")
    parser.add_argument("--traffic-requests", type=int, default=20)
    parser.add_argument("--traffic-interval", type=float, default=0.25)
    parser.add_argument("--traffic-timeout", type=float, default=5.0)
    parser.add_argument(
        "--kubectl-command",
        default="kubectl",
        help='Kubectl command, for example "sudo k3s kubectl".',
    )
    parser.add_argument(
        "--force-rca",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run RCA even if anomaly detection misses the injected incident (default: true).",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation/results/labeled-rca"))
    parser.add_argument(
        "--confirm-disruption",
        action="store_true",
        help="Required acknowledgement that the selected Deployment will temporarily scale to zero.",
    )
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be at least 1")
    if not args.confirm_disruption:
        parser.error("--confirm-disruption is required because this command temporarily scales a Deployment to zero")
    return args


def main() -> None:
    args = parse_args()
    endpoint = args.base_url.rstrip("/") + "/api/live/analyze"
    payload = {
        "system_id": args.system_id,
        "source_service": args.source_service,
        "lookback_minutes": args.lookback_minutes,
        "query_limit": args.query_limit,
        "run_rca_on_any_input": args.force_rca,
        "model_key": args.model_key,
    }
    records: list[dict[str, Any]] = []
    traffic = {"attempted": 0, "successful": 0, "failed": 0}

    fault = ScaleToZeroFault(
        shlex.split(args.kubectl_command),
        args.namespace,
        args.deployment,
        args.rollout_timeout,
    )
    with fault:
        print(f"Injected scale-to-zero fault into {args.namespace}/deployment/{args.deployment}.")
        if args.fault_settle_seconds > 0:
            time.sleep(args.fault_settle_seconds)
        traffic = generate_get_traffic(
            args.traffic_url,
            args.traffic_requests,
            args.traffic_interval,
            args.traffic_timeout,
        )
        for run_index in range(1, args.runs + 1):
            status, data, error, duration_ms = post_json(endpoint, payload, args.timeout)
            record = build_record(args, run_index, status, data, error, duration_ms)
            records.append(record)
            print(
                f"[{run_index}/{args.runs}] status={status} anomaly={record['predicted_is_anomaly']} "
                f"top1={record['top1_service'] or '-'} top1_correct={record['top1_correct']} "
                f"top3_hit={record['top3_hit']}"
            )
            if run_index < args.runs and args.interval > 0:
                time.sleep(args.interval)

    summary = summarize(records, traffic)
    paths = write_outputs(args.output_dir, records, summary)
    print("\nSummary:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\nOutputs:")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
