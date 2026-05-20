from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trigger an Airflow DAG using a JSON conf file.")
    parser.add_argument("--dag-id", required=True)
    parser.add_argument("--logical-date", required=True)
    parser.add_argument("--conf-file", type=Path, required=True)
    parser.add_argument("--airflow-bin", default="/home/airflow/.local/bin/airflow")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    conf = json.loads(args.conf_file.read_text(encoding="utf-8"))
    cmd = [
        args.airflow_bin,
        "dags",
        "trigger",
        args.dag_id,
        "--logical-date",
        args.logical_date,
        "--conf",
        json.dumps(conf),
    ]
    completed = subprocess.run(cmd, check=False, capture_output=True, text=True)
    print(completed.stdout, end="")
    if completed.returncode != 0:
        raise SystemExit(completed.stderr or f"airflow trigger failed with exit code {completed.returncode}")


if __name__ == "__main__":
    main()
