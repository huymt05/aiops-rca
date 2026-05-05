from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aiops_framework.inference.anomaly_service.model_loader import load_artifacts
from aiops_framework.inference.anomaly_service.predictor import predict_scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Analyze anomaly_score distribution and calibrate runtime threshold.')
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--artifact-dir', type=Path, required=True)
    parser.add_argument('--feature-file', type=Path, default=None)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--threshold-min', type=float, default=0.05)
    parser.add_argument('--threshold-max', type=float, default=0.95)
    parser.add_argument('--threshold-steps', type=int, default=91)
    parser.add_argument('--specificity-floor', type=float, default=0.25)
    parser.add_argument('--target-recall', type=float, default=0.85)
    return parser.parse_args()


def read_run_ids(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def sanitize_sample_class(df: pd.DataFrame) -> pd.DataFrame:
    if 'sample_class' in df.columns:
        df = df.copy()
        fill_values = []
        for _, row in df.iterrows():
            sample_class = row.get('sample_class')
            if pd.isna(sample_class) or str(sample_class).strip().lower() in {'', 'nan', 'none'}:
                fill_values.append('fault' if int(row.get('is_anomaly', 0)) == 1 else 'normal')
            else:
                fill_values.append(str(sample_class).strip().lower())
        df['sample_class'] = fill_values
    else:
        df = df.copy()
        df['sample_class'] = np.where(df.get('is_anomaly', 0).astype(int) == 1, 'fault', 'normal')
    return df


def compute_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (scores >= threshold).astype(int)
    tp = int(((y_true == 1) & (pred == 1)).sum())
    tn = int(((y_true == 0) & (pred == 0)).sum())
    fp = int(((y_true == 0) & (pred == 1)).sum())
    fn = int(((y_true == 1) & (pred == 0)).sum())
    specificity = float(tn / (tn + fp)) if (tn + fp) else 0.0
    return {
        'threshold': float(threshold),
        'precision_anomaly': float(precision_score(y_true, pred, zero_division=0)),
        'recall_anomaly': float(recall_score(y_true, pred, zero_division=0)),
        'specificity': specificity,
        'balanced_accuracy': float(balanced_accuracy_score(y_true, pred)),
        'f1_anomaly': float(f1_score(y_true, pred, zero_division=0)),
        'trigger_rate': float(pred.mean()),
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
    }


def pick_runtime_threshold(rows: list[dict[str, float]], target_recall: float, specificity_floor: float) -> dict[str, float]:
    eligible = [row for row in rows if row['recall_anomaly'] >= target_recall and row['specificity'] >= specificity_floor]
    if eligible:
        best = max(eligible, key=lambda row: (row['balanced_accuracy'], row['f1_anomaly']))
        return dict(best)
    best = max(
        rows,
        key=lambda row: (
            0.60 * row['recall_anomaly']
            + 0.20 * row['balanced_accuracy']
            + 0.10 * row['specificity']
            + 0.10 * row['precision_anomaly']
        ),
    )
    return dict(best)


def label_percentiles(scores: np.ndarray) -> dict[str, float]:
    if scores.size == 0:
        return {}
    return {
        'p05': float(np.percentile(scores, 5)),
        'p25': float(np.percentile(scores, 25)),
        'p50': float(np.percentile(scores, 50)),
        'p75': float(np.percentile(scores, 75)),
        'p95': float(np.percentile(scores, 95)),
        'mean': float(np.mean(scores)),
        'std': float(np.std(scores)),
    }


def main() -> None:
    args = parse_args()
    feature_file = args.feature_file or (args.data_root / 'processed' / 'anomaly' / 'window_features_labeled.parquet')
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    artifacts = load_artifacts(args.artifact_dir)
    df = pd.read_parquet(feature_file)
    df = sanitize_sample_class(df)

    train_runs = set(read_run_ids(args.data_root / 'splits' / 'train_runs.txt'))
    val_runs = set(read_run_ids(args.data_root / 'splits' / 'val_runs.txt'))
    test_runs = set(read_run_ids(args.data_root / 'splits' / 'test_runs.txt'))

    df = df.copy()
    df['split_tag'] = np.where(df['run_id'].isin(train_runs), 'train', np.where(df['run_id'].isin(val_runs), 'val', np.where(df['run_id'].isin(test_runs), 'test', '')))

    feature_rows = df[artifacts.feature_columns].replace({np.nan: None}).to_dict(orient='records')
    scores = predict_scores(artifacts, feature_rows)
    df['anomaly_score'] = scores.astype(float)

    current_threshold = float(artifacts.inference_config['threshold'])
    current_metrics = compute_metrics(df['is_anomaly'].to_numpy(dtype=int), scores, current_threshold)
    current_metrics['roc_auc'] = float(roc_auc_score(df['is_anomaly'].to_numpy(dtype=int), scores)) if len(np.unique(df['is_anomaly'])) > 1 else math.nan

    threshold_rows: list[dict[str, float]] = []
    for threshold in np.linspace(args.threshold_min, args.threshold_max, args.threshold_steps):
        threshold_rows.append(compute_metrics(df['is_anomaly'].to_numpy(dtype=int), scores, float(threshold)))

    best_balanced = max(threshold_rows, key=lambda row: row['balanced_accuracy'])
    best_f1 = max(threshold_rows, key=lambda row: row['f1_anomaly'])
    runtime_candidate = pick_runtime_threshold(threshold_rows, args.target_recall, args.specificity_floor)

    summary = {
        'artifact_dir': str(args.artifact_dir),
        'feature_file': str(feature_file),
        'rows': int(len(df)),
        'normal_rows': int((df['is_anomaly'] == 0).sum()),
        'fault_rows': int((df['is_anomaly'] == 1).sum()),
        'current_threshold_metrics': current_metrics,
        'best_balanced_accuracy_threshold': best_balanced,
        'best_f1_threshold': best_f1,
        'recommended_runtime_threshold': runtime_candidate,
        'score_distribution': {
            'normal': label_percentiles(df.loc[df['is_anomaly'] == 0, 'anomaly_score'].to_numpy(dtype=float)),
            'fault': label_percentiles(df.loc[df['is_anomaly'] == 1, 'anomaly_score'].to_numpy(dtype=float)),
        },
    }

    keep_cols = [
        column for column in [
            'system_id', 'run_id', 'window_id', 'window_phase', 'split_tag', 'sample_class',
            'fault_type', 'fault_family', 'root_cause_service', 'fault_target_service',
            'source_service', 'scenario_name', 'is_anomaly', 'anomaly_score'
        ] if column in df.columns
    ]
    df[keep_cols].to_csv(output_dir / 'anomaly_score_analysis.csv', index=False)
    pd.DataFrame(threshold_rows).to_csv(output_dir / 'threshold_candidates.csv', index=False)
    (output_dir / 'threshold_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f'Saved score analysis to {output_dir / "anomaly_score_analysis.csv"}')
    print(f'Saved threshold candidates to {output_dir / "threshold_candidates.csv"}')
    print(f'Saved summary to {output_dir / "threshold_summary.json"}')


if __name__ == '__main__':
    main()
