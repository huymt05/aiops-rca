from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover - optional dependency
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover - optional dependency
    LGBMClassifier = None

try:
    import dagshub
except Exception:  # pragma: no cover - optional dependency
    dagshub = None

try:
    import mlflow
except Exception:  # pragma: no cover - optional dependency
    mlflow = None

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.rca_data_pipeline.run_manifest import build_run_manifest, collect_package_versions, write_run_manifest
from aiops_framework.inference.common.artifact_registry import set_stage


DEFAULT_DATA_ROOT = Path(r"D:\HOCTAP\2025-2026\HK2\DACN\microservices-demo\data_anomaly_balanced_v3")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a supervised anomaly model on RCA pipeline window features.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--feature-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--model-kind", choices=["auto", "ensemble", "xgb", "lgbm", "gbrt"], default="auto")
    parser.add_argument("--optimize-for", choices=["anomaly", "normal"], default="anomaly")
    parser.add_argument(
        "--threshold-bias",
        type=float,
        default=0.0,
        help="Additive adjustment applied after threshold search. Negative values make RCA triggering easier.",
    )
    parser.add_argument("--mlflow", action="store_true", help="Enable MLflow logging for params, metrics, and artifacts.")
    parser.add_argument("--mlflow-experiment", default="aiops-microservices-demo")
    return parser.parse_args()


def read_run_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sanitize_sample_class(df: pd.DataFrame) -> pd.DataFrame:
    if "sample_class" in df.columns:
        fill_values = []
        for _, row in df.iterrows():
            sample_class = row.get("sample_class")
            if pd.isna(sample_class) or str(sample_class).strip().lower() in {"", "nan", "none"}:
                fill_values.append("fault" if int(row.get("is_anomaly", 0)) == 1 else "normal")
            else:
                fill_values.append(str(sample_class).strip().lower())
        df = df.copy()
        df["sample_class"] = fill_values
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    exclude = {
        "system_id",
        "run_id",
        "window_id",
        "window_phase",
        "label",
        "sample_class",
        "phase_policy",
        "fault_type",
        "fault_family",
        "root_cause_service",
        "fault_target_service",
        "fault_target_role",
        "source_service",
        "source_service_role",
        "scenario_name",
        "start_time",
        "fault_start_time",
        "fault_end_time",
        "end_time",
        "is_anomaly",
    }
    numeric_cols = []
    for column in df.columns:
        if column in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[column]):
            numeric_cols.append(column)
    return numeric_cols


def compute_metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float = 0.5) -> dict[str, object]:
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) else 0.0
    metrics = {
        "precision_anomaly": float(precision_score(y_true, pred, zero_division=0)),
        "recall_anomaly": float(recall_score(y_true, pred, zero_division=0)),
        "specificity": specificity,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "f1_anomaly": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, prob)) if len(np.unique(y_true)) > 1 else math.nan,
        "average_precision": float(average_precision_score(y_true, prob)) if len(np.unique(y_true)) > 1 else math.nan,
        "confusion_matrix": {"tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)},
        "threshold": threshold,
    }
    return metrics


def best_threshold(y_true: np.ndarray, prob: np.ndarray, optimize_for: str = "anomaly") -> float:
    best_t = 0.5
    best_score = -1.0
    low, high = (0.2, 0.8) if optimize_for == "anomaly" else (0.35, 0.95)
    for threshold in np.linspace(low, high, 31):
        metrics = compute_metrics(y_true, prob, float(threshold))
        if optimize_for == "normal":
            recall = float(metrics["recall_anomaly"])
            # Keep some anomaly sensitivity so the model does not collapse to predicting mostly normal.
            if recall < 0.25:
                score = -1.0 + recall
            else:
                score = (
                    0.60 * float(metrics["specificity"])
                    + 0.20 * float(metrics["balanced_accuracy"])
                    + 0.10 * float(metrics["precision_anomaly"])
                    + 0.10 * float(metrics["f1_anomaly"])
                )
        else:
            score = f1_score(y_true, (prob >= threshold).astype(int), zero_division=0)
        if score > best_score:
            best_score = score
            best_t = float(threshold)
    return best_t


def apply_threshold_bias(threshold: float, bias: float) -> float:
    adjusted = float(threshold) + float(bias)
    return float(min(0.99, max(0.01, adjusted)))


def ranking_score(metrics: dict[str, object], optimize_for: str = "anomaly") -> float:
    if optimize_for == "normal":
        return (
            0.45 * float(metrics["specificity"])
            + 0.25 * float(metrics["balanced_accuracy"])
            + 0.15 * float(metrics["precision_anomaly"])
            + 0.15 * float(metrics["f1_anomaly"])
        )
    return 0.55 * float(metrics["f1_anomaly"]) + 0.25 * float(metrics["balanced_accuracy"]) + 0.20 * float(metrics["average_precision"])


def normalized_importance(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    denom = float(arr.sum())
    return arr / denom if denom > 0 else arr


def _to_mlflow_param(value: object) -> str | int | float | bool:
    if isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value, ensure_ascii=False)


def setup_mlflow_if_requested(args: argparse.Namespace) -> bool:
    should_enable = bool(
        args.mlflow
        or os.environ.get("MLFLOW_TRACKING_URI")
        or (
            os.environ.get("DAGSHUB_USERNAME")
            and os.environ.get("DAGSHUB_REPO")
        )
    )
    if not should_enable:
        return False
    if mlflow is None:
        raise RuntimeError("MLflow logging was requested but mlflow is not installed.")

    dagshub_user = os.environ.get("DAGSHUB_USERNAME")
    dagshub_repo = os.environ.get("DAGSHUB_REPO")
    if dagshub_user and dagshub_repo:
        if dagshub is None:
            raise RuntimeError("DagsHub logging was requested but dagshub is not installed.")
        dagshub.init(repo_owner=dagshub_user, repo_name=dagshub_repo, mlflow=True)

    experiment_name = str(args.mlflow_experiment).strip() or "aiops-microservices-demo"
    mlflow.set_experiment(experiment_name)
    return True


def log_mlflow_run(
    *,
    args: argparse.Namespace,
    data_root: Path,
    output_dir: Path,
    feature_file: Path,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    best_params: dict[str, object],
    best_thr: float,
    best_val: dict[str, object],
    test_metrics: dict[str, object],
) -> None:
    if mlflow is None:
        return

    run_name = f"anomaly_{output_dir.name}"
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("task", "anomaly_detection")
        mlflow.set_tag("dataset", data_root.name)
        mlflow.set_tag("model_name", output_dir.name)

        params = {
            "data_root": str(data_root),
            "feature_file": str(feature_file),
            "output_dir": str(output_dir),
            "model_kind": args.model_kind,
            "optimize_for": args.optimize_for,
            "random_state": args.random_state,
            "threshold": float(best_thr),
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
        }
        for key, value in params.items():
            mlflow.log_param(key, _to_mlflow_param(value))
        for key, value in best_params.items():
            mlflow.log_param(f"best_{key}", _to_mlflow_param(value))

        for prefix, metrics_dict in (("val", best_val), ("test", test_metrics)):
            for key, value in metrics_dict.items():
                if isinstance(value, (int, float)) and not math.isnan(float(value)):
                    mlflow.log_metric(f"{prefix}_{key}", float(value))

        for artifact_name in [
            "metrics.json",
            "feature_columns.json",
            "feature_importance.csv",
            "inference_config.json",
            "imputer.joblib",
            "model.joblib",
            "model_xgb.joblib",
            "model_lgbm.joblib",
        ]:
            artifact_path = output_dir / artifact_name
            if artifact_path.exists():
                mlflow.log_artifact(str(artifact_path))


def main() -> None:
    args = parse_args()
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
    data_root = args.data_root
    feature_file = args.feature_file or (data_root / "processed" / "anomaly" / "window_features_labeled.parquet")
    output_dir = args.output_dir or (data_root / "models" / "anomaly_xgb_lgbm")
    output_dir.mkdir(parents=True, exist_ok=True)

    train_runs = set(read_run_ids(data_root / "splits" / "train_runs.txt"))
    val_runs = set(read_run_ids(data_root / "splits" / "val_runs.txt"))
    test_runs = set(read_run_ids(data_root / "splits" / "test_runs.txt"))

    df = pd.read_parquet(feature_file)
    df = sanitize_sample_class(df)
    cols = feature_columns(df)

    train_df = df[df["run_id"].isin(train_runs)].copy()
    val_df = df[df["run_id"].isin(val_runs)].copy()
    test_df = df[df["run_id"].isin(test_runs)].copy()

    X_train = train_df[cols].astype(float).copy()
    X_val = val_df[cols].astype(float).copy()
    X_test = test_df[cols].astype(float).copy()
    y_train = train_df["is_anomaly"].to_numpy(dtype=int)
    y_val = val_df["is_anomaly"].to_numpy(dtype=int)
    y_test = test_df["is_anomaly"].to_numpy(dtype=int)

    imputer = SimpleImputer(strategy="median")
    X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=cols)
    X_val = pd.DataFrame(imputer.transform(X_val), columns=cols)
    X_test = pd.DataFrame(imputer.transform(X_test), columns=cols)

    pos_weight = float(max(1.0, (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)))
    sample_weight = np.where(y_train == 1, pos_weight, 1.0)
    if args.model_kind in {"auto", "ensemble", "xgb"} and XGBClassifier is None:
        raise RuntimeError("xgboost is not installed but was requested for anomaly training.")
    if args.model_kind in {"auto", "ensemble", "lgbm"} and LGBMClassifier is None:
        raise RuntimeError("lightgbm is not installed but was requested for anomaly training.")

    xgb_candidates = [
        {
            "kind": "xgb",
            "n_estimators": 240,
            "learning_rate": 0.05,
            "max_depth": 4,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
        },
        {
            "kind": "xgb",
            "n_estimators": 320,
            "learning_rate": 0.03,
            "max_depth": 5,
            "subsample": 0.9,
            "colsample_bytree": 0.8,
        },
    ]
    lgbm_candidates = [
        {
            "kind": "lgbm",
            "n_estimators": 220,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": -1,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
        },
        {
            "kind": "lgbm",
            "n_estimators": 320,
            "learning_rate": 0.03,
            "num_leaves": 63,
            "max_depth": -1,
            "subsample": 0.9,
            "colsample_bytree": 0.8,
        },
    ]
    gbrt_candidates = [{"kind": "gbrt", "learning_rate": 0.05, "max_depth": 3}]

    if args.model_kind == "xgb":
        candidates = xgb_candidates
    elif args.model_kind == "lgbm":
        candidates = lgbm_candidates
    elif args.model_kind == "gbrt":
        candidates = gbrt_candidates
    else:
        candidates = xgb_candidates + lgbm_candidates + ([] if args.model_kind == "ensemble" else gbrt_candidates)

    trained_models: list[dict[str, object]] = []
    tuning_results = []

    for candidate in candidates:
        params = dict(candidate)
        kind = str(params.pop("kind"))
        if kind == "xgb":
            model = XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=args.random_state,
                n_jobs=1,
                tree_method="hist",
                scale_pos_weight=pos_weight,
                **params,
            )
            fit_kwargs = {}
        elif kind == "lgbm":
            model = LGBMClassifier(
                objective="binary",
                random_state=args.random_state,
                n_jobs=1,
                class_weight={0: 1.0, 1: pos_weight},
                verbosity=-1,
                **params,
            )
            fit_kwargs = {}
        else:
            model = GradientBoostingClassifier(
                random_state=args.random_state,
                n_estimators=250,
                subsample=0.9,
                **params,
            )
            fit_kwargs = {"sample_weight": sample_weight}

        try:
            model.fit(X_train, y_train, **fit_kwargs)
        except Exception as exc:
            tuning_results.append({"params": candidate, "fit_error": str(exc)})
            continue

        val_prob = model.predict_proba(X_val)[:, 1]
        threshold = best_threshold(y_val, val_prob, optimize_for=args.optimize_for)
        threshold = apply_threshold_bias(threshold, args.threshold_bias)
        metrics = compute_metrics(y_val, val_prob, threshold)
        rank = ranking_score(metrics, optimize_for=args.optimize_for)
        trained_models.append(
            {
                "kind": kind,
                "params": candidate,
                "model": model,
                "val_prob": val_prob,
                "val_metrics": metrics,
                "ranking_score": rank,
            }
        )
        tuning_results.append({"params": candidate, "val_metrics": metrics, "ranking_score": rank})

    if not trained_models:
        raise RuntimeError("No anomaly model candidate trained successfully. Check the fit_error entries in tuning_results.")

    best_individual = max(trained_models, key=lambda item: float(item["ranking_score"]))
    ensemble_used = False
    best_params: dict[str, object] = dict(best_individual["params"])
    best_val = dict(best_individual["val_metrics"])
    best_thr = float(best_val["threshold"])
    test_prob = best_individual["model"].predict_proba(X_test)[:, 1]
    test_metrics = compute_metrics(y_test, test_prob, best_thr)

    xgb_best = max((item for item in trained_models if item["kind"] == "xgb"), key=lambda item: float(item["ranking_score"]), default=None)
    lgbm_best = max((item for item in trained_models if item["kind"] == "lgbm"), key=lambda item: float(item["ranking_score"]), default=None)

    if args.model_kind in {"auto", "ensemble"} and xgb_best is not None and lgbm_best is not None:
        ensemble_val_prob = 0.5 * np.asarray(xgb_best["val_prob"]) + 0.5 * np.asarray(lgbm_best["val_prob"])
        ensemble_thr = best_threshold(y_val, ensemble_val_prob, optimize_for=args.optimize_for)
        ensemble_thr = apply_threshold_bias(ensemble_thr, args.threshold_bias)
        ensemble_val_metrics = compute_metrics(y_val, ensemble_val_prob, ensemble_thr)
        ensemble_rank = ranking_score(ensemble_val_metrics, optimize_for=args.optimize_for)
        tuning_results.append(
            {
                "params": {"kind": "ensemble", "members": [xgb_best["params"], lgbm_best["params"]]},
                "val_metrics": ensemble_val_metrics,
                "ranking_score": ensemble_rank,
            }
        )
        if ensemble_rank >= float(best_individual["ranking_score"]):
            ensemble_used = True
            best_params = {"kind": "ensemble", "members": [xgb_best["params"], lgbm_best["params"]]}
            best_val = ensemble_val_metrics
            best_thr = ensemble_thr
            xgb_test_prob = xgb_best["model"].predict_proba(X_test)[:, 1]
            lgbm_test_prob = lgbm_best["model"].predict_proba(X_test)[:, 1]
            test_prob = 0.5 * xgb_test_prob + 0.5 * lgbm_test_prob
            test_metrics = compute_metrics(y_test, test_prob, best_thr)

    artifacts = {
        "data_root": str(data_root),
        "feature_file": str(feature_file),
        "feature_columns": cols,
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "train_runs": len(train_runs),
        "val_runs": len(val_runs),
        "test_runs": len(test_runs),
        "optimize_for": args.optimize_for,
        "threshold_bias": args.threshold_bias,
        "ensemble_used": ensemble_used,
        "best_params": best_params,
        "best_threshold": best_thr,
        "val_metrics": best_val,
        "test_metrics": test_metrics,
        "tuning_results": tuning_results,
    }

    (output_dir / "metrics.json").write_text(json.dumps(artifacts, indent=2, ensure_ascii=False), encoding="utf-8")
    importance_parts = []
    if ensemble_used and xgb_best is not None and lgbm_best is not None:
        xgb_imp = normalized_importance(getattr(xgb_best["model"], "feature_importances_", np.zeros(len(cols))))
        lgbm_imp = normalized_importance(getattr(lgbm_best["model"], "feature_importances_", np.zeros(len(cols))))
        importance_parts.append(("xgb", xgb_imp))
        importance_parts.append(("lgbm", lgbm_imp))
        final_importance = (xgb_imp + lgbm_imp) / 2.0
    else:
        chosen_kind = str(best_params.get("kind"))
        chosen_model = best_individual["model"]
        final_importance = normalized_importance(getattr(chosen_model, "feature_importances_", np.zeros(len(cols))))
        importance_parts.append((chosen_kind, final_importance))

    importance_df = pd.DataFrame({"feature": cols, "importance": final_importance})
    for name, values in importance_parts:
        importance_df[f"importance_{name}"] = values
    importance_df.sort_values("feature").to_csv(output_dir / "feature_importance.csv", index=False)

    joblib.dump(imputer, output_dir / "imputer.joblib")
    if ensemble_used and xgb_best is not None and lgbm_best is not None:
        joblib.dump(xgb_best["model"], output_dir / "model_xgb.joblib")
        joblib.dump(lgbm_best["model"], output_dir / "model_lgbm.joblib")
        model_artifacts = {
            "kind": "ensemble",
            "members": [
                {
                    "kind": "xgb",
                    "artifact": "model_xgb.joblib",
                    "params": xgb_best["params"],
                },
                {
                    "kind": "lgbm",
                    "artifact": "model_lgbm.joblib",
                    "params": lgbm_best["params"],
                },
            ],
        }
    else:
        chosen_model = best_individual["model"]
        joblib.dump(chosen_model, output_dir / "model.joblib")
        model_artifacts = {
            "kind": str(best_params.get("kind")),
            "artifact": "model.joblib",
            "params": best_params,
        }

    (output_dir / "feature_columns.json").write_text(json.dumps(cols, indent=2, ensure_ascii=False), encoding="utf-8")
    inference_config = {
        "model_name": output_dir.name,
        "model_kind": str(best_params.get("kind")),
        "optimize_for": args.optimize_for,
        "threshold": best_thr,
        "threshold_bias": args.threshold_bias,
        "feature_columns_artifact": "feature_columns.json",
        "imputer_artifact": "imputer.joblib",
        "model_artifacts": model_artifacts,
        "metrics_artifact": "metrics.json",
        "feature_importance_artifact": "feature_importance.csv",
        "run_manifest_artifact": "run_manifest.json",
        "train_runs": len(train_runs),
        "val_runs": len(val_runs),
        "test_runs": len(test_runs),
    }
    (output_dir / "inference_config.json").write_text(
        json.dumps(inference_config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    package_versions = collect_package_versions(
        ["numpy", "pandas", "scikit-learn", "joblib", "xgboost", "lightgbm", "mlflow", "dagshub"]
    )
    manifest = build_run_manifest(
        repo_root=REPO_ROOT,
        task="anomaly_training",
        output_dir=output_dir,
        train_script=Path(__file__).resolve(),
        data_root=data_root,
        args=vars(args),
        artifacts={
            "metrics": "metrics.json",
            "feature_columns": "feature_columns.json",
            "feature_importance": "feature_importance.csv",
            "inference_config": "inference_config.json",
            "imputer": "imputer.joblib",
            "model": model_artifacts,
        },
        dataset={
            "feature_file": feature_file,
            "train_rows": len(train_df),
            "val_rows": len(val_df),
            "test_rows": len(test_df),
            "train_runs": len(train_runs),
            "val_runs": len(val_runs),
            "test_runs": len(test_runs),
            "feature_columns": cols,
        },
        package_versions=package_versions,
        extra={
            "best_params": best_params,
            "best_threshold": best_thr,
            "val_metrics": best_val,
            "test_metrics": test_metrics,
        },
    )
    write_run_manifest(output_dir, manifest)
    set_stage(data_root / "models", "anomaly", "candidate", output_dir.name, notes="Updated after anomaly training.")

    if setup_mlflow_if_requested(args):
        log_mlflow_run(
            args=args,
            data_root=data_root,
            output_dir=output_dir,
            feature_file=feature_file,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            best_params=best_params,
            best_thr=best_thr,
            best_val=best_val,
            test_metrics=test_metrics,
        )

    print("Best anomaly model selected.")
    print(json.dumps({"val_metrics": best_val, "test_metrics": test_metrics, "best_params": best_params}, indent=2, ensure_ascii=False))
    print(f"Saved metrics to {output_dir / 'metrics.json'}")
    print(f"Updated candidate registry at {data_root / 'models' / 'model_registry.json'}")


if __name__ == "__main__":
    main()
