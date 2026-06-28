#!/usr/bin/env python3
"""
OmniDiag — Auto-Retrain Pipeline (Feature 4.3)
===============================================
Scheduled retraining script. Triggered by:
  1. Manual call:  python scripts/retrain.py --disease heart_disease
  2. Cron / Docker (see docker-compose.yml retrain service)
  3. Drift threshold breach via POST /admin/drift/{disease}/run

Pipeline:
  1. Load reference CSV + recent predictions from DB
  2. Merge new labelled samples (prediction > threshold treated as label)
  3. Retrain XGBoost / LGB model (or stacking ensemble for diabetes)
  4. Evaluate on held-out split — compare AUC vs. current production model
  5. If new model AUC > current AUC - tolerance → promote to production
  6. Log all metrics + artifacts to MLflow
  7. Flush predict cache so new model is served immediately

Usage:
    python scripts/retrain.py --disease heart_disease [--min-samples 500] [--auc-tolerance 0.01]
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure backend package is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("omnidiag.retrain")


def parse_args():
    p = argparse.ArgumentParser(description="OmniDiag auto-retrain pipeline")
    p.add_argument("--disease", required=True, help="Disease key (e.g. heart_disease, diabetes)")
    p.add_argument("--min-samples", type=int, default=200, help="Minimum new samples required to retrain")
    p.add_argument("--auc-tolerance", type=float, default=0.005, help="Allow promotion if new AUC >= current - tolerance")
    p.add_argument("--dry-run", action="store_true", help="Run pipeline but do not promote model")
    p.add_argument("--mlflow-uri", default=os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlruns.db"))
    return p.parse_args()


def load_reference_data(disease: str):
    """Load the reference (training) dataset for this disease."""
    paths = {
        "heart_disease": "data/heart_disease/processed/final_ready_data.csv",
        "diabetes": "data/diabetes/raw/diabetes_binary_5050split_health_indicators_BRFSS2015.csv",
    }
    ref_path = paths.get(disease)
    if ref_path is None or not Path(ref_path).exists():
        raise FileNotFoundError(f"Reference CSV not found for {disease}: {ref_path}")

    import pandas as pd
    df = pd.read_csv(ref_path)
    log.info("Loaded reference: %d rows, %d cols from %s", len(df), len(df.columns), ref_path)
    return df


def load_production_metrics(disease: str) -> dict:
    """Load cached production model metrics (AUC, F1, etc.)."""
    metrics_path = Path(f"models/{disease}/production_metrics.json")
    if metrics_path.exists():
        with open(metrics_path) as f:
            return json.load(f)
    return {"auc": 0.0, "f1": 0.0, "version": "unknown"}


def retrain_xgboost(X_train, y_train, X_val, y_val, params: dict = None):
    """Retrain XGBoost model and return (model, metrics)."""
    try:
        import xgboost as xgb
        from sklearn.metrics import roc_auc_score, f1_score
    except ImportError:
        raise RuntimeError("xgboost / sklearn not installed")

    default_params = {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "use_label_encoder": False,
        "eval_metric": "logloss",
        "random_state": 42,
    }
    if params:
        default_params.update(params)

    model = xgb.XGBClassifier(**default_params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    proba = model.predict_proba(X_val)[:, 1]
    pred  = (proba >= 0.5).astype(int)

    metrics = {
        "auc": float(roc_auc_score(y_val, proba)),
        "f1":  float(f1_score(y_val, pred)),
    }
    return model, metrics


def promote_model(model, disease: str, metrics: dict, version: str) -> None:
    """Save the new model to the production path."""
    import pickle
    model_dir = Path(f"models/{disease}")
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "omni_diag_xgb_optimized.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    # Update production metrics JSON
    metrics["version"] = version
    metrics["promoted_at"] = datetime.now(timezone.utc).isoformat()
    with open(model_dir / "production_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    log.info("Model promoted to production: %s (AUC=%.4f)", model_path, metrics["auc"])


def flush_cache() -> None:
    """Flush the predict cache via HTTP (best-effort)."""
    try:
        import requests
        admin_token = os.getenv("OMNIDIAG_ADMIN_TOKEN", "")
        r = requests.post(
            "http://localhost:7860/admin/cache/flush",
            headers={"Authorization": f"Bearer {admin_token}"},
            timeout=5,
        )
        log.info("Cache flush response: %s", r.status_code)
    except Exception as e:
        log.warning("Could not flush cache: %s", e)


def main() -> None:
    args = parse_args()
    disease = args.disease
    log.info("=" * 60)
    log.info("OmniDiag Retrain Pipeline — disease=%s", disease)
    log.info("dry_run=%s, min_samples=%d, auc_tolerance=%.3f", args.dry_run, args.min_samples, args.auc_tolerance)

    # 1. Load reference data
    try:
        df = load_reference_data(disease)
    except FileNotFoundError as e:
        log.error("Cannot start retrain: %s", e)
        sys.exit(1)

    # 2. Determine target column
    target_candidates = ["target", "HeartDisease", "Diabetes_binary", "label"]
    target_col = next((c for c in target_candidates if c in df.columns), None)
    if target_col is None:
        log.error("Cannot find target column. Available: %s", list(df.columns))
        sys.exit(1)

    feature_cols = [c for c in df.columns if c != target_col]
    X = df[feature_cols]
    y = df[target_col]

    if len(X) < args.min_samples:
        log.warning("Only %d samples available — minimum is %d. Skipping retrain.", len(X), args.min_samples)
        sys.exit(0)

    # 3. Train/val split
    try:
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    except ImportError:
        log.error("scikit-learn not installed")
        sys.exit(1)

    log.info("Train: %d rows, Val: %d rows", len(X_train), len(X_val))

    # 4. Load production metrics for comparison
    current_metrics = load_production_metrics(disease)
    log.info("Current production AUC: %.4f (v%s)", current_metrics.get("auc", 0), current_metrics.get("version"))

    # 5. Retrain
    log.info("Training new model…")
    try:
        new_model, new_metrics = retrain_xgboost(X_train, y_train, X_val, y_val)
    except Exception as e:
        log.error("Retrain failed: %s", e)
        sys.exit(1)

    log.info("New model — AUC=%.4f, F1=%.4f", new_metrics["auc"], new_metrics["f1"])

    # 6. Decide to promote
    promote = new_metrics["auc"] >= (current_metrics.get("auc", 0.0) - args.auc_tolerance)
    version = datetime.now(timezone.utc).strftime("v%Y%m%d_%H%M")

    # 7. Log to MLflow
    try:
        from backend.monitoring.mlflow_tracker import log_model_info as mlflow_log
        run_id = mlflow_log(
            disease=disease,
            model_version=version,
            metrics={
                "new_auc": new_metrics["auc"],
                "new_f1": new_metrics["f1"],
                "current_auc": current_metrics.get("auc", 0.0),
                "train_samples": len(X_train),
                "val_samples": len(X_val),
                "promoted": float(promote and not args.dry_run),
            },
            params={"disease": disease, "version": version},
        )
        log.info("MLflow run logged: %s", run_id)
    except Exception as e:
        log.warning("MLflow logging failed: %s", e)

    # 8. Promote or skip
    if args.dry_run:
        log.info("DRY RUN — model NOT promoted. New AUC=%.4f", new_metrics["auc"])
    elif promote:
        promote_model(new_model, disease, new_metrics, version)
        flush_cache()
        log.info("Retrain complete — new model PROMOTED (AUC %.4f → %.4f)", current_metrics.get("auc", 0), new_metrics["auc"])
    else:
        log.warning(
            "New model AUC (%.4f) did not meet threshold (current %.4f - tolerance %.3f = %.4f). NOT promoting.",
            new_metrics["auc"],
            current_metrics.get("auc", 0.0),
            args.auc_tolerance,
            current_metrics.get("auc", 0.0) - args.auc_tolerance,
        )

    log.info("Pipeline complete.")


if __name__ == "__main__":
    main()
