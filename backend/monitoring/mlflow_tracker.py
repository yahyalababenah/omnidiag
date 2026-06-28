"""
OmniDiag — MLflow Experiment Tracking (Feature 4.2)
=====================================================
Logs model metadata, performance metrics, and hyperparameters to MLflow.

Usage:
    from backend.monitoring.mlflow_tracker import log_model_info, log_prediction_batch

    # On startup (register existing model artifacts):
    log_model_info(disease="heart_disease", model_version="v5.1", metrics={...})

    # After drift run:
    log_drift_metrics(disease="heart_disease", drift_share=0.12, run_date=...)
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

log = logging.getLogger("omnidiag.mlflow")

_mlflow_available = False
try:
    import mlflow
    import mlflow.sklearn
    _mlflow_available = True
except ImportError:
    log.warning("mlflow not installed — experiment tracking unavailable")

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlruns.db")
EXPERIMENT_NAME = "OmniDiag"


def _get_client():
    """Return an MLflow client pointed at the configured tracking server."""
    if not _mlflow_available:
        return None
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    return mlflow.MlflowClient()


def ensure_experiment() -> Optional[str]:
    """Create or get the OmniDiag MLflow experiment. Returns experiment_id."""
    if not _mlflow_available:
        return None
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        exp_id = mlflow.create_experiment(
            EXPERIMENT_NAME,
            tags={"project": "omnidiag", "team": "AI Health"},
        )
        log.info("Created MLflow experiment '%s' (id=%s)", EXPERIMENT_NAME, exp_id)
    else:
        exp_id = exp.experiment_id
    return exp_id


def log_model_info(
    disease: str,
    model_version: str,
    metrics: Dict[str, float],
    params: Optional[Dict[str, Any]] = None,
    artifact_paths: Optional[Dict[str, str]] = None,
) -> Optional[str]:
    """
    Start a new MLflow run and log model metadata + evaluation metrics.
    Returns the run_id or None if MLflow is unavailable.
    """
    if not _mlflow_available:
        log.debug("MLflow unavailable — skipping log_model_info for %s", disease)
        return None

    exp_id = ensure_experiment()
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    with mlflow.start_run(experiment_id=exp_id, run_name=f"{disease}_{model_version}") as run:
        mlflow.set_tags({
            "disease": disease,
            "model_version": model_version,
            "logged_at": datetime.now(timezone.utc).isoformat(),
        })

        if params:
            mlflow.log_params(params)

        if metrics:
            mlflow.log_metrics(metrics)

        # Log model artifact directory if it exists
        model_dir = Path(f"models/{disease}")
        if model_dir.exists() and artifact_paths is None:
            for f in model_dir.glob("*.pkl"):
                mlflow.log_artifact(str(f), artifact_path="model")

        if artifact_paths:
            for name, path in artifact_paths.items():
                if Path(path).exists():
                    mlflow.log_artifact(path, artifact_path=name)

        run_id = run.info.run_id
        log.info("MLflow: logged model info for %s v%s (run_id=%s)", disease, model_version, run_id)
        return run_id


def log_drift_metrics(
    disease: str,
    drift_share: float,
    drifted_columns: int,
    total_columns: int,
    sample_size: int,
    run_date: Optional[datetime] = None,
) -> Optional[str]:
    """Log a drift report as an MLflow run."""
    if not _mlflow_available:
        return None

    exp_id = ensure_experiment()
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    ts = (run_date or datetime.now(timezone.utc)).strftime("%Y%m%d_%H%M%S")

    with mlflow.start_run(experiment_id=exp_id, run_name=f"drift_{disease}_{ts}") as run:
        mlflow.set_tags({
            "disease": disease,
            "run_type": "drift",
            "run_date": ts,
        })
        mlflow.log_metrics({
            "drift_share": drift_share,
            "drifted_columns": float(drifted_columns),
            "total_columns": float(total_columns),
            "sample_size": float(sample_size),
        })
        run_id = run.info.run_id
        log.info("MLflow: logged drift metrics for %s (run_id=%s drift_share=%.2f%%)", disease, run_id, drift_share * 100)
        return run_id


def list_recent_runs(n: int = 20) -> list:
    """Return recent MLflow runs as list of dicts."""
    if not _mlflow_available:
        return []
    client = _get_client()
    exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        return []
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        max_results=n,
        order_by=["start_time DESC"],
    )
    return [
        {
            "run_id": r.info.run_id,
            "name": r.data.tags.get("mlflow.runName", r.info.run_id[:8]),
            "disease": r.data.tags.get("disease"),
            "run_type": r.data.tags.get("run_type", "model"),
            "status": r.info.status,
            "start_time": r.info.start_time,
            "metrics": dict(r.data.metrics),
            "params": dict(r.data.params),
        }
        for r in runs
    ]
