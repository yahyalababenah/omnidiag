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

# MLflow's HTTP client retries a dead tracking server with backoff for minutes.
# That is reasonable for a training job and wrong for an admin endpoint and for
# an image build: the status check hung for over four minutes against a closed
# port instead of answering "unreachable" (measured, Gate 8.6). Bounded here,
# and only when the operator has not set their own values.
os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "1")
os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "5")


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


def log_build_artifact(
    disease: str,
    model_version: str,
    params: Dict[str, Any],
    metrics: Dict[str, float],
    tags: Optional[Dict[str, str]] = None,
) -> Optional[str]:
    """
    Record the artifact that was just built, at the moment it was built.

    This is the only thing in the system that logs to MLflow automatically, and
    it exists because nothing did: the one automatic caller was `retrain_xgb`,
    which fails for heart and is a no-op for diabetes, so the experiment was
    empty by construction rather than by accident (Gate 8.6).

    Build time is the right moment because it is the only moment the shipped
    artifact is created. What goes in is provenance -- the training data's hash,
    the artifact's hash, the reproducibility fingerprint, the declared limits --
    and NOT performance figures, which are measured in the research repository
    and are not recomputed here.

    NEVER raises. A build must not fail because a tracking store was absent,
    and on a deployment whose MLFLOW_TRACKING_URI points at a server there is no
    server to reach during an image build. Returns the run id, or None with a
    printed reason.
    """
    if not _mlflow_available:
        log.info("mlflow not installed — build artifact not logged")
        return None
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)
        with mlflow.start_run(run_name=f"{disease}-build-{model_version}") as run:
            mlflow.set_tags({
                "disease": disease,
                "model_version": model_version,
                "run_type": "build",
                **(tags or {}),
            })
            mlflow.log_params(params)
            if metrics:
                mlflow.log_metrics(metrics)
            return run.info.run_id
    except Exception as exc:  # noqa: BLE001 — tracking is never worth a failed build
        log.warning("mlflow build logging skipped — %s: %s", type(exc).__name__, exc)
        return None


def tracking_state(n: int = 20) -> Dict[str, Any]:
    """
    What MLflow is actually doing, as four distinguishable states.

    `list_recent_runs()` returns [] when the package is missing, when the store
    is unreachable, and when the store is reachable but has no runs. The admin
    endpoint therefore reported `count: 0` for all three, and "MLflow is empty"
    could not be told apart from "MLflow is not installed" -- which is how the
    empty experiment went unexplained. Each state now names itself:

        unavailable  the mlflow package is not installed here
        unreachable  installed, but the tracking store did not answer
        empty        working, and nothing has been logged to it
        ok           working, with runs

    Returns the runs too, so a caller needs one call rather than two.
    """
    if not _mlflow_available:
        return {
            "status": "unavailable",
            "tracking_uri": MLFLOW_TRACKING_URI,
            "detail": "the mlflow package is not installed in this environment",
            "runs": [],
        }
    try:
        client = _get_client()
        experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    except Exception as exc:  # noqa: BLE001 — any store failure is one state
        return {
            "status": "unreachable",
            "tracking_uri": MLFLOW_TRACKING_URI,
            "detail": f"{type(exc).__name__}: {exc}",
            "runs": [],
        }
    if experiment is None:
        return {
            "status": "empty",
            "tracking_uri": MLFLOW_TRACKING_URI,
            "detail": f"no experiment named {EXPERIMENT_NAME!r} in this store",
            "runs": [],
        }
    try:
        runs = list_recent_runs(n=n)
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "unreachable",
            "tracking_uri": MLFLOW_TRACKING_URI,
            "detail": f"{type(exc).__name__}: {exc}",
            "runs": [],
        }
    return {
        "status": "ok" if runs else "empty",
        "tracking_uri": MLFLOW_TRACKING_URI,
        "detail": None if runs else "the experiment exists and has no runs",
        "runs": runs,
    }


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
