"""
OmniDiag — Automated Retraining Pipeline (Feature 4.3)
=======================================================
Pulls annotated samples from the ReviewQueue, merges them with the original
training dataset, and retrains the XGBoost model incrementally.

Usage (CLI):
    python -m backend.active_learning.retrain --disease heart_disease --min-samples 10

API:
    POST /api/v4/admin/retrain  (super_admin only)

Flow:
    1. Query ReviewQueue for annotated (status='annotated') rows
    2. Fetch the linked Prediction's features + the expert label
    3. Append to original CSV (or create a new combined CSV)
    4. Retrain XGBoost (incremental via xgb_model= parameter)
    5. Save new model to disk
    6. Reload the running ModelLoader so new predictions use updated model
    7. Log the run to MLflow (if available)
"""

from __future__ import annotations

import asyncio
import logging
import os
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("omnidiag.retrain")

MODELS_DIR = Path(os.getenv("MODELS_DIR", "models"))


async def get_annotated_samples(
    db,
    disease: str,
    min_samples: int = 5,
) -> Tuple[List[Dict[str, Any]], List[int]]:
    """
    Pull annotated ReviewQueue items from the database.

    Returns (features_list, labels_list) where each features_list[i] is a
    dict of the original prediction inputs and labels_list[i] is the expert label.
    """
    from sqlalchemy import text as _text

    rows = (await db.execute(
        _text("""
            SELECT rq.label, p.input_features
            FROM review_queue rq
            JOIN predictions p ON p.id = rq.prediction_id
            WHERE rq.status = 'reviewed'
              AND rq.label IS NOT NULL
              AND p.disease = :disease
            LIMIT 1000
        """),
        {"disease": disease},
    )).fetchall()

    if len(rows) < min_samples:
        return [], []

    import json as _json
    features_list: List[Dict[str, Any]] = []
    labels: List[int] = []

    for row in rows:
        expert_label, raw_features = row[0], row[1]
        try:
            feats = _json.loads(raw_features) if isinstance(raw_features, str) else dict(raw_features or {})
            if feats:
                features_list.append(feats)
                labels.append(int(expert_label))
        except Exception:
            continue

    return features_list, labels


def retrain_xgb(
    disease: str,
    features_list: List[Dict[str, Any]],
    labels: List[int],
) -> Optional[Path]:
    """
    Incrementally retrain the XGBoost model for the given disease.

    Returns the path to the newly saved model, or None on failure.

    🚧 KNOWN LIMITATION (confirmed via live testing, 2026-09-07):
      - heart_disease: `features_list[i].values()` are the RAW predict-time
        inputs (e.g. Sex="M", ChestPainType="ATA") — not label-encoded or
        feature-engineered. Building `X` directly from these raises
        `ValueError: could not convert string to float: 'ATA'`. Any real
        active-learning cycle for heart_disease currently fails here.
        Fix requires running the same encode→engineer pipeline used by
        ModelLoader.predict() before constructing the DMatrix.
      - diabetes: this function always writes to the single hardcoded path
        `models/{disease}/omni_diag_xgb_optimized.pkl`, which is NOT one of
        the three files EnsembleModelLoader actually loads (xgb_model.pkl,
        lgb_model.pkl, rf_model.pkl + meta_learner.pkl). Even when this
        succeeds numerically, it retrains an orphan file disconnected from
        the live ensemble — a silent no-op for diabetes.
    Hot-reload (ModelLoader.invalidate() / EnsembleModelLoader.invalidate(),
    wired in run_retrain_pipeline() below) is verified working correctly in
    isolation — it is this function's own model-building step that blocks
    the pipeline before reload is ever reached.
    """
    try:
        import xgboost as xgb
        import numpy as np

        model_path = MODELS_DIR / disease / "omni_diag_xgb_optimized.pkl"
        if not model_path.exists():
            log.error(f"Model not found: {model_path}")
            return None

        with open(model_path, "rb") as f:
            model = pickle.load(f)

        X = np.array([list(feat.values()) for feat in features_list], dtype=np.float32)
        y = np.array(labels, dtype=np.float32)

        dtrain = xgb.DMatrix(X, label=y)
        updated_model = xgb.train(
            params={
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "max_depth": 5,
                "learning_rate": 0.05,
            },
            dtrain=dtrain,
            num_boost_round=20,
            xgb_model=model,
            verbose_eval=False,
        )

        backup_path = model_path.with_suffix(f".bak.{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.pkl")
        model_path.rename(backup_path)

        with open(model_path, "wb") as f:
            pickle.dump(updated_model, f)

        log.info(f"Retrained {disease} model saved to {model_path} ({len(labels)} new samples)")
        return model_path

    except Exception as exc:
        log.error(f"Retraining failed for {disease}: {exc!r}")
        return None


def _log_to_mlflow(disease: str, n_samples: int, model_path: Optional[Path]) -> None:
    try:
        from backend.monitoring.mlflow_tracker import log_model_info
        log_model_info(
            disease=disease,
            model_version=f"retrain_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
            metrics={"retrain_samples": n_samples, "success": 1 if model_path else 0},
        )
    except Exception as exc:
        log.warning(f"MLflow logging failed: {exc!r}")


async def run_retrain_pipeline(
    db,
    disease: str,
    min_samples: int = 5,
) -> Dict[str, Any]:
    """
    Full pipeline: fetch annotated samples → retrain → reload → log.

    Returns a status dict suitable for the API response.
    """
    features_list, labels = await get_annotated_samples(db, disease, min_samples)

    if not features_list:
        return {
            "status": "skipped",
            "disease": disease,
            "reason": f"Fewer than {min_samples} annotated samples available",
            "samples_used": 0,
        }

    model_path = retrain_xgb(disease, features_list, labels)

    _log_to_mlflow(disease, len(labels), model_path)

    if model_path:
        # Reload model in running process so next predict uses updated weights.
        # The router (backend.main.router) holds the live ModelLoader /
        # EnsembleModelLoader instance for this disease; invalidate() clears
        # its cached model so the next request lazy-reloads the new weights.
        try:
            from backend.main import router as _router
            loader = _router.model_loaders.get(disease)
            if loader is not None and hasattr(loader, "invalidate"):
                loader.invalidate()
                log.info(f"Hot-reloaded model loader for {disease}")
            else:
                log.warning(
                    f"No active loader found for {disease} — "
                    f"will take effect on next startup"
                )
        except Exception as exc:
            log.warning(f"Model hot-reload failed (will take effect on next startup): {exc!r}")

        return {
            "status": "success",
            "disease": disease,
            "samples_used": len(labels),
            "model_path": str(model_path),
        }
    else:
        return {
            "status": "failed",
            "disease": disease,
            "samples_used": len(labels),
            "reason": "Retraining step failed — check logs",
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="OmniDiag retraining pipeline")
    parser.add_argument("--disease", default="heart_disease", help="Disease module to retrain")
    parser.add_argument("--min-samples", type=int, default=10, help="Minimum annotated samples required")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    async def _main():
        from backend.database import async_session_maker
        async with async_session_maker() as db:
            result = await run_retrain_pipeline(db, args.disease, args.min_samples)
            print(result)

    asyncio.run(_main())
