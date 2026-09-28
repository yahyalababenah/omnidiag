"""
OmniDiag — Automated Retraining Pipeline (Feature 4.3)
=======================================================
Pulls annotated samples from the ReviewQueue and retrains — by a route that
depends on the model family declared in the disease config, because the two
families here cannot be retrained the same way.

Usage (CLI):
    python -m backend.active_learning.retrain --disease heart_disease --min-samples 10

API:
    POST /api/v4/admin/retrain  (super_admin only)

Flow, both families:
    1. Query ReviewQueue for reviewed rows with an expert label
    2. Fetch the linked Prediction's features + that label
    3. Branch on `model.family` from configs/<disease>.yaml

    family in CANDIDATE_FAMILIES (heart, glm_ivap_conformal) — Gate 8.10:
    4. Rebuild the whole stack on 920 UCI rows + the reviewed rows, through
       the same stack.build_bundle() the shipped image build calls
    5. Write it to models/heart_disease/candidates/<mlflow_run_id>/
    6. Measure it against the SHIPPED model: conformal coverage, and a decision
       transition matrix over the 920 read from reference_scores.json
    7. Log provenance + aggregates to MLflow. Replace nothing, reload nothing.
       Promotion is a human decision and is not implemented here.

    any other family (diabetes and the legacy XGBoost path) — unchanged:
    4. Retrain XGBoost incrementally (via xgb_model=)
    5. Save new model to disk
    6. Reload the running ModelLoader so new predictions use updated model
    7. Log the run to MLflow (if available)

Why heart is not on the XGBoost path: it has not shipped an XGBoost since Gate
8.1. `retrain_xgb` looked for a filename this disease does not have and built its
design matrix out of raw predict-time strings, so it could only fail; and had it
somehow succeeded, an incrementally-boosted tree model would have been written
over the path a Spline-GLM is loaded from and hot-reloaded into the live process.
The fix is a different route for that family, not a patch to that function.
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


#: Model families whose retraining produces a reviewed CANDIDATE rather than a
#: replacement. Read from the disease config's `model.family`, never branched on
#: the disease name: the name is not what decides which code can train it, and a
#: second module of this family must not land on the XGBoost path by default.
CANDIDATE_FAMILIES = {"glm_ivap_conformal"}


#: Where the disease YAML files live, matching DiseaseRouter's own default.
CONFIGS_DIR = Path(os.getenv("CONFIGS_DIR", "configs"))


def load_disease_config(disease: str) -> Dict[str, Any]:
    """
    Parse `configs/<disease>.yaml`.

    Read from disk rather than taken off the running router: instantiating a
    DiseaseRouter loads every model for every disease, which a retraining job has
    no reason to do, and the CLI has no router at all.
    """
    import yaml

    path = CONFIGS_DIR / f"{disease}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"no config for {disease!r} at {path}")
    with open(path) as handle:
        return yaml.safe_load(handle) or {}


def _model_family(disease: str) -> Optional[str]:
    """`model.family` from the disease config, or None if it cannot be read."""
    try:
        config = load_disease_config(disease)
    except Exception as exc:  # noqa: BLE001 — an unreadable config is not a family
        log.warning("could not read config for %s: %r", disease, exc)
        return None
    return ((config or {}).get("model") or {}).get("family")


def retrain_candidate(
    disease: str,
    features_list: List[Dict[str, Any]],
    labels: List[int],
) -> Dict[str, Any]:
    """
    Retrain the heart Spline-GLM stack as an isolated candidate (Gate 8.10).

    This is the path for a family that cannot be retrained the way `retrain_xgb`
    retrains: the model is a Spline-GLM whose IVAP calibrator and Mondrian
    conformal cells are cross-fitted over the whole training set, so there is no
    incremental update that leaves the decision layer meaning anything. It is
    rebuilt from scratch, by the same `stack.build_bundle()` the shipped build
    calls, on the merged data.

    Nothing here replaces anything. The candidate goes to
    `models/heart_disease/candidates/<mlflow_run_id>/`, the shipped bundle is not
    written, `reference_scores.json` is read and not written, and no loader is
    invalidated -- promotion is a human decision made by reading the run.

    The MLflow run is opened first and fails loudly if MLflow is unavailable: a
    candidate nobody can compare against the shipped model is not worth the disk.
    """
    from backend.heart_glm import candidate as candidate_mod
    from backend.monitoring.mlflow_tracker import start_candidate_run

    try:
        config = load_disease_config(disease)
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "failed",
            "disease": disease,
            "samples_used": len(labels),
            "reason": f"could not load the disease config: {type(exc).__name__}: {exc}",
        }

    version = candidate_mod._config_version(config)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    try:
        run, run_id = start_candidate_run(disease, f"{version}-{stamp}")
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "failed",
            "disease": disease,
            "samples_used": len(labels),
            "reason": f"no MLflow run — candidate not built: {exc}",
        }

    import mlflow

    with run:
        try:
            card = candidate_mod.build_candidate(
                run_id=run_id,
                features_list=features_list,
                labels=labels,
                config=config,
            )
            params, metrics = candidate_mod.mlflow_payload(card)
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            # The full matrix and the per-cell coverage as one artifact, so the
            # run carries the numbers in the shape they were computed in and not
            # only flattened into metric keys. Aggregates only: the merged CSV
            # and the bundle stay on disk, because both carry patient rows.
            mlflow.log_dict(
                {
                    "transition_matrix": card["transition_matrix"],
                    "conformal_coverage": card["conformal_coverage"],
                    "limit": card["limit"],
                },
                "candidate_comparison.json",
            )
        except Exception as exc:  # noqa: BLE001
            mlflow.set_tag("build_failed", f"{type(exc).__name__}: {exc}")
            log.error("candidate build failed for %s: %r", disease, exc)
            return {
                "status": "failed",
                "disease": disease,
                "samples_used": len(labels),
                "mlflow_run_id": run_id,
                "reason": f"{type(exc).__name__}: {exc}",
            }

    matrix = card["transition_matrix"]
    return {
        "status": "success",
        "disease": disease,
        "samples_used": card["rows_from_review_queue"],
        "outcome": "candidate_built_not_promoted",
        "mlflow_run_id": run_id,
        "candidate_dir": card["candidate_dir"],
        "rows_total": card["rows_total"],
        "rows_rejected": card["rows_rejected"],
        "moved_out_of_uncertain": matrix["moved_out_of_uncertain"],
        "moved_into_uncertain": matrix["moved_into_uncertain"],
        "decisions_changed_total": matrix["decisions_changed_total"],
        "limit": card["limit"],
    }


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

    Two outcomes, decided by the model family in the disease config:

      CANDIDATE_FAMILIES  → build an isolated candidate, log the comparison,
                            replace nothing, reload nothing. (heart, Gate 8.10)
      anything else       → the incremental XGBoost path, unchanged.

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

    # Before any training: which kind of model is this? Asked of the config,
    # because the config is what decides which artifact is live -- heart stopped
    # shipping an XGBoost at Gate 8.1 and `retrain_xgb` was never told.
    if _model_family(disease) in CANDIDATE_FAMILIES:
        return await asyncio.to_thread(retrain_candidate, disease, features_list, labels)

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
        from backend.database import AsyncSessionLocal
        async with AsyncSessionLocal() as db:
            result = await run_retrain_pipeline(db, args.disease, args.min_samples)
            print(result)

    asyncio.run(_main())
