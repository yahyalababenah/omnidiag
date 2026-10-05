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

    any other family — refused, before a single sample is read:
    `status: "unsupported"`, reason "retrain not yet supported for <family>".
    Gate B4 removed the incremental-XGBoost writer (`retrain_xgb`) that used to
    serve every other family. No live module read what it wrote: BRFSS diabetes
    is retired, the NHANES module has no XGBoost file, and the heart revert path
    (sklearn_pipeline) loads a different file and fed it raw strings (W-26). A writer whose output nothing reads is the W-08 hazard, so it went.

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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("omnidiag.retrain")

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


async def run_retrain_pipeline(
    db,
    disease: str,
    min_samples: int = 5,
) -> Dict[str, Any]:
    """
    Full pipeline: refuse what cannot be retrained, fetch annotated samples,
    build a candidate.

      retired disease     -> 410 (backend/retired_diseases.py)
      CANDIDATE_FAMILIES  -> build an isolated candidate, log the comparison,
                             replace nothing, reload nothing. (heart, Gate 8.10)
      any other family    -> {"status": "unsupported", "reason": ...}; nothing
                             is read and nothing is written (gate B4)

    Returns a status dict suitable for the API response.
    """
    # Before anything is read (W-08): a retired disease has no config.
    from backend.retired_diseases import reject_if_retired
    reject_if_retired(disease)

    # The family decides, and it is asked before the samples are fetched, so an
    # unsupported family never reads the review queue at all.
    family = _model_family(disease)
    if family not in CANDIDATE_FAMILIES:
        return {
            "status": "unsupported",
            "disease": disease,
            "samples_used": 0,
            "reason": f"retrain not yet supported for {family}",
        }

    features_list, labels = await get_annotated_samples(db, disease, min_samples)

    if not features_list:
        return {
            "status": "skipped",
            "disease": disease,
            "reason": f"Fewer than {min_samples} annotated samples available",
            "samples_used": 0,
        }

    return await asyncio.to_thread(retrain_candidate, disease, features_list, labels)


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
