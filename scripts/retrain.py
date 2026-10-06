#!/usr/bin/env python3
"""
OmniDiag — retrain entry point (Feature 4.3)
=============================================
Triggered by:
  1. Manual call:  python scripts/retrain.py --disease heart_disease
  2. Docker (see docker-compose.yml, retrain profile)

A module whose family is in CANDIDATE_FAMILIES (heart, `glm_ivap_conformal`,
since Gate 8.1) is handed to the candidate builder by run_candidate_path():
it builds an isolated candidate and records the comparison, and nothing is
promoted (Gate 8.10). Every other family, and every retired disease, is refused
with exit code 2 before anything is read, as the API does.

The legacy XGBoost path (load a CSV, retrain, promote over the production
pickle, flush the cache) wrote a file no live module loaded (the W-08 pattern);
it was unreachable after gate B4 and was deleted in B7.

Usage:
    python scripts/retrain.py --disease heart_disease
"""

import argparse
import logging
import sys
from pathlib import Path

# Ensure backend package is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("omnidiag.retrain")


def parse_args():
    p = argparse.ArgumentParser(description="OmniDiag retrain entry point")
    p.add_argument("--disease", required=True, help="Disease key (e.g. heart_disease)")
    return p.parse_args()


def run_candidate_path(disease: str) -> None:
    """
    Delegate to the candidate builder for a family this script cannot train.

    The deleted legacy path assumed one XGBoost on a random split, judged by
    pooled AUC, promoted over the production path when it cleared a tolerance.
    None of that applies to the heart module: its decision is a conformal SET
    rather than a probability against a cut-point, so it has no AUC-versus-
    threshold to promote on; its headline figure is leave-one-hospital-out, so a
    random split would produce a flattering number that selects nothing; and a
    promotion at all is the thing Gate 8.10 removed from this path.

    So this script does not train it. It hands the whole job to
    backend.active_learning.retrain.retrain_candidate, which builds an isolated
    candidate and records the comparison, and exits.
    """
    from backend.active_learning.retrain import load_disease_config, retrain_candidate
    from backend.heart_glm import candidate as candidate_mod

    log.info("%s is a %s model — building a CANDIDATE, promoting nothing",
             disease, load_disease_config(disease).get("model", {}).get("family"))
    log.info("%s", candidate_mod.CANDIDATE_LIMIT)

    # No reviewed rows are read here: this script's own entry point has no
    # database session, and inventing labels from `prediction > threshold`
    # (which is what the deleted legacy path did) would
    # train the model on its own output. A candidate from reviewed rows is built
    # through the admin endpoint or the module CLI, both of which have a session.
    result = retrain_candidate(disease, [], [])
    log.info("Result: %s", result)
    if result.get("status") != "success":
        sys.exit(1)


def main() -> None:
    args = parse_args()
    disease = args.disease
    log.info("=" * 60)
    log.info("OmniDiag Retrain Pipeline — disease=%s", disease)

    # 0. Families that are not retrained by this script at all.
    try:
        from backend.active_learning.retrain import CANDIDATE_FAMILIES, _model_family
        family = _model_family(disease)
    except Exception as exc:  # noqa: BLE001
        log.error("Cannot determine the model family for %s: %r", disease, exc)
        sys.exit(1)
    if family in CANDIDATE_FAMILIES:
        run_candidate_path(disease)
        return
    # Gate B4: every other family is refused before anything is read, as the API does.
    from backend.retired_diseases import is_retired
    reason = ("retired (backend/retired_diseases.py)" if is_retired(disease)
              else f"retrain not yet supported for {family}")
    log.error("unsupported — %s: %s", disease, reason)
    sys.exit(2)


if __name__ == "__main__":
    main()
