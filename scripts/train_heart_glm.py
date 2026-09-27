#!/usr/bin/env python3
"""
Build the deployed heart-disease model bundle from the training CSV.

This runs at Docker image build time (see Dockerfile), so no model binary is
committed to the repository. The build fails loudly rather than shipping a
model whose outputs differ from the ones measured in the research repo.

    python scripts/train_heart_glm.py --verify          # build + check fingerprint
    python scripts/train_heart_glm.py --write-reference # re-record the fingerprint

Checks, declared before the first run (heart/phase8_1_design.md §3):

  1. sha256 of the training CSV equals the recorded value.          absolute
  2. decision and conformal set identical for all 920 patients.     absolute
  3. max |delta p| <= 1e-6 against the recorded scores.

A failure of 1 or 2 is a bug to report, never a tolerance to relax. Numeric
drift in the last digits (3) is expected across BLAS builds -- the isotonic
and lbfgs steps are not bit-reproducible across CPUs -- which is why the
decision, not the probability, is the absolute condition.
"""

from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.heart_glm import stack  # noqa: E402

TRAINING_CSV = PROJECT_ROOT / "data/heart_disease/processed/uci_heart_by_site.csv"
BUNDLE_PATH = PROJECT_ROOT / "models/heart_disease/heart_l3_glm_stack.pkl"
REFERENCE_PATH = PROJECT_ROOT / "backend/heart_glm/reference_scores.json"
BUILD_TOLERANCE = 1e-6


def _experiments_git_ref() -> str:
    """Best-effort git ref of the research repo, recorded in the model card."""
    for candidate in (Path.home() / "omnidiag_experiments",):
        try:
            out = subprocess.run(
                ["git", "-C", str(candidate), "rev-parse", "--short", "HEAD"],
                capture_output=True, text=True, timeout=10,
            )
            if out.returncode == 0:
                return f"omnidiag_experiments@{out.stdout.strip()}"
        except (OSError, subprocess.SubprocessError):
            pass
    return "unavailable at build time"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="check the output fingerprint")
    parser.add_argument("--write-reference", action="store_true", help="re-record the fingerprint")
    parser.add_argument("--tolerance", type=float, default=BUILD_TOLERANCE)
    args = parser.parse_args()

    print(f"training CSV : {TRAINING_CSV}")
    print(f"sha256       : {stack.sha256_of(TRAINING_CSV)}")

    bundle = stack.build_bundle(TRAINING_CSV, experiments_git_ref=_experiments_git_ref())

    verify_result = None
    if args.write_reference:
        REFERENCE_PATH.write_text(
            json.dumps(stack.reference_scores(bundle, TRAINING_CSV), indent=1) + "\n"
        )
        print(f"reference    : written to {REFERENCE_PATH}")
    elif args.verify:
        verify_result = stack.verify_against_reference(
            bundle, TRAINING_CSV, REFERENCE_PATH, args.tolerance
        )
        result = verify_result
        deltas = ", ".join(f"{k} {v:.2e}" for k, v in result["max_abs_delta"].items())
        print(f"fingerprint  : OK — decisions identical for all 920; max |delta| {deltas}")

    BUNDLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BUNDLE_PATH, "wb") as handle:
        pickle.dump(bundle, handle)

    size_kb = BUNDLE_PATH.stat().st_size / 1024
    print(f"bundle       : {BUNDLE_PATH} ({size_kb:.1f} KB)")
    print(f"pkl sha256   : {stack.sha256_of(BUNDLE_PATH)}   (informational — pickle bytes vary by library build)")

    _log_build_run(bundle, size_kb, verify_result)
    return 0


def _log_build_run(bundle, size_kb: float, verify_result) -> None:
    """
    Record this build in MLflow: provenance, not performance.

    The experiment used to be empty because nothing wrote to it -- the only
    automatic caller was a retrain path that fails for this disease. This is the
    moment the shipped artifact comes into existence, so it is the moment worth
    recording (Gate 8.6).

    Deliberately logs no accuracy figure. Every performance number for this model
    is cross-fitted or leave-one-hospital-out and lives in the research
    repository; recomputing one here, in-sample, would put a flattering number
    next to the artifact with nothing to say it is not the headline.
    """
    card = bundle.get("model_card", {})
    params = {
        "family": card.get("family"),
        "model": card.get("model"),
        "feature_set": card.get("feature_set"),
        "training_data": card.get("training_data"),
        "training_csv_sha256": card.get("training_csv_sha256"),
        "split": card.get("split"),
        "calibration": card.get("calibration"),
        "conformal_alpha": bundle.get("alpha"),
        "features_model": ",".join(bundle.get("features_model", [])),
        "unused_input_features": ",".join(bundle.get("unused_input_features", [])),
        "bundle_sha256": stack.sha256_of(BUNDLE_PATH),
        "experiments_git_ref": card.get("experiments_git_ref", ""),
    }
    metrics = {"bundle_size_kb": round(size_kb, 1)}
    if verify_result is not None:
        # The reproducibility fingerprint: evidence that this build reproduced
        # the recorded one, not a claim about how good the model is.
        metrics["fingerprint_decision_mismatches"] = float(verify_result["decision_mismatches"])
        for name, value in verify_result["max_abs_delta"].items():
            metrics[f"fingerprint_max_abs_delta_{name}"] = float(value)
    for field, impact in (bundle.get("blank_impact") or {}).items():
        metrics[f"blank_impact_decisions_changed_{field}"] = float(impact["decision_changed"])

    from backend.monitoring.mlflow_tracker import log_build_artifact

    run_id = log_build_artifact(
        disease="heart_disease",
        model_version=str(card.get("version", "v7.0.0")),
        params={k: v for k, v in params.items() if v is not None},
        metrics=metrics,
        tags={"verified": "yes" if verify_result is not None else "no"},
    )
    print(f"mlflow       : {'run ' + run_id if run_id else 'not logged (see log) — build unaffected'}")


if __name__ == "__main__":
    raise SystemExit(main())
