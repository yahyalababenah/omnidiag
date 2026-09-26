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

from backend.heart_glm import core  # noqa: E402

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
    print(f"sha256       : {core.sha256_of(TRAINING_CSV)}")

    bundle = core.build_bundle(TRAINING_CSV, experiments_git_ref=_experiments_git_ref())

    if args.write_reference:
        REFERENCE_PATH.write_text(
            json.dumps(core.reference_scores(bundle, TRAINING_CSV), indent=1) + "\n"
        )
        print(f"reference    : written to {REFERENCE_PATH}")
    elif args.verify:
        result = core.verify_against_reference(
            bundle, TRAINING_CSV, REFERENCE_PATH, args.tolerance
        )
        deltas = ", ".join(f"{k} {v:.2e}" for k, v in result["max_abs_delta"].items())
        print(f"fingerprint  : OK — decisions identical for all 920; max |delta| {deltas}")

    BUNDLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BUNDLE_PATH, "wb") as handle:
        pickle.dump(bundle, handle)

    size_kb = BUNDLE_PATH.stat().st_size / 1024
    print(f"bundle       : {BUNDLE_PATH} ({size_kb:.1f} KB)")
    print(f"pkl sha256   : {core.sha256_of(BUNDLE_PATH)}   (informational — pickle bytes vary by library build)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
