"""
Regenerate the heart feature-importance file FROM THE SHIPPED BUNDLE.

Why this exists (F0-1): evaluation_evidence/heart/shap_importance.json describes
a model that was never deployed. It lists Oldpeak, ExerciseAngina, MaxHR and
ST_Slope -- four features the shipped model does not read at all -- and the
config's high_impact_features warning list was picked from it. An importance
file for a different model is worse than none: it is a wrong answer with an
evidence path behind it.

What this computes: mean |phi| over every row of the training file, where phi is
the exact additive SHAP value of the shipped Spline-GLM on the raw log-odds
score (stack.shap_log_odds). The GLM is additive on the logit scale, so these
values are exact, not sampled -- rerunning gives the same numbers bit for bit.

The declared scale limit, carried in the file itself: this ranks features by
their effect on the RAW score. It is NOT an attribution of the IVAP-calibrated
probability the clinician sees, because IVAP is monotone but not linear, so no
additive attribution of that probability exists (the same limit /explain
declares as shap_scale).

Usage:
    python scripts/regen_heart_importance.py            # write the file
    python scripts/regen_heart_importance.py --verify   # recompute and diff
"""
from __future__ import annotations

import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from backend.heart_glm import stack  # noqa: E402

BUNDLE = REPO / "models/heart_disease/heart_l3_glm_stack.pkl"
TRAINING_CSV = REPO / "data/heart_disease/processed/uci_heart_by_site.csv"
OUT = REPO / "evaluation_evidence/heart/heart_l3_glm_importance.json"


def compute() -> dict:
    bundle = pickle.load(open(BUNDLE, "rb"))
    frame = pd.read_csv(TRAINING_CSV)
    encoded = stack.encode_for_training(frame)

    # shap_log_odds explains one row at a time; the per-feature values are a
    # linear function of the design row, so the loop is exact and cheap.
    values = np.array([
        stack.shap_log_odds(bundle, encoded.iloc[[i]])[0] for i in range(len(encoded))
    ])
    mean_abs = np.abs(values).mean(axis=0)

    ranked = sorted(
        ((f, round(float(v), 6)) for f, v in zip(stack.MODEL_FEATURES, mean_abs)),
        key=lambda kv: -kv[1],
    )
    return {
        "model": "heart_l3_glm_stack (Spline-GLM + Venn-Abers + Mondrian conformal)",
        "generated_by": "scripts/regen_heart_importance.py",
        "training_csv_sha256": hashlib.sha256(TRAINING_CSV.read_bytes()).hexdigest(),
        "n_rows": int(len(frame)),
        "scale": "mean_abs_shap_log_odds_raw_score",
        "scale_limit": (
            "Ranks features by their effect on the model's RAW log-odds score. This is "
            "NOT an attribution of the IVAP-calibrated probability shown to a clinician: "
            "IVAP is monotone but not linear, so no additive attribution of that "
            "probability exists. Same limit as /explain's shap_scale."
        ),
        "exact": True,
        "supersedes": (
            "evaluation_evidence/heart/shap_importance.json, which describes a model that "
            "was never deployed (F0-1) and lists four features this model does not read."
        ),
        "importance": dict(ranked),
        "rank": [f for f, _ in ranked],
    }


def main() -> int:
    fresh = compute()
    if "--verify" in sys.argv:
        if not OUT.exists():
            print(f"MISSING: {OUT.relative_to(REPO)} -- run without --verify first")
            return 1
        shipped = json.load(open(OUT))
        same_rank = shipped.get("rank") == fresh["rank"]
        print(f"rank shipped   : {shipped.get('rank')}")
        print(f"rank recomputed: {fresh['rank']}")
        deltas = {
            f: round(abs(shipped["importance"].get(f, 0.0) - v), 8)
            for f, v in fresh["importance"].items()
        }
        worst = max(deltas.values()) if deltas else 0.0
        print(f"max |delta| in mean|phi| : {worst:.2e}")
        print(f"training CSV sha256 match: {shipped.get('training_csv_sha256') == fresh['training_csv_sha256']}")
        if not same_rank or worst > 1e-6:
            print("MISMATCH -- the shipped importance file does not match the shipped bundle")
            return 1
        print("OK -- importance file matches the bundle it claims to describe")
        return 0

    OUT.write_text(json.dumps(fresh, indent=1) + "\n")
    print(f"wrote {OUT.relative_to(REPO)}")
    for feature, value in fresh["importance"].items():
        print(f"  {feature:<16} {value:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
