#!/usr/bin/env python3
"""
Build the frozen drift reference profile for each disease.

WHY A PROFILE AND NOT THE CSV AT RUNTIME
----------------------------------------
1. PSI needs bin edges that belong to the reference. Re-binning against each
   incoming sample lets the reference move with the present, and a PSI computed
   that way is not a number that means anything.
2. A reference read from a CSV changes whenever the CSV changes, silently. The
   profile carries the source's sha256, so a changed source is an explicit
   failure rather than drift that appears out of nowhere.
3. The diabetes CSV is 6.3 MB. Reading it per drift run to recompute a fixed
   distribution is waste.

The profile is committed, and rebuilt with --verify at image build time: if the
regenerated profile differs from the committed one, the build fails. Same shape
as scripts/regen_heart_importance.py (the F0-1 pattern).

HEART: THE 920 AS TRAINED, IN THE SPACE THE MODEL READS
-------------------------------------------------------
The profile is built from `encode_for_training(...)`, not from the raw CSV
columns, and this is not a detail. The chest-pain coding is INVERTED between the
two entry points:

    training  CP_MAP_UCI_RAW  = {ASY: 3, NAP: 2, ATA: 1, TA: 0}
    inference CP_MAP_CLINICAL = {TA: 3, ATA: 2, NAP: 1, ASY: 0}

So a reference built on the raw string `ChestPainType` and compared against API
rows would be comparing opposite meanings, and would report confident drift on
a population that had not moved at all. In encoded space both sides are
`cp_anginal`, and the comparison is between like and like.

Only the seven inputs the model reads are monitored. The other four are profiled
and reported separately: blanking any of them changes the decision for exactly
zero of the 920 patients (Gate 8.3), so letting them raise the drift share would
raise alarms nothing can act on.

DIABETES: REWEIGHTED TO THE DECLARED DEPLOYMENT PREVALENCE
----------------------------------------------------------
The only BRFSS file in this repository is the 50/50 balanced sample, and a drift
reference built on it compares real people against a sample balanced by
construction -- every feature correlated with diabetes reads as drifted.

The unbalanced file is not here, and no new data is downloaded for this. Instead
each feature's reference distribution is reweighted to the prevalence the model
already deploys against:

    P_ref(x) = p * P(x | y=1) + (1 - p) * P(x | y=0),  p = prevalence_deploy

p comes from configs/diabetes.yaml (`prevalence_deploy`), the same number
predict() already corrects its probabilities onto, so the drift reference and
the probability scale end up on one prevalence.

DECLARED LIMIT: that number is Jordan's diabetes prevalence, not BRFSS 2015's own
(~14 %). The reference is reweighted to the prevalence the model is DEPLOYED
against, which is a choice, and it is recorded in the profile itself.

Nothing else in the diabetes path is touched: not the model, not the threshold
(0.108184), not the prevalence correction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.monitoring.drift_stats import (  # noqa: E402
    DEFAULT_ALPHA, MIN_CURRENT_ROWS, PSI_MAJOR, PSI_MINOR,
    bin_counts, quantile_edges,
)

HEART_CSV = ROOT / "data/heart_disease/processed/uci_heart_by_site.csv"
DIABETES_CSV = ROOT / "data/diabetes/raw/diabetes_binary_5050split_health_indicators_BRFSS2015.csv"
N_BINS = 10


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def numeric_profile(values: pd.Series) -> Dict[str, Any]:
    edges = quantile_edges(values.tolist(), N_BINS)
    counts = bin_counts(values.tolist(), edges)
    # The reference SAMPLE, run-length encoded. PSI needs the bins; KS needs the
    # sample, and every numeric feature here has at most 216 distinct values, so
    # this is exact rather than an approximated CDF.
    finite = values.dropna()
    uniq, cnt = np.unique(finite.to_numpy(dtype=float), return_counts=True)
    return {"kind": "numeric", "edges": [None if not np.isfinite(e) else float(e) for e in edges],
            "counts": counts, "n": int(len(values)),
            "n_missing": int(values.isna().sum()),
            "ecdf": {"values": [float(x) for x in uniq], "counts": [int(x) for x in cnt]},
            "note": "the last bin counts missing values; `ecdf` is the exact reference sample, "
                    "run-length encoded, for the KS test"}


def categorical_profile(values: pd.Series) -> Dict[str, Any]:
    cats = Counter("__missing__" if pd.isna(v) else str(v) for v in values)
    order = sorted(cats)
    return {"kind": "categorical", "categories": order,
            "counts": [cats[c] for c in order], "n": int(len(values)),
            "n_missing": int(cats.get("__missing__", 0))}


def heart_profile(csv_path: Optional[Path] = None) -> Dict[str, Any]:
    from backend.heart_glm.stack import (
        CP_MAP_CLINICAL, CP_MAP_UCI_RAW, MODEL_FEATURES, encode_for_training,
    )
    csv_path = csv_path or HEART_CSV
    frame = pd.read_csv(csv_path)
    encoded = encode_for_training(frame)

    # `cp_anginal` is a 0-3 count and `Sex_m` is 0/1: both are categorical here.
    # A KS test on four ordered levels answers a question nobody asked.
    kinds = {"Age": "numeric", "RestingBP": "numeric", "Cholesterol": "numeric",
             "Sex_m": "categorical", "cp_anginal": "categorical",
             "FastingBS_cat": "categorical", "RestingECG": "categorical"}
    assert set(kinds) == set(MODEL_FEATURES), "monitored set must be exactly the model's features"

    monitored = {
        f: (numeric_profile(encoded[f]) if kinds[f] == "numeric"
            else categorical_profile(encoded[f]))
        for f in MODEL_FEATURES
    }
    # Profiled but deliberately outside drift_share (Gate 8.3: zero decisions move)
    unread = {}
    for f in ("MaxHR", "Oldpeak", "ExerciseAngina", "ST_Slope"):
        col = frame[f]
        unread[f] = (numeric_profile(pd.to_numeric(col, errors="coerce"))
                     if f in ("MaxHR", "Oldpeak") else categorical_profile(col))
    return {
        "disease": "heart_disease",
        "space": "encoded — backend.heart_glm.stack.encode_for_training / encode_for_inference",
        "space_reason": ("the chest-pain coding is inverted between training and the API "
                         f"(training {CP_MAP_UCI_RAW}, inference {CP_MAP_CLINICAL}), so a "
                         "comparison on the raw string would compare opposite meanings"),
        "source": {"path": str(csv_path.relative_to(ROOT)) if ROOT in csv_path.parents
                          else str(csv_path),
                   "sha256": sha256(csv_path), "rows": int(len(frame)),
                   "note": "the 920 rows the shipped model is fitted on, same file and same "
                           "sha256 as the build run records"},
        "reweighting": None,
        "monitored": monitored,
        "profiled_but_not_monitored": unread,
        "not_monitored_reason": ("the model does not read these four; blanking any of them "
                                "changes the decision for 0 of 920 patients (Gate 8.3), so "
                                "they cannot move a decision and must not move drift_share"),
    }


def diabetes_profile(csv_path: Optional[Path] = None) -> Dict[str, Any]:
    cfg = yaml.safe_load(open(ROOT / "configs/diabetes.yaml"))
    model_cfg = cfg.get("model", cfg)
    p_deploy = _find_key(cfg, "prevalence_deploy")
    p_train = _find_key(cfg, "prevalence_train")
    if p_deploy is None or p_train is None:
        raise SystemExit("prevalence_deploy / prevalence_train not found in configs/diabetes.yaml")

    csv_path = csv_path or DIABETES_CSV
    frame = pd.read_csv(csv_path)
    target = "Diabetes_binary"
    feats = cfg["features"]
    binary, numerical = list(feats["binary_columns"]), list(feats["numerical_columns"])

    pos = frame[frame[target] == 1]
    neg = frame[frame[target] == 0]
    n_total = len(frame)
    monitored: Dict[str, Any] = {}

    for f in numerical:
        # Edges from the pooled sample: the bins must cover both classes' ranges.
        edges = quantile_edges(frame[f].tolist(), N_BINS)
        c_pos = np.asarray(bin_counts(pos[f].tolist(), edges), dtype=float)
        c_neg = np.asarray(bin_counts(neg[f].tolist(), edges), dtype=float)
        mixed = _reweight(c_pos, c_neg, p_deploy, n_total)
        # The KS reference sample is reweighted the same way, on the feature's own
        # distinct values: same people, reweighted to the deployment prevalence.
        uniq = np.unique(frame[f].dropna().to_numpy(dtype=float))
        u_pos = np.array([int((pos[f] == v).sum()) for v in uniq], dtype=float)
        u_neg = np.array([int((neg[f] == v).sum()) for v in uniq], dtype=float)
        u_mixed = _reweight(u_pos, u_neg, p_deploy, n_total)
        keep = [(v, c) for v, c in zip(uniq, u_mixed) if c > 0]
        monitored[f] = {"kind": "numeric",
                        "edges": [None if not np.isfinite(e) else float(e) for e in edges],
                        "counts": mixed, "n": int(sum(mixed)), "n_missing": 0,
                        "ecdf": {"values": [float(v) for v, _ in keep],
                                 "counts": [int(c) for _, c in keep]},
                        "note": "counts and ecdf both reweighted to prevalence_deploy; "
                                "last bin is missing"}
    for f in binary:
        cats = sorted({str(v) for v in frame[f].dropna().unique()})
        c_pos = np.asarray([int((pos[f].astype(str) == c).sum()) for c in cats], dtype=float)
        c_neg = np.asarray([int((neg[f].astype(str) == c).sum()) for c in cats], dtype=float)
        mixed = _reweight(c_pos, c_neg, p_deploy, n_total)
        monitored[f] = {"kind": "categorical", "categories": cats, "counts": mixed,
                        "n": int(sum(mixed)), "n_missing": 0,
                        "note": "counts reweighted to prevalence_deploy"}

    return {
        "disease": "diabetes",
        "space": "raw BRFSS columns (no encoding inversion exists for this module)",
        "source": {"path": str(csv_path.relative_to(ROOT)) if ROOT in csv_path.parents
                          else str(csv_path),
                   "sha256": sha256(csv_path), "rows": int(n_total),
                   "note": "the 50/50 BALANCED sample — the only BRFSS file in this repository"},
        "reweighting": {
            "applied": True,
            "formula": "P_ref(x) = p * P(x | y=1) + (1 - p) * P(x | y=0)",
            "prevalence_of_sample": float(p_train),
            "prevalence_of_reference": float(p_deploy),
            "prevalence_source": "configs/diabetes.yaml: prevalence_deploy",
            "why": ("a reference built on the 50/50 sample compares real people against a "
                    "sample balanced by construction, so every feature correlated with "
                    "diabetes reads as drifted"),
            "declared_limit": (f"{p_deploy} is Jordan's diabetes prevalence, which is what this "
                               "model deploys against. It is NOT BRFSS 2015's own prevalence "
                               "(~14%). The unbalanced BRFSS file is not in this repository and "
                               "no new data was downloaded for this."),
            "unchanged": "the diabetes model, its threshold (0.108184) and its prevalence "
                         "correction are untouched; this is a monitoring reference only",
        },
        "monitored": monitored,
        "profiled_but_not_monitored": {},
        "not_monitored_reason": "",
    }


def _reweight(c_pos: np.ndarray, c_neg: np.ndarray, p: float, n_total: int) -> List[int]:
    """Mix two class-conditional count vectors at prevalence p, keeping the total n.

    Kept at the original n so the reference stays a distribution of a stated size
    rather than an arbitrary one: it is the same people, reweighted.
    """
    pp = c_pos / c_pos.sum() if c_pos.sum() else np.zeros_like(c_pos)
    pn = c_neg / c_neg.sum() if c_neg.sum() else np.zeros_like(c_neg)
    mixed = (p * pp + (1 - p) * pn) * n_total
    out = np.floor(mixed).astype(int)
    # Hand the rounding remainder to the largest fractions so the total is exact.
    short = int(n_total - out.sum())
    if short > 0:
        for i in np.argsort(-(mixed - out))[:short]:
            out[i] += 1
    return out.tolist()


def _find_key(node: Any, key: str) -> Any:
    if isinstance(node, dict):
        if key in node:
            return node[key]
        for v in node.values():
            found = _find_key(v, key)
            if found is not None:
                return found
    return None


def canonical(profile: Dict[str, Any]) -> str:
    """The profile without its timestamp, for comparing two builds."""
    return json.dumps({k: v for k, v in profile.items() if k != "built_at"},
                      sort_keys=True, separators=(",", ":"))


def _rule() -> Dict[str, Any]:
    return {"alpha": DEFAULT_ALPHA, "psi_minor": PSI_MINOR, "psi_major": PSI_MAJOR,
            "min_current_rows": MIN_CURRENT_ROWS,
            "flag": "Holm-corrected p < alpha AND PSI >= psi_major",
            "not_measured": "output/label drift — no follow-up catheterisation result exists "
                            "for any patient in this system"}


def build_profiles(sources: Optional[Dict[str, Path]] = None) -> Dict[str, Dict[str, Any]]:
    """Build both profiles. `sources` lets a test point at a tampered CSV without
    touching the real one -- see verify_profiles() and its test."""
    src = sources or {}
    built = {"heart_disease": heart_profile(src.get("heart_disease")),
             "diabetes": diabetes_profile(src.get("diabetes"))}
    rule = _rule()
    for profile in built.values():
        profile["rule"] = rule
        profile["built_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return built


def verify_profiles(built: Dict[str, Dict[str, Any]],
                    profile_paths: Optional[Dict[str, Path]] = None) -> List[str]:
    """Compare freshly built profiles against the committed ones.

    Returns the list of failure messages -- empty means everything matched. This
    is what --verify calls, and what its own test calls directly against a
    tampered source: the falsifying case for "the reference is anchored to this
    exact training data" (reviewer protocol rule 7) is that a changed CSV without
    a rebuilt profile must fail here, not just that a matching one passes.
    """
    failures = []
    for disease, profile in built.items():
        out = (profile_paths or {}).get(
            disease, ROOT / "models" / disease / "drift_reference.json")
        if not out.exists():
            failures.append(f"{disease}: {out} is missing")
            continue
        committed = json.loads(out.read_text())
        if canonical(committed) != canonical(profile):
            failures.append(f"{disease}: the rebuilt profile differs from the committed one ({out})")
    return failures


def write_profiles(built: Dict[str, Dict[str, Any]],
                   profile_paths: Optional[Dict[str, Path]] = None) -> None:
    for disease, profile in built.items():
        out = (profile_paths or {}).get(
            disease, ROOT / "models" / disease / "drift_reference.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(profile, indent=1) + "\n")
        print(f"{disease:14s}: wrote {out.relative_to(ROOT) if ROOT in out.parents else out} "
              f"({len(profile['monitored'])} monitored features)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true",
                    help="rebuild and compare against the committed profiles; "
                         "exit 1 on any difference")
    args = ap.parse_args()

    built = build_profiles()
    if args.verify:
        failures = verify_profiles(built)
        for disease, profile in built.items():
            if not any(disease in f for f in failures):
                print(f"{disease:14s}: profile matches "
                      f"({len(profile['monitored'])} monitored features, "
                      f"source sha256 {profile['source']['sha256'][:12]}…)")
        if failures:
            print("\nFAILED:", file=sys.stderr)
            for f in failures:
                print(" -", f, file=sys.stderr)
            return 1
        return 0
    write_profiles(built)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
