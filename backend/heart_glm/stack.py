"""
OmniDiag — Heart CAD Spline-GLM stack (research-side model, Phase 8 Gate 8.1)
=============================================================================
The deployed heart model is a Spline-GLM on the L3 feature set, wrapped in two
layers that are part of the artifact, not post-processing:

    raw score   -> IVAP (inductive Venn-Abers)  -> calibrated probability + [p_lower, p_upper]
    raw score   -> Mondrian conformal by (sex x class), alpha = 0.10
                                                -> {referral, no_referral, uncertain}

Every constant here is transcribed from the research repo's Phase 4-7 scripts
(`p4_common.py`, `p6_common.py`, `p7_final.py`), and `build_bundle()` reproduces
`work/p7_final/heart_l3_glm_stack.pkl` numerically. The decisions behind them:

  D-25  L3 (7 pre-stress-test inputs). Stress-test features may add 3-5 AUC
        points that four hospitals could not establish.
  D-26  A blank FastingBS is NaN and is mode-imputed inside the fold. There is
        no "Missing" category -- that category was a Zurich site marker and
        raised risk by 9-24 points for a blank field (HF-11).
  D-30  Spline-GLM chosen as a DECLARED DEVIATION: the pre-registered selection
        rule produced no candidate (diseased coverage at Hungarian 0.834 < 0.85
        for all three finalists -- a property of the hospitals, not the family).
  D-33  Mondrian conformal by (sex x class) rather than one threshold, because
        one threshold gave women sensitivity 0.74 against 0.94 for men (HF-13).

── The two chest-pain maps ──────────────────────────────────────────────────
UCI's `cp` column is inverted with respect to its own documentation: the code
"ASY" marks patients with all three anginal features and "TA" marks patients
with none (Gate 2, confirmed against the 76-column raw files by two anchors in
three sites independently of the code and of the label). The API keeps the
CLINICAL meaning of its codes, so exactly two maps exist and they are never
mixed:

    CP_MAP_UCI_RAW    raw UCI rows -> anginal-feature count. TRAINING ONLY.
    CP_MAP_CLINICAL   API input    -> anginal-feature count. INFERENCE ONLY.

`encode_for_training()` reads only the first; `encode_for_inference()` reads
only the second. A test asserts that at source level (tests/test_heart_glm_backend.py).

── What this model does NOT promise (D-32, HF-15) ───────────────────────────
Calibration does not transport between hospitals (intercept -0.67 at Hungarian,
+2.37 at Zurich), and the conformal guarantee is marginal over the training
hospitals' mix -- it broke per site in Gate 6 (diseased women at Hungarian:
0.58). A displayed probability is calibrated to the TRAINING MIX, not to the
hospital reading it.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401  (enables IterativeImputer)
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

# ── Feature contract ─────────────────────────────────────────────────────────

#: API fields the model actually reads.
INPUT_FEATURES: List[str] = [
    "Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol", "FastingBS", "RestingECG",
]

#: API fields accepted by the schema that this model does NOT read (L3, D-25).
UNUSED_INPUT_FEATURES: List[str] = ["MaxHR", "Oldpeak", "ExerciseAngina", "ST_Slope"]

#: The inputs HeartDiseaseInput accepts as Optional, i.e. the ones that can
#: arrive blank and be imputed. Which of these deserve a clinician-facing
#: warning is not decided here and not hand-listed anywhere: it is derived from
#: `blank_impact`, measured into the bundle at build time (Gate 8.3).
OPTIONAL_INPUT_FEATURES: List[str] = [
    "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak", "ST_Slope",
]

#: Encoded columns, in the order the ColumnTransformer expects.
MODEL_FEATURES: List[str] = [
    "Age", "RestingBP", "Cholesterol", "Sex_m", "cp_anginal", "FastingBS_cat", "RestingECG",
]

_NUMERIC = ["Age", "RestingBP", "Cholesterol"]
_BINARY = ["Sex_m"]
_CATEGORICAL = ["cp_anginal", "FastingBS_cat", "RestingECG"]

#: Clinical meaning -> number of anginal features. INFERENCE ONLY.
CP_MAP_CLINICAL: Dict[str, float] = {"TA": 3.0, "ATA": 2.0, "NAP": 1.0, "ASY": 0.0}

#: Raw UCI code -> number of anginal features (the inverted coding). TRAINING ONLY.
CP_MAP_UCI_RAW: Dict[str, float] = {"ASY": 3.0, "NAP": 2.0, "ATA": 1.0, "TA": 0.0}

#: Raw UCI code -> the CLINICAL code that means the same thing. DERIVED from the
#: two maps above by matching anginal-feature count, never written out by hand:
#: a hand-written third map is a third place for HF-1 to come back. Used only to
#: translate a declared raw-coded batch upload into ordinary API input, so that
#: encode_for_inference() stays the one and only path that scores anything.
CP_RAW_TO_CLINICAL: Dict[str, str] = {
    raw_code: clinical_code
    for raw_code, count in CP_MAP_UCI_RAW.items()
    for clinical_code, clinical_count in CP_MAP_CLINICAL.items()
    if clinical_count == count
}

# The derivation above matches on anginal-feature count, which is only sound
# while the count is a UNIQUE key in both maps. If a count were ever repeated,
# the comprehension would drop a code silently and leave a three-entry map that
# still satisfies "count-preserving" and "own inverse" for the codes it kept --
# the batch path would then translate one code wrongly and no test would say so.
# So this is checked at import, not only in the tests: a module that cannot
# build this map correctly must refuse to load, because the alternative is a
# process that starts up and mistranslates chest pain.
if len(set(CP_MAP_UCI_RAW.values())) != len(CP_MAP_UCI_RAW):
    raise ImportError(f"CP_MAP_UCI_RAW has a repeated anginal-feature count: {CP_MAP_UCI_RAW}")
if len(set(CP_MAP_CLINICAL.values())) != len(CP_MAP_CLINICAL):
    raise ImportError(f"CP_MAP_CLINICAL has a repeated anginal-feature count: {CP_MAP_CLINICAL}")
if len(CP_RAW_TO_CLINICAL) != len(CP_MAP_UCI_RAW):
    raise ImportError(
        f"CP_RAW_TO_CLINICAL lost a code in derivation: {CP_RAW_TO_CLINICAL} "
        f"from {CP_MAP_UCI_RAW} / {CP_MAP_CLINICAL}"
    )
if set(CP_RAW_TO_CLINICAL) != set(CP_RAW_TO_CLINICAL.values()):
    raise ImportError(
        f"CP_RAW_TO_CLINICAL is not a permutation of the same four codes: {CP_RAW_TO_CLINICAL}"
    )

#: Human-readable label per anginal-feature count, for explanations.
CP_LABEL: Dict[float, str] = {
    3.0: "typical angina", 2.0: "atypical angina",
    1.0: "non-anginal pain", 0.0: "no anginal features",
}

ALPHA = 0.10           # Mondrian conformal miscoverage per (sex x class) cell
GLM_C = 0.1            # LogisticRegression C, tuned in Phase 4 on L3
N_KNOTS = 4
CV_SEED = 42
N_FOLDS = 5

DECISION_REFERRAL = "referral"
DECISION_NO_REFERRAL = "no_referral"
DECISION_UNCERTAIN = "uncertain"

#: Shown instead of Positive/Negative when the conformal set is not a singleton.
UNCERTAIN_DIAGNOSIS = "Uncertain — refer for further evaluation"


# ── Encoding ─────────────────────────────────────────────────────────────────

def _encode(frame: pd.DataFrame, cp_anginal: pd.Series) -> pd.DataFrame:
    """Shared encoding. The caller supplies the already-mapped anginal count,
    so that neither chest-pain map is reachable from both entry points."""
    out = pd.DataFrame(
        {
            "Age": pd.to_numeric(frame["Age"], errors="coerce").astype(float),
            "RestingBP": pd.to_numeric(frame["RestingBP"], errors="coerce").astype(float),
            "Cholesterol": pd.to_numeric(frame["Cholesterol"], errors="coerce").astype(float),
            "Sex_m": frame["Sex"].map({"M": 1.0, "F": 0.0}),
            "cp_anginal": cp_anginal.astype(float),
            # D-26: a blank fasting blood sugar is NaN, mode-imputed in the
            # pipeline. Never its own category.
            "FastingBS_cat": frame["FastingBS"].map({0: "0", 1: "1", 0.0: "0", 1.0: "1"}).astype(object),
            "RestingECG": frame["RestingECG"].astype(object),
        },
        index=frame.index,
    )
    return out[MODEL_FEATURES]


def encode_for_training(frame: pd.DataFrame) -> pd.DataFrame:
    """Encode RAW UCI rows. Uses CP_MAP_UCI_RAW — the inverted coding — only."""
    cp = frame["ChestPainType"].map(CP_MAP_UCI_RAW)
    if cp.isna().any():
        raise ValueError("unmapped ChestPainType value in the UCI training data")
    return _encode(frame, cp)


def encode_for_inference(records: Sequence[Mapping[str, Any]] | pd.DataFrame) -> pd.DataFrame:
    """Encode API input. Uses CP_MAP_CLINICAL — the clinical meaning — only."""
    frame = records if isinstance(records, pd.DataFrame) else pd.DataFrame(list(records))
    cp = frame["ChestPainType"].map(CP_MAP_CLINICAL)
    if cp.isna().any():
        raise ValueError(
            f"ChestPainType must be one of {sorted(CP_MAP_CLINICAL)} "
            f"(clinical meaning: TA = typical angina)"
        )
    return _encode(frame, cp)


def translate_raw_codes(records: Sequence[Mapping[str, Any]]) -> list[dict]:
    """
    Rewrite ChestPainType from the raw UCI coding into the clinical coding.

    For a batch upload that DECLARES it carries raw UCI codes (Gate 8.2). The
    rows then mean what the API means by them, so validation and
    encode_for_inference() handle them like any other upload -- there is no
    second scoring path to keep in step with the first.

    Rows with no ChestPainType, or an unrecognised one, are passed through
    untouched: per-row validation is what reports them, one row at a time, and
    a translation step must not turn one bad cell into a failed batch.
    """
    out = []
    for record in records:
        row = dict(record)
        code = row.get("ChestPainType")
        if isinstance(code, str) and code in CP_RAW_TO_CLINICAL:
            row["ChestPainType"] = CP_RAW_TO_CLINICAL[code]
        out.append(row)
    return out


# ── Model ────────────────────────────────────────────────────────────────────

def build_pipeline() -> Pipeline:
    """Spline-GLM exactly as tuned in Phase 4 and shipped in Phase 7."""
    prep = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("imp", IterativeImputer(random_state=42, max_iter=10)),
                        ("sc", StandardScaler()),
                        ("sp", SplineTransformer(n_knots=N_KNOTS, degree=3, include_bias=False)),
                    ]
                ),
                _NUMERIC,
            ),
            ("bin", SimpleImputer(strategy="most_frequent"), _BINARY),
            (
                "cat",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="most_frequent")),
                        ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="first")),
                    ]
                ),
                _CATEGORICAL,
            ),
        ],
        verbose_feature_names_out=False,
    )
    return Pipeline([("prep", prep), ("lr", LogisticRegression(C=GLM_C, max_iter=5000))])


# ── Calibration and conformal layers ─────────────────────────────────────────

def ivap(cal_scores: np.ndarray, cal_labels: np.ndarray, scores: np.ndarray):
    """Exact inductive Venn-Abers. Returns (p, p_lower, p_upper).

    Verbatim from the research repo (`p6_common.ivap`): for each test score, an
    isotonic fit on the calibration set plus (s, 0) gives p0 and plus (s, 1)
    gives p1; the reported probability is p1 / (1 - p0 + p1). The width
    p1 - p0 is the calibration set's own uncertainty about this patient and is
    reported, never hidden.
    """
    cal_scores = np.asarray(cal_scores, dtype=float)
    cal_labels = np.asarray(cal_labels, dtype=float)
    scores = np.atleast_1d(np.asarray(scores, dtype=float))
    p0 = np.empty(len(scores))
    p1 = np.empty(len(scores))
    for i, s in enumerate(scores):
        xs = np.r_[cal_scores, s]
        for label, out in ((0.0, p0), (1.0, p1)):
            fit = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(
                xs, np.r_[cal_labels, label]
            )
            out[i] = fit.predict([s])[0]
    return p1 / (1 - p0 + p1), p0, p1


def conformal_quantile(nonconformity: np.ndarray, alpha: float = ALPHA) -> float:
    """The (1-alpha) conformal quantile; inf when the cell is too small to give one."""
    a = np.sort(np.asarray(nonconformity, dtype=float))
    n = len(a)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return float("inf") if k > n else float(a[k - 1])


def conformal_set(score: float, sex: str, cells: Mapping[Any, float]) -> Tuple[bool, bool]:
    """Membership of (class 1, class 0) in the prediction set for this patient."""
    in1 = (1.0 - score) <= _cell(cells, sex, 1)
    in0 = score <= _cell(cells, sex, 0)
    return bool(in1), bool(in0)


def decide(score: float, sex: str, cells: Mapping[Any, float]) -> Tuple[str, List[int]]:
    """Mondrian decision. {1} refer, {0} do not, {0,1} or {} -> uncertain.

    An empty set means the score is atypical for BOTH classes, and a both-class
    set means it is typical for both; neither is a decision, so both report
    `uncertain` (D-33). The cost is declared, not hidden: 38-42% of patients
    land here, which is what it costs to keep the per-cell guarantee instead of
    guessing.
    """
    in1, in0 = conformal_set(score, sex, cells)
    if in1 and not in0:
        return DECISION_REFERRAL, [1]
    if in0 and not in1:
        return DECISION_NO_REFERRAL, [0]
    return DECISION_UNCERTAIN, ([0, 1] if in0 and in1 else [])


def _cell(cells: Mapping[Any, float], sex: str, klass: int) -> float:
    """Cell lookup that survives a JSON round-trip of the (sex, class) key."""
    for key in ((sex, klass), f"{sex}{klass}", f"{sex}|{klass}"):
        if key in cells:
            value = cells[key]
            return float("inf") if value is None else float(value)
    raise KeyError(f"no conformal cell for sex={sex!r} class={klass}")


# ── SHAP on the raw log-odds score ───────────────────────────────────────────

def shap_log_odds(bundle: Mapping[str, Any], encoded: pd.DataFrame) -> Tuple[np.ndarray, float]:
    """Exact interventional SHAP for one patient, in log-odds of the RAW score.

    The GLM is additive on the logit scale, so with an independent background
    the Shapley value of a feature is exactly its own centred contribution:

        phi_f = sum_{j in basis(f)} w_j * (x_j - mean_j)
        logit(raw score) = base + sum_f phi_f      (exact, no residual)

    This explains the RAW score. It does NOT decompose the IVAP-calibrated
    probability: IVAP is monotone but not linear, so no additive attribution of
    the displayed probability exists. `shap_scale` says so in the response.
    """
    pipeline = bundle["pipeline"]
    design = np.asarray(pipeline.named_steps["prep"].transform(encoded), dtype=float)[0]
    weights = np.asarray(pipeline.named_steps["lr"].coef_[0], dtype=float)
    background = np.asarray(bundle["shap_background_mean"], dtype=float)
    per_column = weights * (design - background)

    groups = bundle["shap_column_groups"]
    values = np.array([per_column[[i for i in groups[f]]].sum() for f in MODEL_FEATURES])
    base = float(pipeline.named_steps["lr"].intercept_[0] + float(weights @ background))
    return values, base


def _column_groups(pipeline: Pipeline) -> Dict[str, List[int]]:
    """Map each raw model feature to the design-matrix columns it produced."""
    names = list(pipeline.named_steps["prep"].get_feature_names_out())
    groups: Dict[str, List[int]] = {f: [] for f in MODEL_FEATURES}
    for index, name in enumerate(names):
        bare = name.split("__", 1)[-1]
        # Longest first, so "FastingBS_cat" is not captured by a shorter prefix.
        owner = max(
            (f for f in MODEL_FEATURES if bare == f or bare.startswith(f + "_")),
            key=len,
            default=None,
        )
        if owner is None:
            raise ValueError(f"design column {name!r} belongs to no model feature")
        groups[owner].append(index)
    missing = [f for f, cols in groups.items() if not cols]
    if missing:
        raise ValueError(f"model features produced no design column: {missing}")
    return groups


# ── Bundle ───────────────────────────────────────────────────────────────────

def sha256_of(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def measure_blank_impact(
    pipeline: Pipeline, cells: Mapping[str, float], oof: np.ndarray,
    y: np.ndarray, frame: pd.DataFrame,
) -> Dict[str, Dict[str, float]]:
    """
    For each Optional input: what changes if a clinician leaves it blank?

    Measured at build time, on the artifact being built, against the patients
    who actually have the field recorded. Reported on what the clinician is
    shown -- the IVAP probability and the conformal decision -- not on the raw
    score, because the raw score is not what anyone acts on.

    This exists so that the data-completeness warning list is a property of the
    MODEL rather than a list somebody typed into a config. A typed list keeps
    describing the model it was written for: that is exactly how the importance
    file came to describe a model that was never deployed (F0-1), and a warning
    list carries the same risk one level up. Retraining now moves these numbers,
    and the warning list moves with them.
    """
    impact: Dict[str, Dict[str, float]] = {}
    for field in OPTIONAL_INPUT_FEATURES:
        present = frame[frame[field].notna()]
        if present.empty:
            impact[field] = {"n": 0, "decision_changed": 0, "decision_changed_share": 0.0,
                             "mean_delta_points": 0.0, "p90_delta_points": 0.0}
            continue

        def shown(rows: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
            raw = pipeline.predict_proba(encode_for_training(rows))[:, 1]
            prob = ivap(oof, y.astype(float), raw)[0]
            decisions = [decide(float(s), g, cells)[0] for s, g in zip(raw, rows["Sex"])]
            return prob, decisions

        p_present, dec_present = shown(present)
        p_blank, dec_blank = shown(present.assign(**{field: np.nan}))
        delta = (p_blank - p_present) * 100.0
        changed = [a != b for a, b in zip(dec_present, dec_blank)]
        impact[field] = {
            "n": int(len(present)),
            "decision_changed": int(sum(changed)),
            "decision_changed_share": round(float(np.mean(changed)), 6),
            "mean_delta_points": round(float(delta.mean()), 4),
            "p90_delta_points": round(float(np.percentile(delta, 90)), 4),
        }
    return impact


def blank_warning_features(
    blank_impact: Mapping[str, Mapping[str, float]], min_decision_share: float,
) -> List[str]:
    """
    Which blank inputs are worth telling a clinician about: the ones whose
    absence actually changes decisions, at or above the declared share.

    The rule is the config's, the numbers are the bundle's, and the list is
    neither's -- it is derived from both. Warning about a field that changes
    nothing is not extra caution, it is noise that costs the warning its
    meaning.
    """
    return [
        field
        for field in OPTIONAL_INPUT_FEATURES
        if blank_impact.get(field, {}).get("decision_changed_share", 0.0) >= min_decision_share
    ]


def build_bundle(csv_path: str | Path, experiments_git_ref: str = "") -> Dict[str, Any]:
    """Fit the deployable stack on all 920 UCI patients.

    Reproduces `scripts/p7_final.py` in the research repo: 5-fold out-of-fold
    predictions stratified by (site x sex x label) with seed 42 feed both the
    IVAP calibrator and the Mondrian cells; the model itself is then refit on
    every row. Cross-fitting the calibration is what makes the per-cell counts
    usable at all -- there are only 50 diseased women in the whole dataset.
    """
    frame = pd.read_csv(csv_path)
    encoded = encode_for_training(frame)
    y = frame["HeartDisease"].to_numpy(dtype=int)
    sex = frame["Sex"].to_numpy(dtype=object)
    site = frame["site"].to_numpy(dtype=object)

    strata = np.char.add(
        np.char.add(site.astype(str), sex.astype(str)), y.astype(str)
    )
    oof = np.zeros(len(y))
    for train_idx, test_idx in StratifiedKFold(
        N_FOLDS, shuffle=True, random_state=CV_SEED
    ).split(encoded, strata):
        fold = build_pipeline().fit(encoded.iloc[train_idx], y[train_idx])
        oof[test_idx] = fold.predict_proba(encoded.iloc[test_idx])[:, 1]

    pipeline = build_pipeline().fit(encoded, y)

    cells = {
        f"{g}{k}": conformal_quantile(
            (1 - oof[(sex == g) & (y == 1)]) if k == 1 else oof[(sex == g) & (y == 0)]
        )
        for g in ("F", "M")
        for k in (0, 1)
    }
    design = np.asarray(pipeline.named_steps["prep"].transform(encoded), dtype=float)

    return {
        "pipeline": pipeline,
        "cal_scores": oof,
        "cal_labels": y.astype(float),
        "conformal_cells": cells,
        "alpha": ALPHA,
        "features_input": list(INPUT_FEATURES),
        "features_model": list(MODEL_FEATURES),
        "unused_input_features": list(UNUSED_INPUT_FEATURES),
        "cp_map_clinical": dict(CP_MAP_CLINICAL),
        "cp_map_uci_raw": dict(CP_MAP_UCI_RAW),
        "shap_background_mean": design.mean(axis=0),
        "shap_column_groups": _column_groups(pipeline),
        "review_queue_width_cut": float(np.quantile(_ivap_width(oof, y, oof), 0.10)),
        "blank_impact": measure_blank_impact(pipeline, cells, oof, y, frame),
        "model_card": {
            "family": "glm_ivap_conformal",
            "model": f"Spline-GLM (SplineTransformer n_knots={N_KNOTS}) + LogisticRegression C={GLM_C}",
            "feature_set": "L3 (7 pre-stress-test inputs), D-25",
            # Counted, not typed. The shipped build passes the 920-row UCI file
            # and these read "920 patients / 4 sites" as they always have; the
            # retrain path (Gate 8.10) passes a merged file, and a hardcoded 920
            # would have made its candidate card state a row count the candidate
            # was not trained on.
            "training_data": (
                f"{csv_path} — {len(frame)} patients, "
                f"{frame['site'].nunique()} sites "
                f"({', '.join(sorted(frame['site'].astype(str).unique()))})"
            ),
            "training_csv_sha256": sha256_of(csv_path),
            "split": (
                f"fit on all {len(frame)}; IVAP and Mondrian cells cross-fitted on "
                f"{N_FOLDS}-fold OOF stratified by site x sex x label, seed {CV_SEED}"
            ),
            "calibration": "exact inductive Venn-Abers (IVAP) on the training-hospital mix",
            "conformal": f"Mondrian by (sex x class), alpha={ALPHA}",
            "built": date.today().isoformat(),
            "experiments_git_ref": experiments_git_ref,
            "decisions": ["D-25", "D-26", "D-30", "D-32", "D-33"],
        },
    }


def _ivap_width(cal_scores: np.ndarray, cal_labels: np.ndarray, scores: np.ndarray) -> np.ndarray:
    _, p0, p1 = ivap(cal_scores, cal_labels, scores)
    return p1 - p0


def reference_scores(bundle: Mapping[str, Any], csv_path: str | Path) -> Dict[str, Any]:
    """The output fingerprint checked at image build time (Gate 8.1 §3)."""
    frame = pd.read_csv(csv_path)
    encoded = encode_for_training(frame)
    raw = bundle["pipeline"].predict_proba(encoded)[:, 1]
    p, p0, p1 = ivap(bundle["cal_scores"], bundle["cal_labels"], raw)
    decisions = [decide(s, g, bundle["conformal_cells"]) for s, g in zip(raw, frame["Sex"])]
    return {
        "training_csv_sha256": sha256_of(csv_path),
        "n": int(len(raw)),
        "raw": [float(v) for v in raw],
        "ivap": [float(v) for v in p],
        "p_lower": [float(v) for v in p0],
        "p_upper": [float(v) for v in p1],
        "decision": [d for d, _ in decisions],
        "conformal_set": ["".join(str(k) for k in s) for _, s in decisions],
    }


def verify_against_reference(
    bundle: Mapping[str, Any],
    csv_path: str | Path,
    reference_path: str | Path,
    tolerance: float,
) -> Dict[str, Any]:
    """Compare a freshly built bundle with the recorded fingerprint.

    Two conditions, declared before the first run (Gate 8.1 §3):
      (a) decision and conformal set identical for all 920 — absolute;
      (b) max |delta p| <= tolerance.
    A failure of (a) is a bug to report, never a tolerance to relax.
    """
    with open(reference_path) as handle:
        reference = json.load(handle)
    fresh = reference_scores(bundle, csv_path)

    if fresh["training_csv_sha256"] != reference["training_csv_sha256"]:
        raise ValueError(
            "training CSV sha256 does not match the recorded value:\n"
            f"  recorded {reference['training_csv_sha256']}\n"
            f"  actual   {fresh['training_csv_sha256']}"
        )
    if fresh["n"] != reference["n"]:
        raise ValueError(f"row count changed: {reference['n']} -> {fresh['n']}")

    mismatched = [
        i
        for i in range(fresh["n"])
        if fresh["decision"][i] != reference["decision"][i]
        or fresh["conformal_set"][i] != reference["conformal_set"][i]
    ]
    deltas = {
        key: float(np.max(np.abs(np.array(fresh[key]) - np.array(reference[key]))))
        for key in ("raw", "ivap", "p_lower", "p_upper")
    }
    result = {"max_abs_delta": deltas, "decision_mismatches": len(mismatched)}

    if mismatched:
        raise ValueError(
            f"decision or conformal set changed for {len(mismatched)} of {fresh['n']} patients "
            f"(rows {mismatched[:10]}{'...' if len(mismatched) > 10 else ''}). "
            f"This is a bug to report, not a tolerance to relax. Deltas: {deltas}"
        )
    worst = max(deltas.values())
    if worst > tolerance:
        raise ValueError(
            f"max |delta p| = {worst:.3e} exceeds the declared tolerance {tolerance:.0e}. "
            f"Per quantity: {deltas}"
        )
    return result
