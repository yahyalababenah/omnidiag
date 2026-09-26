"""
Gate 8.1 — the shipped heart model.

What these tests protect, and why each one exists:

  * HF-1 (S1). UCI's chest-pain codes are inverted with respect to their own
    documentation, and the previous model inherited that: a clinician choosing
    "Typical Angina" got a LOWER risk than "Asymptomatic" for 100% of patients,
    flipping 26.4% of decisions. The API keeps the clinical meaning, so the fix
    lives in the bundle's two maps — and the test runs through the API, not the
    model, because that is where a clinician meets it.

    It asserts ONLY typical > no-anginal-features. The full clinical order does
    NOT hold and is not claimed: the unconstrained GLM learns
    typical 0.736 > none 0.444 > atypical 0.389 > non-anginal 0.242 (measured
    on all 920), because "no anginal features yet catheterised anyway" carries
    real referral signal in this cohort. That is inherited from how the data
    were collected, not a clinical ordering, and it is stated in the config and
    the defence document rather than constrained away.

  * The two maps never mix. A single crossed import would restore HF-1
    silently, so it is checked at source level.

  * The bundle reproduces the research artifact. The build script's fingerprint
    check is the gate; this is the same check from the test suite's side.

  * SHAP is exact, not approximate. Shipping an approximate explanation while
    calling it SHAP is the same class of defect as F0-1.

  * Importing every backend module keeps startup alive. The package imports its
    folder, so one broken module takes down BOTH diseases.
"""

from __future__ import annotations

import ast
import importlib
import json
import os
import pkgutil
import time

import numpy as np
import pandas as pd
import pytest

from backend.heart_glm import stack
from backend.model_backends import registered_families
from backend.router import OmniDiagRouter

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TRAINING_CSV = os.path.join(_ROOT, "data/heart_disease/processed/uci_heart_by_site.csv")
_REFERENCE = os.path.join(_ROOT, "backend/heart_glm/reference_scores.json")

PATIENT = {
    "Age": 54, "Sex": "M", "ChestPainType": "ATA", "RestingBP": 140,
    "Cholesterol": 289, "FastingBS": 0, "RestingECG": "Normal",
    "MaxHR": 122, "ExerciseAngina": "N", "Oldpeak": 0.0, "ST_Slope": "Flat",
}


@pytest.fixture(scope="module")
def router() -> OmniDiagRouter:
    return OmniDiagRouter(configs_dir=os.path.join(_ROOT, "configs"))


@pytest.fixture(scope="module")
def backend(router):
    return router._get_loader("heart_disease")


@pytest.fixture(scope="module")
def training_frame() -> pd.DataFrame:
    return pd.read_csv(_TRAINING_CSV)


# ═════════════════════════════════════════════════════════════════════════════
# Startup
# ═════════════════════════════════════════════════════════════════════════════

def test_every_backend_module_imports():
    """The package imports its whole folder; a broken module kills both diseases."""
    import backend.model_backends as package

    for module in pkgutil.iter_modules(package.__path__):
        if not module.name.startswith("_"):
            importlib.import_module(f"{package.__name__}.{module.name}")
    assert "glm_ivap_conformal" in registered_families()


def test_heart_dispatches_to_the_new_family(backend):
    assert backend.family == "glm_ivap_conformal"
    assert backend.capabilities.explainer == "linear"
    assert not backend.capabilities.supports_tree_shap


def test_heart_exposes_no_threshold_and_no_risk_bands(router):
    """A single cut-point is what produced HF-13; this module must expose none."""
    info = router.get_disease_info("heart_disease")
    assert info["inference_threshold"] is None
    assert info["risk_bands"] is None


# ═════════════════════════════════════════════════════════════════════════════
# HF-1 — chest pain, through the API surface
# ═════════════════════════════════════════════════════════════════════════════

def test_typical_angina_outranks_no_anginal_features_for_every_patient(router, training_frame):
    """The HF-1 assertion, on all 920 patients, raw score and calibrated alike."""
    backend = router._get_loader("heart_disease")
    frame = training_frame[backend.feature_names].copy()

    scores = {}
    for code in ("TA", "ASY"):
        variant = frame.copy()
        variant["ChestPainType"] = code
        raw = backend.predict_proba(variant)
        calibrated, _, _ = stack.ivap(
            backend.bundle["cal_scores"], backend.bundle["cal_labels"], raw
        )
        scores[code] = (raw, calibrated)

    assert np.all(scores["TA"][0] > scores["ASY"][0]), "raw score: HF-1 is back"
    assert np.all(scores["TA"][1] >= scores["ASY"][1]), "calibrated: HF-1 is back"


def test_typical_angina_outranks_no_anginal_features_through_predict(router):
    """The same property one patient at a time, through router.predict()."""
    typical = router.predict("heart_disease", {**PATIENT, "ChestPainType": "TA"})
    asymptomatic = router.predict("heart_disease", {**PATIENT, "ChestPainType": "ASY"})
    assert typical["confidence"] > asymptomatic["confidence"]


def test_the_two_chest_pain_maps_are_never_crossed():
    """Source-level: each encoder reads exactly one map."""
    source = ast.parse(open(stack.__file__).read())
    functions = {
        node.name: ast.dump(node)
        for node in source.body
        if isinstance(node, ast.FunctionDef)
    }
    assert "CP_MAP_CLINICAL" not in functions["encode_for_training"]
    assert "CP_MAP_UCI_RAW" not in functions["encode_for_inference"]
    assert "CP_MAP_UCI_RAW" in functions["encode_for_training"]
    assert "CP_MAP_CLINICAL" in functions["encode_for_inference"]


def test_inference_rejects_an_unknown_chest_pain_code():
    with pytest.raises(ValueError, match="clinical meaning"):
        stack.encode_for_inference([{**PATIENT, "ChestPainType": "unknown"}])


# ═════════════════════════════════════════════════════════════════════════════
# The bundle is the measured model
# ═════════════════════════════════════════════════════════════════════════════

def test_bundle_matches_the_recorded_fingerprint(backend, training_frame):
    """Same check the Docker build runs: decisions absolute, probabilities to 1e-6."""
    reference = json.load(open(_REFERENCE))
    assert stack.sha256_of(_TRAINING_CSV) == reference["training_csv_sha256"]

    encoded = stack.encode_for_training(training_frame)
    raw = backend.bundle["pipeline"].predict_proba(encoded)[:, 1]
    probability, _, _ = stack.ivap(
        backend.bundle["cal_scores"], backend.bundle["cal_labels"], raw
    )
    decisions = [
        stack.decide(float(s), g, backend.bundle["conformal_cells"])[0]
        for s, g in zip(raw, training_frame["Sex"])
    ]
    assert decisions == reference["decision"]
    assert np.max(np.abs(raw - np.array(reference["raw"]))) <= 1e-6
    assert np.max(np.abs(probability - np.array(reference["ivap"]))) <= 1e-6


def test_uncertain_is_a_referral(backend, training_frame):
    """`prediction` must count uncertain patients as referred.

    Reading only the confident referrals as positives drops sensitivity to 0.60
    for women and 0.43 for men (Phase 7). Any consumer counting positives reads
    this field, so this is where the property is pinned.
    """
    rows = training_frame.head(200).to_dict(orient="records")
    results = backend.predict_batch(rows)

    for row, result in zip(rows, results):
        raw = backend.predict_proba(pd.DataFrame([row])[backend.feature_names])[0]
        decision, _ = stack.decide(float(raw), row["Sex"], backend.bundle["conformal_cells"])
        if decision == stack.DECISION_UNCERTAIN:
            assert result["prediction"] == 1
            assert result["diagnosis"] == stack.UNCERTAIN_DIAGNOSIS
        elif decision == stack.DECISION_REFERRAL:
            assert result["prediction"] == 1 and result["diagnosis"] == "Positive"
        else:
            assert result["prediction"] == 0 and result["diagnosis"] == "Negative"


def test_predict_batch_matches_predict_row_by_row(backend, training_frame):
    rows = training_frame.head(25).to_dict(orient="records")
    batch = backend.predict_batch(rows)
    for row, batched in zip(rows, batch):
        single = backend.predict(row)
        assert single["prediction"] == batched["prediction"]
        assert single["confidence"] == pytest.approx(batched["confidence"], abs=1e-12)


def test_blank_fasting_bs_does_not_silently_raise_risk(backend, training_frame):
    """HF-11: the previous model added 7.6 points for a blank FastingBS, with
    no warning. D-26 removed the mechanism; this holds the line."""
    rows = training_frame[training_frame["FastingBS"].notna()].head(150)
    with_value = backend.predict_proba(rows[backend.feature_names])
    blanked = rows.copy()
    blanked["FastingBS"] = None
    without = backend.predict_proba(blanked[backend.feature_names])
    assert float(np.mean(without - with_value)) * 100 <= 1.0

    warned = backend.predict(blanked.iloc[0].to_dict())
    assert "FastingBS" in warned.get("data_completeness_warning", "")


# ═════════════════════════════════════════════════════════════════════════════
# SHAP — exact, and said to be on the raw scale
# ═════════════════════════════════════════════════════════════════════════════

def test_shap_is_additive_on_the_raw_log_odds(backend, training_frame):
    rows = training_frame.head(50)
    raw = backend.predict_proba(rows[backend.feature_names])
    logit = np.log(raw / (1 - raw))
    for i in range(len(rows)):
        result = backend.shap_values(pd.DataFrame([rows.iloc[i][backend.feature_names]]))
        assert abs(result.base_value + result.values.sum() - logit[i]) < 1e-8


def test_shap_matches_shaps_own_linear_and_permutation_explainers(backend, training_frame):
    shap = pytest.importorskip("shap")
    bundle = backend.bundle
    pipeline = bundle["pipeline"]
    rows = training_frame.head(10)[backend.feature_names]
    encoded = stack.encode_for_inference(rows)
    design = np.asarray(pipeline.named_steps["prep"].transform(encoded), dtype=float)
    background = np.asarray(bundle["shap_background_mean"], dtype=float)
    groups = bundle["shap_column_groups"]

    linear = shap.LinearExplainer(
        pipeline.named_steps["lr"], (background, np.zeros((len(background), len(background))))
    ).shap_values(design)

    def raw_logit(matrix: np.ndarray) -> np.ndarray:
        proba = pipeline.named_steps["lr"].predict_proba(matrix)[:, 1]
        return np.log(proba / (1 - proba))

    permutation = shap.PermutationExplainer(
        raw_logit, background.reshape(1, -1)
    ).shap_values(design, silent=True)

    for i in range(len(rows)):
        ours = backend.shap_values(pd.DataFrame([rows.iloc[i]])).values
        for j, feature in enumerate(stack.MODEL_FEATURES):
            columns = groups[feature]
            assert abs(ours[j] - linear[i][columns].sum()) < 1e-6
            assert abs(ours[j] - permutation[i][columns].sum()) < 1e-6


def test_explain_declares_the_scale_and_flags_imputed_values(router):
    blank = {**PATIENT, "Cholesterol": None}
    result = router.explain("heart_disease", blank)
    assert result["shap_scale"] == "log_odds_raw_score"
    flags = {item["feature"]: item["imputed"] for item in result["chart_data"]}
    assert flags["Cholesterol"] is True
    assert flags["Age"] is False
    # Inputs this model does not read never appear in an explanation of it.
    assert set(flags) == set(stack.MODEL_FEATURES)


# ═════════════════════════════════════════════════════════════════════════════
# Cost
# ═════════════════════════════════════════════════════════════════════════════

def test_single_patient_latency_is_reasonable(backend):
    backend.predict(dict(PATIENT))  # warm
    start = time.perf_counter()
    for _ in range(20):
        backend.predict(dict(PATIENT))
    elapsed_ms = (time.perf_counter() - start) / 20 * 1000
    assert elapsed_ms < 250, f"{elapsed_ms:.0f} ms per patient"
