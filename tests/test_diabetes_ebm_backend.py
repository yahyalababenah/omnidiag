"""
Gate 9.1 — the NHANES/EBM diabetes backend.

These tests guard the things Phase 9 measured and decided, not the framework.
Each one names the finding or decision it protects, so that a future change that
breaks it says why it matters rather than just going red.
"""

import numpy as np
import pandas as pd
import pytest
import yaml

from backend.model_backends import get_backend, registered_families
from backend.model_backends.diabetes_ebm_conformal import (
    CBC_BANNED,
    GLYCEMIC_BANNED,
    DiabetesEbmConformalBackend,
    MandatoryFieldMissing,
    age_band,
)

CONFIG_PATH = "configs/diabetes_nhanes.yaml"

# A 58-year-old man with central obesity, low HDL, sedentary. Referred.
PATIENT = {
    "RIDAGEYR": 58, "RIAGENDR": 1.0, "BMXBMI": 31.2, "ADIPOSITY_BAND": 2.0,
    "SBP": 138, "DBP": 84, "BPXPLS": 78, "MCQ300C": 1.0, "CVD_ANY": 0.0,
    "PAQ650": 0.0, "PAQ665": 0.0, "LBDHDD": 41, "LBXSCH": 205, "LBXSTR": 190,
    "LBXSATSI": 28, "LBXSGTSI": 34, "LBXSCR": 0.95, "LBXSBU": 15,
    "LBXSAL": 4.2, "LBXSUA": 6.4,
}


@pytest.fixture(scope="module")
def config():
    with open(CONFIG_PATH) as handle:
        return yaml.safe_load(handle)


@pytest.fixture(scope="module")
def backend(config):
    return get_backend("ebm_platt_conformal")(config).load()


# ── registry ─────────────────────────────────────────────────────────────

def test_family_registers_by_existing():
    """model_backends/__init__.py imports every module in the package, so an
    import error in any one of them breaks startup for EVERY disease. This is
    the smoke test the Phase 9 brief asked for."""
    assert "ebm_platt_conformal" in registered_families()


def test_adding_this_family_did_not_disturb_the_others():
    for family in ("glm_ivap_conformal", "sklearn_pipeline", "stacking_ensemble"):
        assert family in registered_families()


# ── D9-03: the CBC ban (F9-29) ───────────────────────────────────────────

def test_no_blood_count_column_is_an_input(backend):
    """A complete blood count raises AUC +0.0218 against the HbA1c label and
    -0.0040 against a glucose label: the gain is an assay artefact, not risk
    information. It must never become an input."""
    assert not CBC_BANNED.intersection(backend.feature_names)


def test_no_glycaemic_measurement_is_an_input(backend):
    """D9-09. The module's claim is that it finds dysglycaemia WITHOUT measuring
    glycaemia. Glucose passes the usual leakage test — it is available at
    inference time — and fails the one that matters: it measures the same latent
    quantity as the label. Insufficiency is not non-leakage; glucose alone reaches
    AUC 0.7029, which is what a proxy target looks like."""
    assert not GLYCEMIC_BANNED.intersection(backend.feature_names)
    for name in ("LBXGH", "LBXSGL", "LBDSGLSI", "LBXGLU", "LBXSOSSI"):
        assert name in GLYCEMIC_BANNED


def test_bundle_with_glucose_is_refused(config, tmp_path):
    """The ban is code, not a comment. A bundle built elsewhere cannot smuggle a
    glycaemic column in, and this is the test that stops D9-08 being re-made."""
    import joblib

    original = joblib.load(config["model"]["weights_path"])
    tampered = dict(original)
    tampered["features"] = list(original["features"]) + ["LBXSGL"]
    path = tmp_path / "with_glucose.joblib"
    joblib.dump(tampered, path)

    cfg = {**config, "model": {**config["model"], "weights_path": str(path)}}
    with pytest.raises(ValueError, match="D9-09"):
        DiabetesEbmConformalBackend(cfg).load()


def test_config_lists_the_glycaemic_ban(config):
    banned = set(config["features"]["glycaemic_banned"])
    assert GLYCEMIC_BANNED.issubset(banned)


def test_bundle_with_a_cbc_column_is_refused(config, tmp_path):
    import joblib

    original = joblib.load(config["model"]["weights_path"])
    tampered = dict(original)
    tampered["features"] = list(original["features"]) + ["LBXRDW"]
    path = tmp_path / "tampered.joblib"
    joblib.dump(tampered, path)

    cfg = {**config, "model": {**config["model"], "weights_path": str(path)}}
    with pytest.raises(ValueError, match="D9-03"):
        DiabetesEbmConformalBackend(cfg).load()


# ── the model is its own explanation ─────────────────────────────────────

def test_contributions_sum_to_the_score(backend):
    """EBM additivity. The per-term contributions ARE the model; an explanation
    that does not sum to the score is worse than none. Measured 2.67e-15 across
    the whole test cycle in Gate 9.0b."""
    explained = backend.explain(PATIENT)
    total = explained["intercept"] + sum(
        term["contribution"] for term in explained["term_contributions"]
    )
    frame = pd.DataFrame([PATIENT])[backend.feature_names]
    expected = float(np.asarray(backend.bundle["model"].decision_function(frame)).ravel()[0])
    assert abs(total - expected) < 1e-8


def test_per_feature_values_also_sum_to_the_score(backend):
    """Pairwise terms are split evenly between their two features for the chart.
    The split is presentation; the sum has to stay exact."""
    result = backend.shap_values(pd.DataFrame([PATIENT]))
    frame = pd.DataFrame([PATIENT])[backend.feature_names]
    expected = float(np.asarray(backend.bundle["model"].decision_function(frame)).ravel()[0])
    assert abs(result.base_value + result.values.sum() - expected) < 1e-8


def test_explanation_states_its_scale(backend):
    """Contributions explain the raw score, not the calibrated probability.
    Consumers must not have to infer that."""
    assert backend.explain(PATIENT)["shap_scale"] == "log_odds_raw_score"


# ── D9-06: the missing-field firewall (F9-28) ────────────────────────────

@pytest.mark.parametrize("field", ["RIDAGEYR", "BMXBMI", "ADIPOSITY_BAND",
                                   "LBDHDD", "PAQ650", "PAQ665"])
def test_mandatory_field_is_refused_not_scored(backend, field):
    """Blanking PAQ650 shifts median risk by +0.108 and flips 31% of decisions.
    A blank is read as a learned value, not as 'unknown', so the row must be
    refused rather than scored."""
    with pytest.raises(MandatoryFieldMissing) as caught:
        backend.predict({**PATIENT, field: None})
    assert field in caught.value.fields


def test_mandatory_list_matches_the_config(backend, config):
    assert sorted(backend.mandatory_fields) == sorted(config["features"]["mandatory"])


def test_optional_blank_scores_but_warns(backend):
    result = backend.predict({**PATIENT, "LBXSUA": None})
    assert "data_completeness_warning" in result
    assert "LBXSUA" in result["data_completeness_warning"]


# ── D9-04 / D9-05: the probability and decision layers ───────────────────

def test_decision_is_a_set_not_a_threshold(backend, config):
    result = backend.predict(PATIENT)
    assert result["decision"] in {"referral", "no_referral", "uncertain"}
    assert result["output_type"] == "conformal_decision"
    assert config["model"]["inference_threshold"] is None
    assert config["model"]["risk_bands"] is None


def test_uncertain_counts_as_a_referral(backend):
    """Both referral and uncertain mean an HbA1c is ordered. Counting only the
    confident referrals as positives is the silent sensitivity drop this family
    exists to avoid."""
    for patient in (PATIENT, {**PATIENT, "RIDAGEYR": 28, "BMXBMI": 22.0,
                              "ADIPOSITY_BAND": 0.0, "PAQ650": 1.0, "PAQ665": 1.0}):
        result = backend.predict(patient)
        if result["decision"] == "uncertain":
            assert result["prediction"] == 1
            assert result["decision_is_referral"] is True


def test_interval_brackets_the_probability(backend):
    result = backend.predict(PATIENT)
    assert result["probability_lower"] <= result["confidence"] <= result["probability_upper"]
    assert 0.0 <= result["probability_lower"] and result["probability_upper"] <= 1.0


def test_platt_preserves_the_ranking(backend):
    """Platt is monotone in the raw score, which is why it cost no AUC where
    isotonic cost 0.0013. If this ever fails, the calibrator is not Platt."""
    raw = np.linspace(0.02, 0.98, 60)
    calibrated = backend._platt(raw)
    assert np.all(np.diff(calibrated) > 0)


def test_conformal_layer_is_group_conditional(backend, config):
    """alpha is read from the config rather than hardcoded: D9-07 moved it once
    already and a hardcoded copy here would have to be chased every time."""
    conformal = backend.bundle["conformal"]
    assert conformal["layer"] == "group_conditional_age_band"
    assert conformal["alpha"] == config["model"]["conformal"]["alpha"]
    # one quantile per (band, class); F9-31 is why this is not two numbers
    assert len(conformal["q_group"]) == 6


@pytest.mark.parametrize("age,band", [(20, 0), (39, 0), (40, 1), (59, 1), (60, 2), (81, 2)])
def test_age_band_boundaries(age, band):
    assert age_band(age) == band


# ── config honesty ───────────────────────────────────────────────────────

def test_config_states_no_deployment_prevalence(config):
    """F9-12. Jordan's 0.237 is total diabetes prevalence including diagnosed
    cases; this model's target is dysglycaemia among the undiagnosed and
    untreated. A wrong scalar here is the same class of error as an inverted
    label."""
    assert config["prevalence"]["deploy"] is None
    assert config["prevalence"]["requires_local_recalibration"] is True


def test_config_headline_auc_matches_the_bundle(backend, config):
    """No number reaches a config without being reproduced from the artifact."""
    measured = backend.bundle["metrics"]["test_auc_raw"]
    assert abs(config["performance"]["auc"] - measured) < 5e-5


def test_config_conformal_numbers_match_the_bundle(backend, config):
    measured = backend.bundle["metrics"]["conformal"]
    stated = config["model"]["conformal"]
    assert stated["alpha"] == measured["alpha"]
    assert abs(stated["coverage_pos"] - measured["coverage_pos"]) < 5e-5
    assert abs(stated["coverage_neg"] - measured["coverage_neg"]) < 5e-5


def test_every_limitation_has_a_finding_id(config):
    for item in config["limitations"]:
        assert item["id"].startswith(("F9-", "D9-"))
        assert len(item["text"]) > 80


# ── batch ────────────────────────────────────────────────────────────────

def test_batch_matches_single(backend):
    rows = [PATIENT, {**PATIENT, "RIDAGEYR": 34, "ADIPOSITY_BAND": 0.0}]
    batch = backend.predict_batch(rows)
    assert len(batch) == 2
    for one, many in zip((backend.predict(r) for r in rows), batch):
        assert one["decision"] == many["decision"]
        assert abs(one["confidence"] - many["confidence"]) < 1e-12


def test_empty_batch_is_empty(backend):
    assert backend.predict_batch([]) == []
