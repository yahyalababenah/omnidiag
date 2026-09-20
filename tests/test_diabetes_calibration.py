"""
Tests — Diabetes prevalence correction and threshold (regression guard)
=======================================================================
Pins the two diabetes decision fixes:
  * threshold selected on training out-of-fold predictions, not y_test
  * Bayes prior-shift correction applied in EnsembleModelLoader.predict()

Sections
  A. correction formula — pure unit tests, no model
  B. decision invariance — correction changes the scale, never the decision
  C. /predict integration with the REAL router (diabetes + heart)
  D. heart non-regression — golden outputs captured before the change
  E. extreme and malformed diabetes input

The real models are loaded once per module (module-scoped fixture; the
shared `app` fixture is module-scoped, so the real router can be swapped in
and the mock restored cleanly). No network, no external data.
"""

import os

import numpy as np
import pytest
import yaml

from backend.prevalence_correction import (
    apply_prevalence_correction,
    invert_prevalence_correction,
    prior_odds_ratio,
)
# Captured at import time, before conftest's `app` fixture patches the class.
from backend.router import OmniDiagRouter as _RealOmniDiagRouter

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CONFIGS_DIR = os.path.join(_ROOT, "configs")

with open(os.path.join(_CONFIGS_DIR, "diabetes.yaml")) as _f:
    _DIABETES_MODEL_CFG = yaml.safe_load(_f)["model"]
PI_TRAIN = float(_DIABETES_MODEL_CFG["prevalence_train"])
PI_DEPLOY = float(_DIABETES_MODEL_CFG["prevalence_deploy"])
THRESHOLD_DEPLOYED = float(_DIABETES_MODEL_CFG["inference_threshold"])


def correct(p, pi_train=PI_TRAIN, pi_deploy=PI_DEPLOY):
    return apply_prevalence_correction(p, pi_train, pi_deploy)


# ═════════════════════════════════════════════════════════════════════════════
# A. Correction formula
# ═════════════════════════════════════════════════════════════════════════════

class TestCorrectionFormula:
    def test_strictly_monotonic_on_200_points(self):
        p = np.linspace(0.001, 0.999, 200)
        out = correct(p)
        assert out.shape == p.shape
        assert np.all(np.diff(out) > 0)

    def test_fixes_zero_and_one(self):
        assert correct(0.0) == 0.0
        assert correct(1.0) == 1.0

    def test_equal_priors_is_identity(self):
        assert prior_odds_ratio(0.3, 0.3) == pytest.approx(1.0, abs=1e-12)
        p = np.linspace(0.0, 1.0, 101)
        np.testing.assert_allclose(correct(p, 0.3, 0.3), p, rtol=0, atol=1e-9)

    def test_lower_deploy_prior_lowers_every_interior_probability(self):
        assert PI_DEPLOY < PI_TRAIN, "config premise of this test"
        p = np.linspace(0.001, 0.999, 200)
        assert np.all(correct(p) < p)

    def test_matches_closed_form_for_configured_priors(self):
        # R = (0.14/0.86) / (0.5/0.5) = 0.162790...
        r = (PI_DEPLOY / (1 - PI_DEPLOY)) / (PI_TRAIN / (1 - PI_TRAIN))
        assert prior_odds_ratio(PI_TRAIN, PI_DEPLOY) == pytest.approx(r, rel=1e-12)
        assert correct(0.5) == pytest.approx(PI_DEPLOY, abs=1e-12)  # p = pi_train -> pi_deploy
        assert correct(0.3) == pytest.approx(r * 0.3 / (0.7 + r * 0.3), rel=1e-12)

    @pytest.mark.parametrize("p", [0.0, 1.0, 1e-12, 1 - 1e-12])
    def test_numerically_stable_at_edges(self, p):
        out = correct(p)
        assert np.isfinite(out)
        assert 0.0 <= out <= 1.0
        back = invert_prevalence_correction(out, PI_TRAIN, PI_DEPLOY)
        assert np.isfinite(back)
        assert back == pytest.approx(p, abs=1e-9)

    def test_inverse_round_trips(self):
        p = np.linspace(0.0, 1.0, 201)
        np.testing.assert_allclose(
            invert_prevalence_correction(correct(p), PI_TRAIN, PI_DEPLOY), p, atol=1e-12
        )

    @pytest.mark.parametrize("bad", [-0.01, 1.01, float("nan"), float("inf")])
    def test_rejects_non_probabilities(self, bad):
        with pytest.raises(ValueError):
            correct(bad)

    @pytest.mark.parametrize("prior", [0.0, 1.0, -0.1, 1.5])
    def test_rejects_degenerate_priors(self, prior):
        with pytest.raises(ValueError):
            correct(0.5, prior, 0.14)
        with pytest.raises(ValueError):
            correct(0.5, 0.5, prior)


# ═════════════════════════════════════════════════════════════════════════════
# B. Decision invariance — the most important property in this file
# ═════════════════════════════════════════════════════════════════════════════

def _confusion(y, p, t):
    pred = p >= t
    return np.array([
        [np.sum(~pred & (y == 0)), np.sum(pred & (y == 0))],
        [np.sum(~pred & (y == 1)), np.sum(pred & (y == 1))],
    ])


class TestDecisionInvariance:
    @pytest.fixture(scope="class")
    def scores(self):
        rng = np.random.default_rng(42)
        y = rng.integers(0, 2, 20_000)
        # Informative but overlapping scores, plus exact ties and the endpoints.
        p = np.clip(rng.beta(2 + 3 * y, 5 - 3 * y), 0.0, 1.0)
        p[:50] = 0.0
        p[50:100] = 1.0
        p[100:400] = np.round(p[100:400], 2)
        return y, p

    @pytest.mark.parametrize("pi_deploy", [0.10, 0.14, 0.20, 0.30])
    @pytest.mark.parametrize("t", [0.05, 0.2808542713567839, 0.5, 0.73])
    def test_confusion_matrix_identical_after_correction(self, scores, pi_deploy, t):
        y, p = scores
        before = _confusion(y, p, t)
        after = _confusion(y, correct(p, PI_TRAIN, pi_deploy), correct(t, PI_TRAIN, pi_deploy))
        assert before.sum() == len(y)
        assert 0 < before[:, 1].sum() < len(y), "threshold must split the sample"
        np.testing.assert_array_equal(before, after)

    def test_threshold_that_lands_on_a_score_is_still_invariant(self, scores):
        # Ties at the threshold are the case where rounding would bite.
        y, p = scores
        t = float(p[150])
        for pi_deploy in (0.10, 0.14, 0.20, 0.30):
            np.testing.assert_array_equal(
                _confusion(y, p, t),
                _confusion(y, correct(p, PI_TRAIN, pi_deploy), correct(t, PI_TRAIN, pi_deploy)),
            )

    def test_shipped_threshold_is_the_corrected_oof_threshold(self):
        # 0.280854 is the raw threshold selected on training OOF predictions
        # (evaluation_evidence/diabetes/oof_threshold.json). The config must hold
        # its corrected value, stated to 6 decimals.
        assert invert_prevalence_correction(THRESHOLD_DEPLOYED, PI_TRAIN, PI_DEPLOY) == pytest.approx(
            0.2808542713567839, abs=1e-5
        )
        assert THRESHOLD_DEPLOYED != pytest.approx(0.275), "test-set threshold must not come back"


# ═════════════════════════════════════════════════════════════════════════════
# Shared real-model fixtures
# ═════════════════════════════════════════════════════════════════════════════

DIABETES_LOW = {
    "HighBP": 0, "HighChol": 0, "CholCheck": 1, "BMI": 22.0, "Smoker": 0,
    "Stroke": 0, "HeartDiseaseorAttack": 0, "PhysActivity": 1, "Fruits": 1,
    "Veggies": 1, "HvyAlcoholConsump": 0, "AnyHealthcare": 1, "NoDocbcCost": 0,
    "GenHlth": 1, "MentHlth": 0, "PhysHlth": 0, "DiffWalk": 0, "Sex": 0,
    "Age": 3, "Education": 6, "Income": 8,
}
DIABETES_HIGH = {
    "HighBP": 1, "HighChol": 1, "CholCheck": 1, "BMI": 38.0, "Smoker": 1,
    "Stroke": 0, "HeartDiseaseorAttack": 1, "PhysActivity": 0, "Fruits": 0,
    "Veggies": 0, "HvyAlcoholConsump": 0, "AnyHealthcare": 1, "NoDocbcCost": 0,
    "GenHlth": 4, "MentHlth": 5, "PhysHlth": 15, "DiffWalk": 1, "Sex": 1,
    "Age": 10, "Education": 4, "Income": 3,
}
_BINARY = ["HighBP", "HighChol", "CholCheck", "Smoker", "Stroke", "HeartDiseaseorAttack",
           "PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump", "AnyHealthcare",
           "NoDocbcCost", "DiffWalk", "Sex"]
# Bounds from backend/schemas.py::DiabetesInput
DIABETES_MIN = {**{k: 0 for k in _BINARY}, "BMI": 10.0, "MentHlth": 0, "PhysHlth": 0,
                "GenHlth": 1, "Age": 1, "Education": 1, "Income": 1}
DIABETES_MAX = {**{k: 1 for k in _BINARY}, "BMI": 100.0, "MentHlth": 30, "PhysHlth": 30,
                "GenHlth": 5, "Age": 13, "Education": 6, "Income": 8}

HEART_CASES = {
    "typical_up_slope": {
        "Age": 55, "Sex": "M", "ChestPainType": "ATA", "RestingBP": 130,
        "Cholesterol": 250, "FastingBS": 0, "RestingECG": "Normal", "MaxHR": 150,
        "ExerciseAngina": "N", "Oldpeak": 1.5, "ST_Slope": "Up",
    },
    "asymptomatic_flat": {
        "Age": 63, "Sex": "M", "ChestPainType": "ASY", "RestingBP": 145,
        "Cholesterol": 233, "FastingBS": 1, "RestingECG": "LVH", "MaxHR": 108,
        "ExerciseAngina": "Y", "Oldpeak": 2.6, "ST_Slope": "Flat",
    },
    "young_female": {
        "Age": 34, "Sex": "F", "ChestPainType": "NAP", "RestingBP": 118,
        "Cholesterol": 210, "FastingBS": 0, "RestingECG": "Normal", "MaxHR": 175,
        "ExerciseAngina": "N", "Oldpeak": 0.0, "ST_Slope": "Up",
    },
}
# Golden heart outputs, RECAPTURED 2026-09-20 from OmniDiagRouter.predict on
# deploy/v2-platform after the heart_full_tuned.pkl Pipeline replacement
# (see backend/model_loader.py commit "feat(heart): load heart_full_tuned.pkl
# as a self-contained sklearn Pipeline"). This is a DELIBERATE break of the
# previous safety net, not drift: the heart model itself changed (new
# Pipeline, new training data, new threshold 0.3695 instead of argmax 0.5),
# so its outputs on these exact patients are expected to differ from every
# prior capture. From this commit on, any further drift here again means
# something leaked into heart that shouldn't have.
HEART_GOLDEN = {
    "asymptomatic_flat": {"prediction": 1, "confidence": 0.9833390712738037, "diagnosis": "Positive"},
    "typical_up_slope": {"prediction": 0, "confidence": 0.33861273527145386, "diagnosis": "Negative"},
    "young_female": {"prediction": 0, "confidence": 0.039755821228027344, "diagnosis": "Negative"},
}

CORRECTION_KEYS = {
    "probability_raw", "probability_corrected", "prevalence_correction_applied",
    "prevalence_train", "prevalence_deploy", "inference_threshold_raw",
    "risk_bands", "risk_bands_raw",
}
RISK_BANDS_RAW = _DIABETES_MODEL_CFG["risk_bands"]


@pytest.fixture(scope="module")
def real_router():
    return _RealOmniDiagRouter(configs_dir=_CONFIGS_DIR)


@pytest.fixture(scope="module")
def diabetes_loader(real_router):
    return real_router._get_loader("diabetes")


@pytest.fixture(scope="module")
def live_app(app, real_router):
    """The shared app, with the real router swapped in and the limiter off."""
    import backend.main as main_module

    previous_router = main_module.router
    limiter = main_module.app.state.limiter
    previous_enabled = limiter.enabled
    main_module.router = real_router
    limiter.enabled = False
    yield main_module.app
    main_module.router = previous_router
    limiter.enabled = previous_enabled


@pytest.fixture
async def live_client(live_app):
    from httpx import ASGITransport, AsyncClient

    async with AsyncClient(transport=ASGITransport(app=live_app), base_url="http://test") as c:
        yield c


async def _post_predict(client, disease, payload):
    # patient_id bypasses the response cache, so every call is computed fresh.
    return await client.post(
        f"/api/v4/{disease}/predict",
        json=payload,
        params={"patient_id": "00000000-0000-0000-0000-000000000000"},
    )


# ═════════════════════════════════════════════════════════════════════════════
# C. /predict integration
# ═════════════════════════════════════════════════════════════════════════════

class TestDiabetesPredictApi:
    @pytest.mark.parametrize("payload", [DIABETES_LOW, DIABETES_HIGH], ids=["low", "high"])
    async def test_response_carries_audit_fields(self, live_client, payload):
        resp = await _post_predict(live_client, "diabetes", payload)
        assert resp.status_code == 200, resp.text
        data = resp.json()
        for key in ("probability_raw", "probability_corrected", "inference_threshold",
                    "prevalence_correction_applied", "confidence", "prediction"):
            assert key in data, key
        assert data["prevalence_correction_applied"] is True
        assert data["prevalence_train"] == PI_TRAIN
        assert data["prevalence_deploy"] == PI_DEPLOY
        assert data["inference_threshold"] == pytest.approx(THRESHOLD_DEPLOYED)
        # The number the UI shows is the corrected one.
        assert data["confidence"] == data["probability_corrected"]

    @pytest.mark.parametrize("payload", [DIABETES_LOW, DIABETES_HIGH], ids=["low", "high"])
    async def test_corrected_is_formula_of_raw_and_lower(self, live_client, payload):
        data = (await _post_predict(live_client, "diabetes", payload)).json()
        raw, cor = data["probability_raw"], data["probability_corrected"]
        assert 0.0 < raw < 1.0
        assert cor < raw
        assert cor == pytest.approx(correct(raw), abs=1e-12)

    @pytest.mark.parametrize(
        "payload", [DIABETES_LOW, DIABETES_HIGH, DIABETES_MIN, DIABETES_MAX],
        ids=["low", "high", "min", "max"],
    )
    async def test_decision_matches_corrected_probability(self, live_client, payload):
        data = (await _post_predict(live_client, "diabetes", payload)).json()
        expected = int(data["probability_corrected"] >= data["inference_threshold"])
        assert data["prediction"] == expected
        assert data["diagnosis"] == ("Positive" if expected else "Negative")
        # ... and the raw pair gives the same decision.
        assert expected == int(data["probability_raw"] >= data["inference_threshold_raw"])

    async def test_risk_bands_are_returned_on_the_displayed_scale(self, live_client):
        # The UI colours the badge with these; they must be on the same scale as
        # `confidence`, i.e. corrected, or a Positive patient reads LOW.
        data = (await _post_predict(live_client, "diabetes", DIABETES_HIGH)).json()
        assert data["risk_bands_raw"] == pytest.approx(RISK_BANDS_RAW)
        for band, raw_value in RISK_BANDS_RAW.items():
            assert data["risk_bands"][band] == pytest.approx(correct(raw_value))
            assert data["risk_bands"][band] < raw_value
        assert data["risk_bands"]["moderate"] < data["risk_bands"]["high"]

    async def test_band_membership_is_unchanged_by_the_correction(self, live_client):
        # Same patient, same band before and after: the correction rescales.
        for payload in (DIABETES_LOW, DIABETES_HIGH, DIABETES_MIN, DIABETES_MAX):
            d = (await _post_predict(live_client, "diabetes", payload)).json()
            def band(p, cuts):
                return "HIGH" if p >= cuts["high"] else "MODERATE" if p >= cuts["moderate"] else "LOW"
            assert band(d["probability_corrected"], d["risk_bands"]) == \
                   band(d["probability_raw"], d["risk_bands_raw"])

    def test_heart_disease_info_declares_no_bands(self, real_router):
        # Heart keeps the frontend default constants; nothing leaked into it.
        assert real_router.get_disease_info("heart_disease")["risk_bands"] is None
        diabetes_bands = real_router.get_disease_info("diabetes")["risk_bands"]
        assert diabetes_bands["high"] == pytest.approx(correct(RISK_BANDS_RAW["high"]))

    async def test_fixtures_exercise_both_decisions(self, live_client):
        # Guards the test above against passing trivially on one class only.
        low = (await _post_predict(live_client, "diabetes", DIABETES_LOW)).json()
        high = (await _post_predict(live_client, "diabetes", DIABETES_HIGH)).json()
        assert low["prediction"] == 0
        assert high["prediction"] == 1

    @pytest.mark.parametrize("case", sorted(HEART_CASES))
    async def test_heart_response_has_no_correction_fields(self, live_client, case):
        # Heart still applies no prevalence correction (unaffected by the
        # Pipeline replacement), but now states its own decision threshold
        # like diabetes does — CORRECTION_KEYS (prevalence/risk-band fields)
        # must still be absent; inference_threshold is the one field heart
        # gained.
        resp = await _post_predict(live_client, "heart_disease", HEART_CASES[case])
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert CORRECTION_KEYS.isdisjoint(data), CORRECTION_KEYS & set(data)
        assert set(data) == {"prediction", "confidence", "diagnosis", "inference_threshold"}
        assert data["inference_threshold"] == pytest.approx(0.3695)


# ═════════════════════════════════════════════════════════════════════════════
# D. Heart non-regression
# ═════════════════════════════════════════════════════════════════════════════

class TestHeartNonRegression:
    @pytest.mark.parametrize("case", sorted(HEART_CASES))
    def test_router_output_unchanged(self, real_router, case):
        out = real_router.predict("heart_disease", dict(HEART_CASES[case]))
        golden = HEART_GOLDEN[case]
        assert out["prediction"] == golden["prediction"]
        assert out["diagnosis"] == golden["diagnosis"]
        assert out["confidence"] == pytest.approx(golden["confidence"], abs=1e-6)

    def test_heart_loader_is_not_the_ensemble_loader(self, real_router):
        from backend.ensemble_loader import EnsembleModelLoader

        assert not isinstance(real_router._get_loader("heart_disease"), EnsembleModelLoader)

    def test_threshold_is_read_from_the_model_file_not_a_constant(self, real_router):
        # bundle["threshold"] must be the number the .pkl actually carries,
        # and predict() must consult it at call time -- not a 0.5 argmax or
        # any other literal baked into model_loader.py. Proven by mutating
        # the loaded bundle's threshold in place and watching the decision
        # for the same patient/probability flip both ways.
        loader = real_router._get_loader("heart_disease")
        bundle = loader.model  # forces load
        assert bundle["threshold"] == pytest.approx(0.3695)

        patient = HEART_CASES["typical_up_slope"]
        proba = HEART_GOLDEN["typical_up_slope"]["confidence"]  # 0.3386...
        original_threshold = bundle["threshold"]
        try:
            bundle["threshold"] = proba + 0.05
            result_above = loader.predict(dict(patient))
            assert result_above["prediction"] == 0
            assert result_above["inference_threshold"] == pytest.approx(proba + 0.05)

            bundle["threshold"] = proba - 0.05
            result_below = loader.predict(dict(patient))
            assert result_below["prediction"] == 1
            assert result_below["inference_threshold"] == pytest.approx(proba - 0.05)
        finally:
            bundle["threshold"] = original_threshold  # real_router is module-scoped


# ═════════════════════════════════════════════════════════════════════════════
# E. Extreme and malformed input
# ═════════════════════════════════════════════════════════════════════════════

class TestDiabetesEdgeInputs:
    @pytest.mark.parametrize("payload", [DIABETES_MIN, DIABETES_MAX], ids=["min", "max"])
    def test_schema_bounds_do_not_raise(self, diabetes_loader, payload):
        out = diabetes_loader.predict(dict(payload))
        for key in ("probability_raw", "probability_corrected", "confidence"):
            assert np.isfinite(out[key])
            assert 0.0 <= out[key] <= 1.0
        assert out["prediction"] in (0, 1)

    async def test_missing_column_is_a_clear_client_error(self, live_client):
        payload = {k: v for k, v in DIABETES_HIGH.items() if k != "BMI"}
        resp = await _post_predict(live_client, "diabetes", payload)
        assert 400 <= resp.status_code < 500, (resp.status_code, resp.text)
        assert "BMI" in resp.text

    def test_missing_column_at_loader_names_the_column(self, diabetes_loader):
        payload = {k: v for k, v in DIABETES_HIGH.items() if k != "HighBP"}
        with pytest.raises(Exception) as exc:
            diabetes_loader.predict(payload)
        assert "HighBP" in str(exc.value)

    def test_loader_refuses_config_without_priors(self):
        from backend.ensemble_loader import EnsembleModelLoader

        cfg = {"model": {k: v for k, v in _DIABETES_MODEL_CFG.items() if k != "prevalence_deploy"}}
        with pytest.raises(KeyError, match="prevalence_deploy"):
            EnsembleModelLoader(cfg)
