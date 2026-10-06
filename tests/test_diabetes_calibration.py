"""
Tests — Diabetes prevalence correction and threshold (regression guard)
=======================================================================
Pins the two diabetes decision fixes:
  * threshold selected on training out-of-fold predictions, not y_test
  * Bayes prior-shift correction applied in EnsembleModelLoader.predict()

Sections
  A. correction formula — pure unit tests, no model
  B. decision invariance — correction changes the scale, never the decision
  C. /predict integration with the REAL router (heart; the BRFSS cases were
     deleted with that module, gate B7)
  D. heart non-regression — golden outputs captured before the change

The real models are loaded once per module (module-scoped fixture; the
shared `app` fixture is module-scoped, so the real router can be swapped in
and the mock restored cleanly). No network, no external data.
"""

import os
import sys
from pathlib import Path

from backend.heart_glm import stack as stack_module

import numpy as np
import pytest
import yaml

from backend.prevalence_correction import (
    apply_prevalence_correction,
    prior_odds_ratio,
)
# Captured at import time, before conftest's `app` fixture patches the class.
from backend.router import OmniDiagRouter as _RealOmniDiagRouter

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CONFIGS_DIR = os.path.join(_ROOT, "configs")

# The BRFSS module is retired and its config archived (gate B3). These are its
# last shipped values, written out so the prevalence-correction tests no longer
# need the file at import time; test_pinned_values_match_the_archived_config
# keeps them honest against it.
_ARCHIVED_DIABETES_CFG = os.path.join(_ROOT, "archive", "post_expo_2026-10", "configs", "diabetes.yaml")
PI_TRAIN = 0.5
PI_DEPLOY = 0.237
THRESHOLD_DEPLOYED = 0.108184


def invert_prevalence_correction(p_corrected, prevalence_train, prevalence_deploy):
    # The library's inverse went with ScaledProbability (gate B5). It was this
    # identity -- the same map with the priors swapped -- which still holds.
    return apply_prevalence_correction(p_corrected, prevalence_deploy, prevalence_train)


def correct(p, pi_train=PI_TRAIN, pi_deploy=PI_DEPLOY):
    return apply_prevalence_correction(p, pi_train, pi_deploy)


# ═════════════════════════════════════════════════════════════════════════════
# A. Correction formula
# ═════════════════════════════════════════════════════════════════════════════

class TestCorrectionFormula:
    def test_pinned_values_match_the_archived_config(self):
        with open(_ARCHIVED_DIABETES_CFG) as f:
            model = yaml.safe_load(f)["model"]
        assert (PI_TRAIN, PI_DEPLOY, THRESHOLD_DEPLOYED) == (
            float(model["prevalence_train"]), float(model["prevalence_deploy"]),
            float(model["inference_threshold"]),
        )
        assert RISK_BANDS_RAW == model["risk_bands"]

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

# Case names state the CLINICAL meaning of the chest pain and the API code that
# carries it, because the previous names did not and one of them was wrong in
# exactly the way HF-1 was: "typical_up_slope" sends ChestPainType "ATA", which
# is ATYPICAL angina. A case name that misdescribes its own input is how an
# inverted encoding survives a test suite.
#
#   API code -> clinical meaning (the meaning the API has always documented)
#   TA  -> typical angina        ATA -> atypical angina
#   NAP -> non-anginal pain      ASY -> asymptomatic (no anginal features)
HEART_CASES = {
    # ChestPainType "ATA" = atypical angina.
    "clinical_atypical_angina_up_slope": {
        "Age": 55, "Sex": "M", "ChestPainType": "ATA", "RestingBP": 130,
        "Cholesterol": 250, "FastingBS": 0, "RestingECG": "Normal", "MaxHR": 150,
        "ExerciseAngina": "N", "Oldpeak": 1.5, "ST_Slope": "Up",
    },
    # ChestPainType "ASY" = asymptomatic, i.e. no anginal features.
    "clinical_no_anginal_features_flat": {
        "Age": 63, "Sex": "M", "ChestPainType": "ASY", "RestingBP": 145,
        "Cholesterol": 233, "FastingBS": 1, "RestingECG": "LVH", "MaxHR": 108,
        "ExerciseAngina": "Y", "Oldpeak": 2.6, "ST_Slope": "Flat",
    },
    # ChestPainType "NAP" = non-anginal pain.
    "clinical_non_anginal_pain_young_female": {
        "Age": 34, "Sex": "F", "ChestPainType": "NAP", "RestingBP": 118,
        "Cholesterol": 210, "FastingBS": 0, "RestingECG": "Normal", "MaxHR": 175,
        "ExerciseAngina": "N", "Oldpeak": 0.0, "ST_Slope": "Up",
    },
}
# Golden heart outputs, RECAPTURED 2026-09-26 on phase8/heart-fixes-and-monitoring
# after the Spline-GLM + Venn-Abers + Mondrian conformal replacement (Gate 8.1).
# This is a DELIBERATE break of the previous net, not drift: the model, its
# feature set, its calibration and its decision rule all changed.
#
# The atypical-angina case (formerly `typical_up_slope`) is GONE rather than
# updated. Its old expectation
# (Negative, 0.3386) WAS the HF-1 defect being asserted as correct behaviour:
# UCI's chest-pain codes are inverted, so a patient the clinician records as
# typical angina scored LOW. That property now has a test of its own, stated
# as a property rather than a captured number, in
# tests/test_heart_glm_backend.py::test_typical_angina_outranks_no_anginal_features_*.
#
# `confidence` here is the Venn-Abers probability, and a patient whose
# conformal set is not a singleton is reported as a referral for further
# evaluation (prediction 1) — see backend/heart_glm/stack.py.
HEART_GOLDEN = {
    "clinical_no_anginal_features_flat": {"prediction": 1, "confidence": 0.6875, "diagnosis": "Uncertain — refer for further evaluation"},
    "clinical_non_anginal_pain_young_female": {"prediction": 0, "confidence": 0.023809523809523808, "diagnosis": "Negative"},
}

CORRECTION_KEYS = {
    "probability_raw", "probability_corrected", "prevalence_correction_applied",
    "prevalence_train", "prevalence_deploy", "inference_threshold_raw",
    "risk_bands", "risk_bands_raw",
}
RISK_BANDS_RAW = {"high": 0.7, "moderate": 0.4}


@pytest.fixture(scope="module")
def real_router():
    return _RealOmniDiagRouter(configs_dir=_CONFIGS_DIR)


def _flush_cache_sync():
    """
    Drop every cached /predict payload.

    Necessary because live_app swaps the router on the SHARED app object
    while the prediction cache stays the same. A cache key is (disease,
    patient data) and says nothing about which router produced the payload,
    so a response the MOCK router cached in another test module is a valid
    hit for the real router here — and vice versa. That surfaced as
    test_heart_response_has_no_correction_fields failing only in a full-suite
    run: it received the mock's {prediction, confidence, diagnosis} instead
    of the real response, which also carries inference_threshold.

    Flushed on both entry and exit so neither direction leaks.
    """
    try:
        from fastapi_cache import FastAPICache
        backend = FastAPICache.get_backend()
        if hasattr(backend, "_store"):
            backend._store.clear()
    except Exception:
        pass


@pytest.fixture(scope="module")
def live_app(app, real_router):
    """The shared app, with the real router swapped in and the limiter off."""
    import backend.main as main_module

    previous_router = main_module.router
    limiter = main_module.app.state.limiter
    previous_enabled = limiter.enabled
    main_module.router = real_router
    limiter.enabled = False
    _flush_cache_sync()
    yield main_module.app
    _flush_cache_sync()
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

class TestHeartPredictApi:
    def test_heart_disease_info_declares_no_bands(self, real_router):
        # Heart keeps the frontend default constants; nothing leaked into it.
        assert real_router.get_disease_info("heart_disease")["risk_bands"] is None

    @pytest.mark.parametrize("case", sorted(HEART_CASES))
    async def test_heart_response_has_no_correction_fields(self, live_client, case):
        # Heart applies no prevalence correction, so CORRECTION_KEYS must be
        # absent. It also publishes NO inference_threshold: its decision is a
        # conformal set, and a single cut-point is exactly what produced the sex
        # gap in sensitivity this model was built to close (HF-13). A threshold
        # reappearing here is a regression.
        #
        # The response was pinned to exactly three keys until Gate 8.4. It now
        # carries its decision as data -- the decision itself, the referral flag,
        # the Venn-Abers interval, and what it is calibrated to -- because every
        # consumer that lacked them was inferring from the disease name instead:
        # the frontend showed a 0.5 threshold this model does not have, and the
        # review queue judged uncertainty by entropy around that same 0.5. The
        # key set is still pinned, so a field cannot be added without a decision.
        resp = await _post_predict(live_client, "heart_disease", HEART_CASES[case])
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert CORRECTION_KEYS.isdisjoint(data), CORRECTION_KEYS & set(data)
        assert set(data) == {
            "prediction", "confidence", "diagnosis",
            "decision", "conformal_set", "decision_is_referral",
            "probability_lower", "probability_upper",
            "output_type", "probability_scale",
        }, sorted(data)
        assert "inference_threshold" not in data
        # The absences are declared, not merely missing: `output_type` is what
        # tells a consumer there is no threshold and no band to look for, so it
        # never has to guess from the disease name. Bands themselves belong to
        # GET /api/v4/diseases, which reports null for this module.
        assert data["output_type"] == "conformal_decision"
        assert data["probability_scale"] == "ivap_calibrated_training_mix"
        # And the interval brackets the probability it belongs to.
        assert data["probability_lower"] <= data["confidence"] <= data["probability_upper"]


# ═════════════════════════════════════════════════════════════════════════════
# D. Heart non-regression
# ═════════════════════════════════════════════════════════════════════════════

class TestHeartNonRegression:
    @pytest.mark.parametrize("case", sorted(HEART_GOLDEN))
    def test_router_output_unchanged(self, real_router, case):
        out = real_router.predict("heart_disease", dict(HEART_CASES[case]))
        golden = HEART_GOLDEN[case]
        assert out["prediction"] == golden["prediction"]
        assert out["diagnosis"] == golden["diagnosis"]
        assert out["confidence"] == pytest.approx(golden["confidence"], abs=1e-6)

    def test_heart_is_served_by_its_own_backend(self, real_router):
        # The ensemble loader this used to rule out was deleted with BRFSS (gate B4).
        from backend.model_backends.heart_glm_conformal import HeartGlmConformalBackend

        assert isinstance(real_router._get_loader("heart_disease"), HeartGlmConformalBackend)

    @pytest.mark.parametrize("bad_oldpeak", ["inf", "-inf", "nan"], ids=["inf", "-inf", "nan"])
    async def test_batch_isolates_a_row_with_inf_or_nan_oldpeak(self, live_client, doctor_token, bad_oldpeak):
        """
        Regression guard for HM-3 (archive/post_expo_2026-10/WEAKNESS_REGISTER.md): before Oldpeak got
        ge/le bounds, a single row with Oldpeak=inf passed pydantic
        validation, reached ModelLoader.predict_batch()'s single vectorized
        predict_proba() call, and crashed it for the WHOLE validated group
        (StandardScaler/check_array reject the entire matrix on one such
        value) -- confirmed by direct test before the fix. The three rows
        below sit in one /batch request with a good row on each side of the
        bad one, so a regression that lets inf/nan back in would show up as
        the two good rows failing too, not just the bad one.
        """
        good = dict(HEART_CASES["clinical_no_anginal_features_flat"])
        bad = dict(good)
        bad["Oldpeak"] = bad_oldpeak

        header = ",".join(good.keys())
        good_row = ",".join(str(v) for v in good.values())
        bad_row = ",".join(str(v) for v in bad.values())
        csv_bytes = "\n".join([header, good_row, bad_row, good_row]).encode("utf-8")

        resp = await live_client.post(
            "/api/v4/heart_disease/batch",
            files={"file": ("test.csv", csv_bytes, "text/csv")},
            headers={"Authorization": f"Bearer {doctor_token}"},
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()

        assert data["total"] == 3
        assert data["succeeded"] == 2, data
        assert data["failed"] == 1, data

        by_row = {r["row"]: r for r in data["results"]}
        assert by_row[1]["status"] == "ok"
        assert by_row[2]["status"] == "error"
        assert "Oldpeak" in by_row[2]["error"]
        assert by_row[3]["status"] == "ok"
        # The two good rows are identical patients -- same result, and it
        # matches the non-regression golden value for this exact patient.
        golden = HEART_GOLDEN["clinical_no_anginal_features_flat"]
        for row in (1, 3):
            assert by_row[row]["prediction"] == golden["prediction"]
            assert by_row[row]["confidence"] == pytest.approx(golden["confidence"], abs=1e-6)

    @staticmethod
    def _csv_row(patient, blank_key=None):
        return ",".join("" if k == blank_key else str(v) for k, v in patient.items())

    @pytest.mark.parametrize("missing_field", ["ST_Slope", "MaxHR"], ids=["ST_Slope", "MaxHR"])
    async def test_batch_low_impact_missing_field_succeeds_without_warning(self, live_client, doctor_token, missing_field):
        """
        Regression guard for HM-5 (archive/post_expo_2026-10/WEAKNESS_REGISTER.md). A blank field must
        not warn unless it changes the prediction, or the warning stops meaning
        anything. Since Gate 8.1 the model reads seven inputs (L3, D-25), so
        ST_Slope and MaxHR are not read at all and blanking them cannot
        possibly matter. The list now comes from the config, checked against
        the shipped artifact, instead of from a SHAP file belonging to a model
        that was never deployed (F0-1).
        """
        patient = dict(HEART_CASES["clinical_atypical_angina_up_slope"])
        header = ",".join(patient.keys())
        row = self._csv_row(patient, blank_key=missing_field)
        csv_bytes = "\n".join([header, row]).encode("utf-8")

        resp = await live_client.post(
            "/api/v4/heart_disease/batch",
            files={"file": ("test.csv", csv_bytes, "text/csv")},
            headers={"Authorization": f"Bearer {doctor_token}"},
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["succeeded"] == 1, data
        assert data["results"][0]["status"] == "ok"
        assert data["results"][0].get("data_completeness_warning") is None

    @pytest.mark.parametrize(
        "missing_field", ["Cholesterol", "FastingBS", "RestingBP"],
        ids=["Cholesterol", "FastingBS", "RestingBP"],
    )
    async def test_batch_high_impact_missing_field_succeeds_with_warning(self, live_client, doctor_token, missing_field):
        """
        Regression guard for HM-5, and for HF-11. These three are Optional on
        the schema AND read by the model, so a blank one is imputed: the row
        must still succeed (the point of HM-5) but carry
        data_completeness_warning naming the field. FastingBS is on the list
        because in the previous model a blank one raised risk by 7.6 points and
        flipped 8.1% of decisions with NO warning at all (HF-11).
        """
        patient = dict(HEART_CASES["clinical_atypical_angina_up_slope"])
        header = ",".join(patient.keys())
        row = self._csv_row(patient, blank_key=missing_field)
        csv_bytes = "\n".join([header, row]).encode("utf-8")

        resp = await live_client.post(
            "/api/v4/heart_disease/batch",
            files={"file": ("test.csv", csv_bytes, "text/csv")},
            headers={"Authorization": f"Bearer {doctor_token}"},
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["succeeded"] == 1, data
        result = data["results"][0]
        assert result["status"] == "ok"
        assert result.get("data_completeness_warning") is not None
        assert missing_field in result["data_completeness_warning"]

    async def test_batch_complete_row_has_no_warning(self, live_client, doctor_token):
        """HM-5: a fully complete row is unaffected -- no warning key at all,
        exactly like every /batch response before this change."""
        patient = dict(HEART_CASES["clinical_atypical_angina_up_slope"])
        header = ",".join(patient.keys())
        row = self._csv_row(patient)
        csv_bytes = "\n".join([header, row]).encode("utf-8")

        resp = await live_client.post(
            "/api/v4/heart_disease/batch",
            files={"file": ("test.csv", csv_bytes, "text/csv")},
            headers={"Authorization": f"Bearer {doctor_token}"},
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["succeeded"] == 1, data
        assert "data_completeness_warning" not in data["results"][0] or data["results"][0]["data_completeness_warning"] is None

    def test_decision_comes_from_the_conformal_cells_not_a_constant(self, real_router):
        # The heart decision must be read from the bundle's Mondrian cells at
        # call time, not from any literal in the backend. Proven the same way
        # the old threshold test worked: mutate the loaded artifact in place
        # and watch the same patient's decision move both ways.
        loader = real_router._get_loader("heart_disease")
        cells = loader.bundle["conformal_cells"]
        assert set(cells) == {"F0", "F1", "M0", "M1"}

        patient = HEART_CASES["clinical_no_anginal_features_flat"]
        original = dict(cells)
        try:
            # Nothing is typical of class 0 -> class 0 leaves the set, leaving
            # {1} alone: a confident referral.
            cells["M0"] = -1.0
            cells["M1"] = 1.0
            assert loader.predict(dict(patient))["diagnosis"] == "Positive"

            # Mirror image: only class 0 survives -> no referral.
            cells["M0"] = 1.0
            cells["M1"] = -1.0
            assert loader.predict(dict(patient))["diagnosis"] == "Negative"

            # Both survive -> uncertain, and uncertain is still a referral.
            cells["M0"] = 1.0
            cells["M1"] = 1.0
            uncertain = loader.predict(dict(patient))
            assert uncertain["prediction"] == 1
            assert uncertain["diagnosis"].startswith("Uncertain")
        finally:
            cells.clear()
            cells.update(original)  # real_router is module-scoped


class TestBatchChestPainCodingGuard:
    """
    Gate 8.2 — the two guards on the batch upload path.

    The raw UCI file codes ChestPainType by anginal-feature COUNT, the inverse
    of the clinical meaning this API takes (HF-1). Uploaded as-is and read as
    clinical input, 609 of its 920 rows change decision and 130 diseased
    patients lose their referral (measured, results/p8_2_guard_size.json). One
    guard refuses the file as distributed; the other lets a caller declare the
    coding and converts explicitly.
    """

    HEART_HEADER = ",".join(HEART_CASES["clinical_atypical_angina_up_slope"].keys())

    def _rows(self, *patients):
        body = [self.HEART_HEADER]
        body += [",".join(str(v) for v in p.values()) for p in patients]
        return "\n".join(body).encode("utf-8")

    async def _post(self, live_client, doctor_token, csv_bytes, **params):
        return await live_client.post(
            "/api/v4/heart_disease/batch",
            files={"file": ("test.csv", csv_bytes, "text/csv")},
            headers={"Authorization": f"Bearer {doctor_token}"},
            params=params,
        )

    # ── Layer 1: the raw research export is refused ──────────────────────────

    @pytest.mark.parametrize("marker", ["site", "HeartDisease", "num"])
    async def test_csv_with_a_raw_uci_marker_column_is_refused(self, live_client, doctor_token, marker):
        """
        A column that exists in the UCI export and in no API schema means this
        is a research file, not a clinician's upload. Refused with the reason
        and the fix, not a bare 400.
        """
        patient = dict(HEART_CASES["clinical_atypical_angina_up_slope"])
        csv_bytes = "\n".join([
            self.HEART_HEADER + f",{marker}",
            ",".join(str(v) for v in patient.values()) + ",cleveland",
        ]).encode("utf-8")

        resp = await self._post(live_client, doctor_token, csv_bytes)
        assert resp.status_code == 422, resp.text
        # The app flattens a dict detail into its standard error envelope.
        body = resp.json()
        assert body["code"] == "RAW_UCI_EXPORT_REJECTED"
        assert body["marker_columns_found"] == [marker]
        # The message must name the fix, or the user has no way forward.
        assert "chest_pain_coding=uci_raw" in body["error"]

    async def test_an_ordinary_clinical_upload_is_untouched(self, live_client, doctor_token):
        """The guard must not fire on the file a clinician actually uploads."""
        patient = dict(HEART_CASES["clinical_no_anginal_features_flat"])
        resp = await self._post(live_client, doctor_token, self._rows(patient))
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["succeeded"] == 1, data
        assert data["chest_pain_coding"] == "clinical"
        # Unchanged against the golden value: the default path did not move.
        golden = HEART_GOLDEN["clinical_no_anginal_features_flat"]
        assert data["results"][0]["confidence"] == pytest.approx(golden["confidence"], abs=1e-6)

    async def test_marker_column_is_allowed_once_the_coding_is_declared(self, live_client, doctor_token):
        """
        Declaring uci_raw is the documented way to upload the research file, so
        the marker column stops being a reason to refuse. It is not a model
        input either way.
        """
        patient = dict(HEART_CASES["clinical_atypical_angina_up_slope"])
        csv_bytes = "\n".join([
            self.HEART_HEADER + ",site",
            ",".join(str(v) for v in patient.values()) + ",cleveland",
        ]).encode("utf-8")

        resp = await self._post(live_client, doctor_token, csv_bytes, chest_pain_coding="uci_raw")
        assert resp.status_code == 200, resp.text
        assert resp.json()["chest_pain_coding"] == "uci_raw"

    # ── Layer 2: the declared coding actually converts ───────────────────────

    async def test_declared_raw_coding_scores_as_the_clinical_opposite(self, live_client, doctor_token):
        """
        The point of the whole gate. A row whose ChestPainType is the raw code
        'TA' means NO anginal features; read as clinical input it would mean
        typical angina. Declaring uci_raw must produce the same answer as
        uploading 'ASY' -- the clinical code for the same thing -- and a
        different one from uploading 'TA' as clinical.
        """
        raw = dict(HEART_CASES["clinical_atypical_angina_up_slope"], ChestPainType="TA")
        equivalent = dict(raw, ChestPainType="ASY")   # CP_RAW_TO_CLINICAL["TA"]

        as_raw = await self._post(live_client, doctor_token, self._rows(raw), chest_pain_coding="uci_raw")
        as_clinical_equivalent = await self._post(live_client, doctor_token, self._rows(equivalent))
        as_clinical_literal = await self._post(live_client, doctor_token, self._rows(raw))
        for resp in (as_raw, as_clinical_equivalent, as_clinical_literal):
            assert resp.status_code == 200, resp.text

        raw_conf = as_raw.json()["results"][0]["confidence"]
        equivalent_conf = as_clinical_equivalent.json()["results"][0]["confidence"]
        literal_conf = as_clinical_literal.json()["results"][0]["confidence"]

        assert raw_conf == pytest.approx(equivalent_conf, abs=1e-9)
        # And the inversion this gate is about is real, not cosmetic.
        assert abs(raw_conf - literal_conf) > 0.05, (raw_conf, literal_conf)

    async def test_unknown_coding_is_refused(self, live_client, doctor_token):
        patient = dict(HEART_CASES["clinical_atypical_angina_up_slope"])
        resp = await self._post(
            live_client, doctor_token, self._rows(patient), chest_pain_coding="whatever"
        )
        assert resp.status_code == 400, resp.text
        assert resp.json()["code"] == "INVALID_CHEST_PAIN_CODING"

    async def test_diabetes_declares_no_coding_and_is_unaffected(self, live_client, doctor_token):
        """
        Both guards read the disease's own config, never the disease name. A
        disease that declares no coding has none to pass, and its batch path is
        unchanged.
        """
        resp = await live_client.post(
            "/api/v4/diabetes_nhanes/batch",
            files={"file": ("d.csv", b"BMXBMI\n25.0\n", "text/csv")},
            headers={"Authorization": f"Bearer {doctor_token}"},
            params={"chest_pain_coding": "uci_raw"},
        )
        assert resp.status_code == 400, resp.text
        assert resp.json()["code"] == "CODING_NOT_APPLICABLE"

    async def test_the_coding_reaches_the_audit_row(self, live_client, doctor_token, db_session):
        """
        Gate 8.2: the coding is recorded, not just returned. A batch result can
        be re-read later; the coding it was scored under cannot, unless it was
        written down. Asserted end to end through AuditMiddleware rather than by
        setting the column directly, because what could break is the handoff.
        """
        from sqlalchemy import select

        from backend.db_models.audit_log import AuditLog

        patient = dict(HEART_CASES["clinical_atypical_angina_up_slope"])
        resp = await self._post(
            live_client, doctor_token, self._rows(patient), chest_pain_coding="uci_raw"
        )
        assert resp.status_code == 200, resp.text

        rows = (await db_session.execute(
            select(AuditLog).where(AuditLog.endpoint == "/api/v4/heart_disease/batch")
        )).scalars().all()
        assert rows, "the batch upload wrote no audit row at all"
        assert any((r.details or {}).get("chest_pain_coding") == "uci_raw" for r in rows), \
            [r.details for r in rows]

    # ── The translation map is derived, not hand-written ─────────────────────

    def test_raw_to_clinical_map_is_derived_from_the_two_maps(self):
        """
        A hand-written third map would be a third place for HF-1 to come back.
        This asserts the map is exactly the count-preserving one and that it is
        its own inverse.
        """
        from backend.heart_glm import stack

        for raw_code, clinical_code in stack.CP_RAW_TO_CLINICAL.items():
            assert stack.CP_MAP_UCI_RAW[raw_code] == stack.CP_MAP_CLINICAL[clinical_code]
            assert stack.CP_RAW_TO_CLINICAL[clinical_code] == raw_code
        assert set(stack.CP_RAW_TO_CLINICAL) == set(stack.CP_MAP_UCI_RAW)
        # No code maps to itself: the two codings disagree on all four values.
        assert not any(k == v for k, v in stack.CP_RAW_TO_CLINICAL.items())

    def test_derived_map_is_pinned_literally(self):
        """
        The derivation is sound only while the anginal-feature count is a unique
        key in both maps; a repeated count would silently drop a code and leave
        a three-entry map that still passes "count-preserving" and "own
        inverse" for the codes it kept. stack.py refuses to import in that case
        (asserted below), and this pins the result so a change has to be
        deliberate.
        """
        from backend.heart_glm import stack

        assert stack.CP_RAW_TO_CLINICAL == {
            "ASY": "TA", "NAP": "ATA", "ATA": "NAP", "TA": "ASY",
        }

    def test_a_repeated_anginal_count_refuses_to_import(self):
        """
        The check lives in the module, not only here: a build that cannot derive
        the map correctly must fail to start rather than serve mistranslated
        chest pain. Exercised by re-executing the module source with one map
        mutated, which is the only way to test an import-time guard.
        """
        source = Path(stack_module.__file__).read_text()
        mutated = source.replace(
            '{"ASY": 3.0, "NAP": 2.0, "ATA": 1.0, "TA": 0.0}',
            '{"ASY": 3.0, "NAP": 2.0, "ATA": 2.0, "TA": 0.0}',
        )
        assert mutated != source, "the map literal moved -- update this test"
        with pytest.raises(ImportError, match="repeated anginal-feature count"):
            exec(compile(mutated, stack_module.__file__, "exec"), {"__name__": "mutated_stack"})

    def test_translation_passes_through_rows_it_cannot_map(self):
        """
        A bad cell must stay a one-row validation error, not break the batch.
        """
        from backend.heart_glm import stack

        out = stack.translate_raw_codes([{"ChestPainType": "TA"}, {"ChestPainType": None}, {}])
        assert out == [{"ChestPainType": "ASY"}, {"ChestPainType": None}, {}]


class TestHeartImportanceFile:
    """
    Gate 8.2 — the importance file must describe the model that ships.
    """

    def test_importance_matches_the_shipped_bundle(self):
        """
        Same method as F0-1 used to catch the old file: recompute and compare.
        Fails if the shipped file drifts from the bundle, in value or in rank.
        """
        import subprocess

        repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        result = subprocess.run(
            [sys.executable, os.path.join(repo, "scripts/regen_heart_importance.py"), "--verify"],
            capture_output=True, text=True, cwd=repo,
        )
        assert result.returncode == 0, result.stdout + result.stderr

    def test_importance_covers_exactly_the_features_the_model_reads(self):
        """
        The old file listed Oldpeak, ExerciseAngina, MaxHR and ST_Slope -- four
        features the shipped model does not read. That is what made it wrong
        rather than merely stale.
        """
        import json

        from backend.heart_glm import stack

        repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(repo, "evaluation_evidence/heart/heart_l3_glm_importance.json")) as fh:
            data = json.load(fh)
        assert set(data["importance"]) == set(stack.MODEL_FEATURES)
        assert data["rank"][0] == "cp_anginal", "chest pain is the model's strongest feature"
        # The scale limit must travel with the numbers.
        assert "not an attribution" in data["scale_limit"].lower()


class TestReviewQueueUsesTheModelsDecision:
    """
    Gate 8.4, end to end against the database.

    Auto-queue for human review used to score entropy around a decision
    threshold for every module. Heart has none, so it got the 0.5 default, and
    the outcome was wrong both ways: a patient the model called UNCERTAIN far
    from 0.5 was not queued, and a confidently decided patient near 0.5 was. The
    endpoint now queues on the module's own decision when it publishes one.
    """

    BASE = {
        "Age": 54, "Sex": "M", "ChestPainType": "ATA", "RestingBP": 140,
        "Cholesterol": 289, "FastingBS": 0, "RestingECG": "Normal",
        "MaxHR": 122, "ExerciseAngina": "N", "Oldpeak": 0.0, "ST_Slope": "Flat",
    }

    async def _predict(self, live_client, doctor_token, patient):
        return await live_client.post(
            "/api/v4/heart_disease/predict",
            json=patient,
            headers={"Authorization": f"Bearer {doctor_token}"},
        )

    @staticmethod
    async def _stored_record(db_session, patient):
        """
        The prediction row for THIS patient, matched on its stored inputs.

        Not "the most recent heart row": created_at has one-second resolution on
        SQLite, so rows written by sibling tests in the same module tie and the
        lookup silently returned another case's record.
        """
        from sqlalchemy import select

        from backend.db_models.prediction import Prediction

        rows = (await db_session.execute(
            select(Prediction).where(Prediction.disease == "heart_disease")
        )).scalars().all()
        matched = [
            r for r in rows
            if r.input_features.get("Age") == patient["Age"]
            and r.input_features.get("ChestPainType") == patient["ChestPainType"]
            and r.input_features.get("Sex") == patient["Sex"]
            and r.input_features.get("RestingBP") == patient["RestingBP"]
        ]
        assert matched, f"no stored prediction for Age={patient['Age']} {patient['ChestPainType']}"
        return matched[-1]

    # Every case here is one where the OLD rule and the new one DISAGREE, which
    # is the only kind that tests anything: the first three fixtures tried were
    # patients both rules happened to treat alike, so the test passed even with
    # the fix reverted. Each row notes what the old entropy-around-0.5 rule did.
    @pytest.mark.parametrize(
        "patient, decision, should_queue",
        [
            # UNCERTAIN at p=0.72 -- far enough from 0.5 that the old rule left
            # it unqueued, though the model had said it could not place them.
            (dict(Age=25, Sex="M", ChestPainType="TA", RestingBP=100,
                  Cholesterol=150, RestingECG="Normal"), "uncertain", True),
            # A CONFIDENT referral at p=0.576 -- close enough to 0.5 that the old
            # rule sent a decided patient for review.
            (dict(Age=31, Sex="F", ChestPainType="TA", RestingBP=180,
                  Cholesterol=350, RestingECG="ST"), "referral", False),
            # And a confident non-referral, which neither rule queues.
            (dict(Age=54, Sex="M", ChestPainType="NAP", RestingBP=140,
                  Cholesterol=289, RestingECG="Normal"), "no_referral", False),
        ],
        ids=["uncertain_far_from_half", "confident_near_half", "no_referral"],
    )
    async def test_only_an_uncertain_decision_is_queued(
        self, live_client, doctor_token, db_session, patient, decision, should_queue
    ):
        from sqlalchemy import select

        from backend.db_models.review_queue import ReviewQueue

        patient = dict(self.BASE, **patient)
        resp = await self._predict(live_client, doctor_token, patient)
        assert resp.status_code == 200, resp.text
        # The fixture drives the rule, so a changed model must change the fixture
        # rather than silently weaken the test.
        assert resp.json()["decision"] == decision, resp.json()

        record = await self._stored_record(db_session, patient)
        # The decision and its interval are stored, not just returned.
        assert record.decision == decision
        assert record.probability_lower <= record.confidence <= record.probability_upper

        queued = (await db_session.execute(
            select(ReviewQueue).where(ReviewQueue.prediction_id == record.id)
        )).scalars().first()
        assert (queued is not None) is should_queue, (
            f"decision={decision}: queued={queued is not None}, expected {should_queue}"
        )

    async def test_the_stored_threshold_is_null_not_a_default(
        self, live_client, doctor_token, db_session
    ):
        """
        A queued heart row must not record a decision_threshold: writing 0.5
        there would make the audit trail claim the model used a cut-point.
        """
        from sqlalchemy import select

        from backend.db_models.prediction import Prediction
        from backend.db_models.review_queue import ReviewQueue

        resp = await self._predict(live_client, doctor_token, dict(self.BASE))
        assert resp.status_code == 200, resp.text
        assert resp.json()["decision"] == "uncertain"

        record = await self._stored_record(db_session, dict(self.BASE))
        queued = (await db_session.execute(
            select(ReviewQueue).where(ReviewQueue.prediction_id == record.id)
        )).scalars().first()
        assert queued is not None
        assert queued.decision_threshold is None, queued.decision_threshold
