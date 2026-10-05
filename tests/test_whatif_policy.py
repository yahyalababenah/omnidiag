"""
Tests — What-If (counterfactual) clinical validity
===================================================
Sections
  A. mutability policy on every demo patient (and edge cases) of both
     diseases: no returned scenario — nor best_achievable — ever changes an
     immutable feature, moves a lever the forbidden way, goes below a floor,
     or touches an engineered feature
  B. defence in depth: violating candidates injected into generation never
     come out
  C. honest result when nothing crosses the threshold (best_achievable)
  D. determinism across processes with different PYTHONHASHSEED
  E. heart /counterfactuals with missing Optional fields: never a 500

The policy below is written out independently of the code on purpose: a
later loosening of the code's policy fails these tests instead of silently
redefining what they check.
"""

import json
import os
import subprocess
import sys

import pytest

from backend.router import OmniDiagRouter as _RealOmniDiagRouter

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CONFIGS_DIR = os.path.join(_ROOT, "configs")

# ── The policy, as specified (independent copy) ─────────────────────────────
HEART_ALLOWED = {
    "RestingBP": ("decrease", 110),
    "Cholesterol": ("decrease", 150),
    "FastingBS": ("to", 0),
}
# NHANES (backend/diabetes_what_if_levers.py). "increase" is the HDL lever
# (F9-32): it may only move up, never above the ceiling, and must leave an HDL
# already above it alone.
NHANES_ALLOWED = {
    "BMXBMI": ("decrease", 24.9),
    "ADIPOSITY_BAND": ("decrease", 0),
    "SBP": ("decrease", 120),
    "DBP": ("decrease", 80),
    "LBXSTR": ("decrease", 150),
    "LBXSCH": ("decrease", 200),
    "LBXSGTSI": ("decrease", 40),
    "LBXSATSI": ("decrease", 40),
    "LBXSUA": ("decrease", 6.0),
    "LBDHDD": ("increase", 60),
    "PAQ650": ("to", 1),
    "PAQ665": ("to", 1),
}
NHANES_IMMUTABLE = {
    "RIDAGEYR", "RIAGENDR", "MCQ300C", "CVD_ANY", "BPXPLS", "LBXSCR", "LBXSBU", "LBXSAL",
}
POLICIES = {
    "heart_disease": HEART_ALLOWED,
    "diabetes_nhanes": NHANES_ALLOWED,
}
# ADIPOSITY_BAND arrives as a name; its order is what "decrease" means.
_BAND_LEVEL = {"normal": 0, "increased": 1, "high": 2}
# ── Demo patients (frontend/src/mockPatients.js; drift checked in section D)
DEMO = {
    "heart_disease": {
        "P-001": {"Age": 54, "Sex": "M", "ChestPainType": "ATA", "RestingBP": 140, "Cholesterol": 289, "FastingBS": 0, "RestingECG": "Normal", "MaxHR": 122, "ExerciseAngina": "N", "Oldpeak": 0, "ST_Slope": "Flat"},
        "P-002": {"Age": 62, "Sex": "F", "ChestPainType": "ASY", "RestingBP": 158, "Cholesterol": 340, "FastingBS": 1, "RestingECG": "LVH", "MaxHR": 98, "ExerciseAngina": "Y", "Oldpeak": 2.3, "ST_Slope": "Down"},
        "P-003": {"Age": 45, "Sex": "M", "ChestPainType": "NAP", "RestingBP": 120, "Cholesterol": 210, "FastingBS": 0, "RestingECG": "Normal", "MaxHR": 160, "ExerciseAngina": "N", "Oldpeak": 0.5, "ST_Slope": "Up"},
    },
    # N-001 is cleared (not applicable), N-002 crosses, N-003 only has a fallback.
    "diabetes_nhanes": {
        "N-001": {"RIDAGEYR": 28, "RIAGENDR": 0, "BMXBMI": 22, "ADIPOSITY_BAND": "normal", "SBP": 108, "DBP": 68, "BPXPLS": 70, "MCQ300C": 0, "CVD_ANY": 0, "PAQ650": 1, "PAQ665": 1, "LBDHDD": 66, "LBXSCH": 185, "LBXSTR": 70, "LBXSATSI": 16, "LBXSGTSI": 14, "LBXSCR": 0.85, "LBXSBU": 13, "LBXSAL": 4.3, "LBXSUA": 5},
        "N-002": {"RIDAGEYR": 45, "RIAGENDR": 1, "BMXBMI": 30, "ADIPOSITY_BAND": "high", "SBP": 126, "DBP": 80, "BPXPLS": 72, "MCQ300C": 0, "CVD_ANY": 0, "PAQ650": 0, "PAQ665": 1, "LBDHDD": 46, "LBXSCH": 195, "LBXSTR": 140, "LBXSATSI": 26, "LBXSGTSI": 30, "LBXSCR": 0.9, "LBXSBU": 14, "LBXSAL": 4.3, "LBXSUA": 5.8},
        "N-003": {"RIDAGEYR": 58, "RIAGENDR": 1, "BMXBMI": 33, "ADIPOSITY_BAND": "high", "SBP": 138, "DBP": 84, "BPXPLS": 70, "MCQ300C": 1, "CVD_ANY": 0, "PAQ650": 0, "PAQ665": 0, "LBDHDD": 38, "LBXSCH": 185, "LBXSTR": 240, "LBXSATSI": 36, "LBXSGTSI": 52, "LBXSCR": 0.85, "LBXSBU": 13, "LBXSAL": 4.3, "LBXSUA": 6.8},
    },
}
# Extra positives that exercise every lever and the "no crossing" path.
EDGE = {
    "heart_disease": {
        "all_levers": {"Age": 66, "Sex": "M", "ChestPainType": "ASY", "RestingBP": 190, "Cholesterol": 420, "FastingBS": 1, "RestingECG": "ST", "MaxHR": 110, "ExerciseAngina": "Y", "Oldpeak": 1.5, "ST_Slope": "Flat"},
        "near_threshold": {"Age": 55, "Sex": "M", "ChestPainType": "ATA", "RestingBP": 170, "Cholesterol": 330, "FastingBS": 1, "RestingECG": "Normal", "MaxHR": 120, "ExerciseAngina": "N", "Oldpeak": 1.0, "ST_Slope": "Flat"},
    },
    "diabetes_nhanes": {
        # Every one of the 12 levers engaged.
        "all_levers": {"RIDAGEYR": 50, "RIAGENDR": 1, "BMXBMI": 36, "ADIPOSITY_BAND": "high", "SBP": 150, "DBP": 95, "BPXPLS": 70, "MCQ300C": 1, "CVD_ANY": 0, "PAQ650": 0, "PAQ665": 0, "LBDHDD": 35, "LBXSCH": 260, "LBXSTR": 300, "LBXSATSI": 70, "LBXSGTSI": 80, "LBXSCR": 0.85, "LBXSBU": 13, "LBXSAL": 4.3, "LBXSUA": 8.0},
        # F9-32: an HDL already above the ceiling must never be pulled down to it.
        "hdl_above_ceiling": {"RIDAGEYR": 58, "RIAGENDR": 1, "BMXBMI": 33, "ADIPOSITY_BAND": "high", "SBP": 138, "DBP": 84, "BPXPLS": 70, "MCQ300C": 1, "CVD_ANY": 0, "PAQ650": 0, "PAQ665": 0, "LBDHDD": 75, "LBXSCH": 185, "LBXSTR": 240, "LBXSATSI": 36, "LBXSGTSI": 52, "LBXSCR": 0.85, "LBXSBU": 13, "LBXSAL": 4.3, "LBXSUA": 6.8},
        # Optional levers left blank: skipped, never guessed.
        "missing_levers": {"RIDAGEYR": 58, "RIAGENDR": 1, "BMXBMI": 33, "ADIPOSITY_BAND": "high", "SBP": None, "DBP": None, "BPXPLS": 70, "MCQ300C": 1, "CVD_ANY": 0, "PAQ650": 0, "PAQ665": 0, "LBDHDD": 38, "LBXSCH": None, "LBXSTR": None, "LBXSATSI": None, "LBXSGTSI": None, "LBXSCR": 0.85, "LBXSBU": 13, "LBXSAL": 4.3, "LBXSUA": None},
        # Still flagged with every lever already at its target: nothing to move.
        "no_lever_left": {"RIDAGEYR": 76, "RIAGENDR": 1, "BMXBMI": 24, "ADIPOSITY_BAND": "normal", "SBP": 118, "DBP": 76, "BPXPLS": 88, "MCQ300C": 1, "CVD_ANY": 1, "PAQ650": 1, "PAQ665": 1, "LBDHDD": 62, "LBXSCH": 190, "LBXSTR": 140, "LBXSATSI": 30, "LBXSGTSI": 35, "LBXSCR": 1.6, "LBXSBU": 30, "LBXSAL": 3.6, "LBXSUA": 5.5},
    },
}


DISEASES = ("heart_disease", "diabetes_nhanes")


def _patients(disease):
    return {**DEMO[disease], **EDGE.get(disease, {})}


def _cases():
    for disease in DISEASES:
        for name, patient in _patients(disease).items():
            yield disease, name, patient


def _marks(disease):
    # BRFSS diabetes cases were marked brfss; that module and its cases are gone
    # (gate B6), so no live case carries a mark.
    return []


CASE_PARAMS = [pytest.param(d, n, p, id=f"{d}-{n}", marks=_marks(d)) for d, n, p in _cases()]
DISEASE_PARAMS = [pytest.param(d, id=d, marks=_marks(d)) for d in DISEASES]


def _as_pairs(scenario, patient):
    """(feature, before, after) for every change, whichever shape it came in."""
    changes = scenario["changes"]
    if isinstance(changes, list):
        return [(c["feature"], c["original_value"], c["counterfactual_value"]) for c in changes]
    return [(f, patient.get(f), v) for f, v in changes.items()]


def _level(value):
    return float(_BAND_LEVEL[value]) if isinstance(value, str) and value in _BAND_LEVEL else float(value)


def assert_allowed(disease, scenario, patient):
    allowed = POLICIES[disease]
    pairs = _as_pairs(scenario, patient)
    assert pairs, "a scenario must change something"
    for feat, before, after in pairs:
        assert feat in allowed, f"immutable feature {feat} changed {before} -> {after}"
        assert before is not None, f"missing lever {feat} was used"
        assert _level(before) == _level(patient[feat])
        before, after = _level(before), _level(after)
        kind, bound = allowed[feat]
        if kind == "decrease":
            assert after < before, f"{feat} {before} -> {after} is not a decrease"
            assert after >= bound, f"{feat} {after} below floor {bound}"
        elif kind == "increase":
            assert after > before, f"{feat} {before} -> {after} is not an increase"
            assert after <= bound, f"{feat} {after} above ceiling {bound}"
        else:
            assert after == float(bound), f"{feat} {before} -> {after}, only -> {bound} allowed"


def _validated(disease, patient):
    from backend.schemas import get_schema_for_disease

    return get_schema_for_disease(disease)(**patient).model_dump()


@pytest.fixture(scope="module")
def real_router():
    return _RealOmniDiagRouter(configs_dir=_CONFIGS_DIR)


class PerDiseaseResults(dict):
    """{disease: {case name: /counterfactuals result}}, one disease at a time.

    A disease is computed the first time a test asks for it, so a heart test
    never loads another disease's model: when one module's files are absent
    (the BRFSS weights are not in a clean clone), only its own tests fail.
    """

    def __init__(self, router):
        super().__init__()
        self._router = router

    def __missing__(self, disease):
        self[disease] = {
            name: self._router.counterfactuals(disease, _validated(disease, patient))
            for name, patient in _patients(disease).items()
        }
        return self[disease]


@pytest.fixture(scope="module")
def cf_results(real_router):
    return PerDiseaseResults(real_router)


# ═════════════════════════════════════════════════════════════════════════════
# A. Policy on every demo patient
# ═════════════════════════════════════════════════════════════════════════════

class TestPolicyOnDemoPatients:
    @pytest.mark.parametrize("disease,name,patient", CASE_PARAMS)
    def test_every_scenario_respects_the_policy(self, cf_results, disease, name, patient):
        result = cf_results[disease][name]
        patient = _validated(disease, patient)
        for scenario in result["counterfactuals"]:
            assert_allowed(disease, scenario, patient)
        if result.get("best_achievable"):
            assert_allowed(disease, result["best_achievable"], patient)

    @pytest.mark.parametrize("disease", DISEASE_PARAMS)
    def test_the_cases_exercise_scenarios_and_fallbacks(self, cf_results, disease):
        # Guards section A against passing trivially on empty results.
        results = cf_results[disease].values()
        assert any(r["counterfactuals"] for r in results), f"{disease}: no case yields a scenario"
        assert any(r.get("best_achievable") for r in results), f"{disease}: no case yields a fallback"

    @pytest.mark.parametrize("disease", DISEASE_PARAMS)
    def test_crossing_scenarios_really_cross(self, real_router, cf_results, disease):
        """A "crossing" scenario must actually reach a negative decision.

        Read the disease's decision rule from its config rather than assuming
        one: a module that publishes an `inference_threshold` is checked
        against it, and one whose decision is a conformal set (heart, since
        Gate 8.1) publishes no threshold and is checked by re-predicting. Both
        are the same assertion — the scenario really crosses — stated in the
        terms the module actually decides in.
        """
        threshold = real_router.get_disease_info(disease)["inference_threshold"]
        for name, result in cf_results[disease].items():
            for scenario in result["counterfactuals"]:
                assert scenario["crosses_threshold"] is True
                prob = scenario.get("new_probability_corrected", scenario.get("probability"))
                if threshold is not None:
                    assert prob < threshold + 1e-4, (disease, name, prob)
                else:
                    patient = _validated(disease, _patients(disease)[name])
                    moved = {**patient, **{
                        c["feature"]: c["counterfactual_value"] for c in scenario["changes"]
                    }}
                    assert real_router.predict(disease, moved)["prediction"] == 0, (
                        disease, name, prob,
                    )


    @pytest.mark.parametrize("disease", DISEASE_PARAMS)
    def test_reported_changes_fully_explain_the_probability(self, real_router, cf_results, disease):
        # Re-predict from the patient plus ONLY the reported changes: a hidden
        # change to any other feature would show up as a mismatch here.
        for name, result in cf_results[disease].items():
            patient = _validated(disease, _patients(disease)[name])
            scenarios = result["counterfactuals"] + (
                [result["best_achievable"]] if result.get("best_achievable") else []
            )
            for scenario in scenarios:
                changed = dict(patient)
                for feat, _, after in _as_pairs(scenario, patient):
                    changed[feat] = after
                prob = real_router.predict(disease, changed)["confidence"]
                reported = scenario.get("new_probability_corrected", scenario.get("probability"))
                assert prob == pytest.approx(reported, abs=1e-4), (disease, name)


class TestNhanesPolicy:
    def test_code_policy_matches_the_specification(self):
        from backend.diabetes_what_if_levers import DIABETES_LEVERS, IMMUTABLE

        assert {k: (kind, float(v)) for k, (kind, v) in DIABETES_LEVERS.items()} == {
            k: (kind, float(v)) for k, (kind, v) in NHANES_ALLOWED.items()
        }
        assert set(IMMUTABLE) == NHANES_IMMUTABLE

    def test_every_input_is_either_a_lever_or_immutable(self, real_router):
        # A field added to the schema must be classified on purpose, not left
        # to whatever the generator does with an unknown feature.
        from backend.schemas import get_schema_for_disease

        fields = set(get_schema_for_disease("diabetes_nhanes").model_fields)
        assert fields == set(NHANES_ALLOWED) | NHANES_IMMUTABLE
        assert not set(NHANES_ALLOWED) & NHANES_IMMUTABLE

    def test_an_hdl_above_the_ceiling_is_never_moved(self, cf_results):
        # F9-32: a healthy HDL of 75 must not be "improved" down to 60.
        result = cf_results["diabetes_nhanes"]["hdl_above_ceiling"]
        scenarios = result["counterfactuals"] + (
            [result["best_achievable"]] if result.get("best_achievable") else []
        )
        assert scenarios, "the case must produce something to check"
        for scenario in scenarios:
            assert "LBDHDD" not in {c["feature"] for c in scenario["changes"]}

    def test_all_improvements_leaves_a_healthy_hdl_alone(self):
        # The direct F9-32 guard. The scenario test above can pass even with the
        # bug back, because lowest_achievable drops a lever that raises risk, and
        # lowering HDL does.
        from backend.diabetes_what_if_levers import all_improvements

        patient = _patients("diabetes_nhanes")["hdl_above_ceiling"]
        assert all_improvements(patient)["LBDHDD"] == patient["LBDHDD"]
        low = {**patient, "LBDHDD": 38}
        assert all_improvements(low)["LBDHDD"] == 60

    def test_a_missing_lever_is_skipped_not_guessed(self, cf_results):
        patient = _patients("diabetes_nhanes")["missing_levers"]
        missing = {f for f, v in patient.items() if v is None}
        result = cf_results["diabetes_nhanes"]["missing_levers"]
        scenarios = result["counterfactuals"] + (
            [result["best_achievable"]] if result.get("best_achievable") else []
        )
        assert scenarios, "the case must produce something to check"
        for scenario in scenarios:
            assert not missing & {c["feature"] for c in scenario["changes"]}


# ═════════════════════════════════════════════════════════════════════════════
# B. Defence in depth
# ═════════════════════════════════════════════════════════════════════════════

class TestFinalFilter:
    def test_heart_loader_never_emits_injected_violations(self, real_router, monkeypatch):
        from backend import counterfactual_generator as cg

        patient = _validated("heart_disease", DEMO["heart_disease"]["P-002"])
        # Forbidden levers that would flip this patient: every candidate
        # built from all_improvements() now also rewrites immutable features.
        monkeypatch.setattr(cg, "all_improvements", lambda p, policy: {
            **p, "ChestPainType": "ATA", "ExerciseAngina": "N", "Oldpeak": 0.0,
            "ST_Slope": "Up", "MaxHR": 180, "RestingBP": 120,
        })
        result = real_router.counterfactuals("heart_disease", patient)
        for scenario in result["counterfactuals"]:
            assert_allowed("heart_disease", scenario, patient)
        if result.get("best_achievable"):
            assert_allowed("heart_disease", result["best_achievable"], patient)


    def test_nhanes_backend_never_emits_injected_violations(self, real_router, monkeypatch):
        from backend.model_backends import diabetes_ebm_conformal as nhanes

        patient = _validated("diabetes_nhanes", EDGE["diabetes_nhanes"]["hdl_above_ceiling"])
        # Every candidate now also rewrites immutable facts and pulls the healthy
        # HDL down (the F9-32 failure) -- all of which would flip the decision.
        monkeypatch.setattr(nhanes, "all_improvements", lambda p, policy=None: {
            **p, "RIDAGEYR": 20.0, "MCQ300C": 0, "CVD_ANY": 0, "LBXSCR": 0.6,
            "LBDHDD": 60.0, "BMXBMI": 22.0,
        })
        result = real_router.counterfactuals("diabetes_nhanes", patient)
        assert result["counterfactuals"] == []
        assert result["best_achievable"] is None


# ═════════════════════════════════════════════════════════════════════════════
# C. Honest result when nothing crosses
# ═════════════════════════════════════════════════════════════════════════════

class TestBestAchievable:
    def test_heart_p002_reports_best_achievable(self, cf_results):
        result = cf_results["heart_disease"]["P-002"]
        assert result["counterfactuals"] == []
        assert result["crosses_threshold"] is False
        best = result["best_achievable"]
        assert best["crosses_threshold"] is False
        assert {c["feature"] for c in best["changes"]} == {"RestingBP", "Cholesterol", "FastingBS"}
        assert 0 < best["probability"] < 0.9650428295135498
        assert best["risk_reduction_relative_pct"] > 0
        assert "Referral is recommended" in result["message"]

    def test_nhanes_n003_reports_best_achievable(self, cf_results):
        result = cf_results["diabetes_nhanes"]["N-003"]
        assert result["counterfactuals"] == []
        assert result["crosses_threshold"] is False
        best = result["best_achievable"]
        assert best["crosses_threshold"] is False
        assert best["probability"] < result["baseline_probability"]
        assert best["risk_reduction_relative_pct"] > 0
        # NHANES recommends an HbA1c test, not a "referral" (heart's wording).
        assert "HbA1c test" in result["message"]

    def test_nhanes_crossing_patient_has_no_fallback(self, cf_results):
        result = cf_results["diabetes_nhanes"]["N-002"]
        assert result["crosses_threshold"] is True
        assert result["best_achievable"] is None
        assert len(result["counterfactuals"]) == 1

    def test_nhanes_cleared_patient_is_not_applicable(self, cf_results):
        result = cf_results["diabetes_nhanes"]["N-001"]
        assert result["status"] == "not_applicable"
        assert result["counterfactuals"] == []


# ═════════════════════════════════════════════════════════════════════════════
# D. Determinism across processes
# ═════════════════════════════════════════════════════════════════════════════

_PROBE = r"""
import json, logging, sys, warnings
warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
sys.path.insert(0, sys.argv[1])
from backend.router import OmniDiagRouter
from backend.schemas import get_schema_for_disease
demo = json.loads(sys.argv[2])
r = OmniDiagRouter(sys.argv[1] + "/configs")
out = {}
for disease, patients in demo.items():
    for name, p in patients.items():
        res = r.counterfactuals(disease, get_schema_for_disease(disease)(**p).model_dump())
        out[disease + "/" + name] = {"counterfactuals": res["counterfactuals"],
                                     "best_achievable": res.get("best_achievable")}
print(json.dumps(out, sort_keys=True))
"""


class TestDeterminism:
    @pytest.mark.parametrize("disease", DISEASE_PARAMS)
    def test_identical_scenarios_under_different_hash_seeds(self, disease):
        runs = []
        for seed in ("1", "4242"):
            env = {**os.environ, "PYTHONHASHSEED": seed}
            proc = subprocess.run(
                [sys.executable, "-c", _PROBE, _ROOT, json.dumps({disease: DEMO[disease]})],
                capture_output=True, text=True, env=env, timeout=900,
            )
            assert proc.returncode == 0, proc.stderr[-2000:]
            runs.append(json.loads(proc.stdout.strip().splitlines()[-1]))
        assert runs[0] == runs[1]

    def test_embedded_demo_patients_match_the_frontend(self):
        src = os.path.join(_ROOT, "frontend", "src", "mockPatients.js")
        js = "import(process.argv[1]).then(m => console.log(JSON.stringify(m.default)))"
        try:
            import shutil
            import tempfile

            with tempfile.TemporaryDirectory() as tmp:
                copy = os.path.join(tmp, "mock.mjs")
                shutil.copy(src, copy)
                out = subprocess.run(["node", "-e", js, copy], capture_output=True,
                                     text=True, check=True, timeout=60).stdout
        except (OSError, subprocess.CalledProcessError):
            pytest.skip("node unavailable")
        mock = json.loads(out)
        for disease, patients in DEMO.items():
            frontend = {p["id"]: p["data"] for p in mock[disease]}
            assert frontend == patients


# ═════════════════════════════════════════════════════════════════════════════
# E. Missing Optional fields (heart): never a 500
# ═════════════════════════════════════════════════════════════════════════════

MISSING_OPTIONAL = {
    "Age": 58, "Sex": "M", "ChestPainType": "ASY", "RestingECG": "Normal",
    "ExerciseAngina": "Y", "RestingBP": None, "Cholesterol": None,
    "FastingBS": None, "MaxHR": None, "Oldpeak": None, "ST_Slope": None,
}


@pytest.fixture(scope="module")
def live_app(app, real_router):
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
async def live_client(live_app, db_tables):
    from httpx import ASGITransport, AsyncClient

    async with AsyncClient(transport=ASGITransport(app=live_app), base_url="http://test") as c:
        yield c


class TestHeartMissingOptional:
    async def test_all_optional_missing_is_not_a_500(self, live_client):
        resp = await live_client.post("/api/v4/heart_disease/counterfactuals", json=MISSING_OPTIONAL)
        assert resp.status_code == 200, resp.text
        data = resp.json()
        # No lever supplied -> nothing to move; never a guessed value.
        assert data["counterfactuals"] == []
        assert data["best_achievable"] is None
        assert "No modifiable factor is available" in data["message"]

    @pytest.mark.parametrize("missing", ["RestingBP", "Cholesterol", "FastingBS"])
    async def test_a_missing_lever_is_skipped_not_guessed(self, live_client, missing):
        patient = dict(EDGE["heart_disease"]["all_levers"])
        patient[missing] = None
        resp = await live_client.post("/api/v4/heart_disease/counterfactuals", json=patient)
        assert resp.status_code == 200, resp.text
        data = resp.json()
        for scenario in data["counterfactuals"] + ([data["best_achievable"]] if data.get("best_achievable") else []):
            assert missing not in {c["feature"] for c in scenario["changes"]}
            assert_allowed("heart_disease", scenario, _validated("heart_disease", patient))
