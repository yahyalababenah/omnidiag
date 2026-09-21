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
DIABETES_ALLOWED = {
    "BMI": ("decrease", 18.5),
    "PhysActivity": ("to", 1),
    "Fruits": ("to", 1),
    "Veggies": ("to", 1),
    "HvyAlcoholConsump": ("to", 0),
}
HEART_ALLOWED = {
    "RestingBP": ("decrease", 110),
    "Cholesterol": ("decrease", 150),
    "FastingBS": ("to", 0),
}
ENGINEERED = {
    "BMI_Age_Interaction", "Health_Index", "Lifestyle_Score",
    "SES_Composite", "Diabetes_Clinical_Risk",
}

# ── Demo patients (frontend/src/mockPatients.js; drift checked in section D)
DEMO = {
    "heart_disease": {
        "P-001": {"Age": 54, "Sex": "M", "ChestPainType": "ATA", "RestingBP": 140, "Cholesterol": 289, "FastingBS": 0, "RestingECG": "Normal", "MaxHR": 122, "ExerciseAngina": "N", "Oldpeak": 0, "ST_Slope": "Flat"},
        "P-002": {"Age": 62, "Sex": "F", "ChestPainType": "ASY", "RestingBP": 158, "Cholesterol": 340, "FastingBS": 1, "RestingECG": "LVH", "MaxHR": 98, "ExerciseAngina": "Y", "Oldpeak": 2.3, "ST_Slope": "Down"},
        "P-003": {"Age": 45, "Sex": "M", "ChestPainType": "NAP", "RestingBP": 120, "Cholesterol": 210, "FastingBS": 0, "RestingECG": "Normal", "MaxHR": 160, "ExerciseAngina": "N", "Oldpeak": 0.5, "ST_Slope": "Up"},
    },
    "diabetes": {
        "D-001": {"HighBP": 0, "HighChol": 0, "CholCheck": 1, "BMI": 24, "Smoker": 0, "Stroke": 0, "HeartDiseaseorAttack": 0, "PhysActivity": 1, "Fruits": 1, "Veggies": 1, "HvyAlcoholConsump": 0, "AnyHealthcare": 1, "NoDocbcCost": 0, "GenHlth": 2, "MentHlth": 0, "PhysHlth": 0, "DiffWalk": 0, "Sex": 0, "Age": 4, "Education": 6, "Income": 7},
        "D-002": {"HighBP": 1, "HighChol": 0, "CholCheck": 1, "BMI": 29, "Smoker": 0, "Stroke": 0, "HeartDiseaseorAttack": 0, "PhysActivity": 1, "Fruits": 1, "Veggies": 1, "HvyAlcoholConsump": 0, "AnyHealthcare": 1, "NoDocbcCost": 0, "GenHlth": 3, "MentHlth": 0, "PhysHlth": 2, "DiffWalk": 0, "Sex": 1, "Age": 6, "Education": 5, "Income": 6},
        "D-003": {"HighBP": 1, "HighChol": 1, "CholCheck": 1, "BMI": 33, "Smoker": 1, "Stroke": 0, "HeartDiseaseorAttack": 0, "PhysActivity": 0, "Fruits": 0, "Veggies": 0, "HvyAlcoholConsump": 0, "AnyHealthcare": 1, "NoDocbcCost": 0, "GenHlth": 4, "MentHlth": 5, "PhysHlth": 12, "DiffWalk": 1, "Sex": 1, "Age": 9, "Education": 4, "Income": 4},
        "D-004": {"HighBP": 1, "HighChol": 0, "CholCheck": 1, "BMI": 26, "Smoker": 0, "Stroke": 0, "HeartDiseaseorAttack": 0, "PhysActivity": 1, "Fruits": 1, "Veggies": 1, "HvyAlcoholConsump": 0, "AnyHealthcare": 1, "NoDocbcCost": 0, "GenHlth": 3, "MentHlth": 0, "PhysHlth": 0, "DiffWalk": 0, "Sex": 0, "Age": 6, "Education": 5, "Income": 6},
    },
}
# Extra positives that exercise every lever and the "no crossing" path.
EDGE = {
    "heart_disease": {
        "all_levers": {"Age": 66, "Sex": "M", "ChestPainType": "ASY", "RestingBP": 190, "Cholesterol": 420, "FastingBS": 1, "RestingECG": "ST", "MaxHR": 110, "ExerciseAngina": "Y", "Oldpeak": 1.5, "ST_Slope": "Flat"},
        "near_threshold": {"Age": 55, "Sex": "M", "ChestPainType": "ATA", "RestingBP": 170, "Cholesterol": 330, "FastingBS": 1, "RestingECG": "Normal", "MaxHR": 120, "ExerciseAngina": "N", "Oldpeak": 1.0, "ST_Slope": "Flat"},
    },
    "diabetes": {
        "all_levers": {"HighBP": 1, "HighChol": 1, "CholCheck": 1, "BMI": 38, "Smoker": 1, "Stroke": 0, "HeartDiseaseorAttack": 0, "PhysActivity": 0, "Fruits": 0, "Veggies": 0, "HvyAlcoholConsump": 1, "AnyHealthcare": 1, "NoDocbcCost": 1, "GenHlth": 4, "MentHlth": 10, "PhysHlth": 10, "DiffWalk": 1, "Sex": 1, "Age": 10, "Education": 3, "Income": 3},
    },
}


def _cases():
    for disease in ("heart_disease", "diabetes"):
        for name, patient in {**DEMO[disease], **EDGE.get(disease, {})}.items():
            yield disease, name, patient


def _as_pairs(scenario, patient):
    """(feature, before, after) for every change, whichever shape it came in."""
    changes = scenario["changes"]
    if isinstance(changes, list):
        return [(c["feature"], c["original_value"], c["counterfactual_value"]) for c in changes]
    return [(f, patient.get(f), v) for f, v in changes.items()]


def assert_allowed(disease, scenario, patient):
    allowed = HEART_ALLOWED if disease == "heart_disease" else DIABETES_ALLOWED
    pairs = _as_pairs(scenario, patient)
    assert pairs, "a scenario must change something"
    for feat, before, after in pairs:
        assert feat not in ENGINEERED, f"engineered feature {feat} was perturbed directly"
        assert feat in allowed, f"immutable feature {feat} changed {before} -> {after}"
        assert before is not None, f"missing lever {feat} was used"
        assert float(before) == float(patient[feat])
        kind, bound = allowed[feat]
        if kind == "decrease":
            assert float(after) < float(before), f"{feat} {before} -> {after} is not a decrease"
            assert float(after) >= bound, f"{feat} {after} below floor {bound}"
        else:
            assert float(after) == float(bound), f"{feat} {before} -> {after}, only -> {bound} allowed"


def _validated(disease, patient):
    from backend.schemas import get_schema_for_disease

    return get_schema_for_disease(disease)(**patient).model_dump()


@pytest.fixture(scope="module")
def real_router():
    return _RealOmniDiagRouter(configs_dir=_CONFIGS_DIR)


@pytest.fixture(scope="module")
def cf_results(real_router):
    return {
        (disease, name): real_router.counterfactuals(disease, _validated(disease, patient))
        for disease, name, patient in _cases()
    }


# ═════════════════════════════════════════════════════════════════════════════
# A. Policy on every demo patient
# ═════════════════════════════════════════════════════════════════════════════

class TestPolicyOnDemoPatients:
    @pytest.mark.parametrize("disease,name,patient", list(_cases()), ids=[f"{d}-{n}" for d, n, _ in _cases()])
    def test_every_scenario_respects_the_policy(self, cf_results, disease, name, patient):
        result = cf_results[(disease, name)]
        patient = _validated(disease, patient)
        for scenario in result["counterfactuals"]:
            assert_allowed(disease, scenario, patient)
        if result.get("best_achievable"):
            assert_allowed(disease, result["best_achievable"], patient)

    def test_the_cases_exercise_scenarios_and_fallbacks(self, cf_results):
        # Guards section A against passing trivially on empty results.
        with_scenarios = [k for k, r in cf_results.items() if r["counterfactuals"]]
        with_fallback = [k for k, r in cf_results.items() if r.get("best_achievable")]
        assert {d for d, _ in with_scenarios} == {"heart_disease", "diabetes"}
        assert {d for d, _ in with_fallback} == {"heart_disease", "diabetes"}

    def test_crossing_scenarios_really_cross(self, real_router, cf_results):
        for (disease, name), result in cf_results.items():
            for scenario in result["counterfactuals"]:
                assert scenario["crosses_threshold"] is True
                prob = scenario.get("new_probability_corrected", scenario.get("probability"))
                threshold = real_router.get_disease_info(disease)["inference_threshold"]
                assert prob < threshold + 1e-4, (disease, name, prob)


    def test_reported_changes_fully_explain_the_probability(self, real_router, cf_results):
        # Re-predict from the patient plus ONLY the reported changes: a hidden
        # change to any other feature would show up as a mismatch here.
        for (disease, name), result in cf_results.items():
            patient = _validated(disease, {**DEMO[disease], **EDGE.get(disease, {})}[name])
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


class TestPolicyViolationsFunction:
    def test_rejects_immutable_wrong_direction_and_floor(self):
        from backend.counterfactual_generator import DIABETES_POLICY, policy_violations

        patient = DEMO["diabetes"]["D-003"]
        assert policy_violations(patient, {"HighBP": 0}, DIABETES_POLICY)
        assert policy_violations(patient, {"Smoker": 0}, DIABETES_POLICY)
        assert policy_violations(patient, {"MentHlth": 0}, DIABETES_POLICY)
        assert policy_violations(patient, {"BMI": 35.0}, DIABETES_POLICY)        # gain
        assert policy_violations(patient, {"BMI": 17.0}, DIABETES_POLICY)        # below floor
        assert policy_violations(patient, {"Diabetes_Clinical_Risk": 1.0}, DIABETES_POLICY)
        assert not policy_violations(patient, {"BMI": 30.0, "PhysActivity": 1}, DIABETES_POLICY)

    def test_code_policy_matches_the_specification(self):
        from backend.counterfactual_generator import DIABETES_POLICY, IMMUTABLE_FEATURES

        assert {k: (kind, float(v)) for k, (kind, v) in DIABETES_POLICY.items()} == {
            k: (kind, float(v)) for k, (kind, v) in DIABETES_ALLOWED.items()
        }
        assert IMMUTABLE_FEATURES == {
            "HighBP", "HighChol", "CholCheck", "Stroke", "HeartDiseaseorAttack", "Smoker",
            "DiffWalk", "Age", "Sex", "Education", "Income", "AnyHealthcare", "NoDocbcCost",
            "GenHlth", "MentHlth", "PhysHlth",
        }


# ═════════════════════════════════════════════════════════════════════════════
# B. Defence in depth
# ═════════════════════════════════════════════════════════════════════════════

class TestFinalFilter:
    def test_diabetes_generator_never_emits_injected_violations(self, real_router, monkeypatch):
        from backend import counterfactual_generator as cg

        patient = _validated("diabetes", DEMO["diabetes"]["D-003"])

        def poisoned(self, patient_data):
            bad = []
            for change in ({"HighBP": 0.0, "HighChol": 0.0, "Smoker": 0.0, "GenHlth": 1.0},
                           {"BMI": 45.0}, {"MentHlth": 20.0}, {"BMI": 12.0, "PhysActivity": 1.0}):
                cand = dict(patient_data)
                cand.update(change)
                bad.append(cand)
            return bad

        monkeypatch.setattr(cg.CounterfactualGenerator, "_sample_candidates", poisoned)
        monkeypatch.setattr(cg, "all_improvements", lambda p, policy: {**p, "HighBP": 0.0, "HighChol": 0.0})
        result = real_router.counterfactuals("diabetes", patient)
        assert result["counterfactuals"] == []
        assert result["best_achievable"] is None

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


# ═════════════════════════════════════════════════════════════════════════════
# C. Honest result when nothing crosses
# ═════════════════════════════════════════════════════════════════════════════

class TestBestAchievable:
    def test_heart_p002_reports_best_achievable(self, cf_results):
        result = cf_results[("heart_disease", "P-002")]
        assert result["counterfactuals"] == []
        assert result["crosses_threshold"] is False
        best = result["best_achievable"]
        assert best["crosses_threshold"] is False
        assert {c["feature"] for c in best["changes"]} == {"RestingBP", "Cholesterol", "FastingBS"}
        assert 0 < best["probability"] < 0.9650428295135498
        assert best["risk_reduction_relative_pct"] > 0
        assert "Referral is recommended" in result["message"]

    def test_diabetes_d003_reports_best_achievable(self, cf_results):
        result = cf_results[("diabetes", "D-003")]
        assert result["counterfactuals"] == []
        assert result["crosses_threshold"] is False
        best = result["best_achievable"]
        assert best["crosses_threshold"] is False
        assert best["changes"] == {"BMI": 18.5, "Fruits": 1.0, "PhysActivity": 1.0, "Veggies": 1.0}
        assert best["new_probability_corrected"] < best["baseline_probability_corrected"]

    def test_crossing_patient_has_no_fallback(self, cf_results):
        result = cf_results[("diabetes", "D-002")]
        assert result["crosses_threshold"] is True
        assert result["best_achievable"] is None
        assert 1 <= len(result["counterfactuals"]) <= 3


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
    def test_identical_scenarios_under_different_hash_seeds(self):
        runs = []
        for seed in ("1", "4242"):
            env = {**os.environ, "PYTHONHASHSEED": seed}
            proc = subprocess.run(
                [sys.executable, "-c", _PROBE, _ROOT, json.dumps(DEMO)],
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
