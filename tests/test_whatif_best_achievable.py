"""
Tests — What-If best_achievable never exceeds the baseline
==========================================================
"Best achievable" used to be every allowed lever pushed at once. A lever can
raise the model's estimate (PhysActivity does in the diabetes model), so on
the live Space Case C showed "49.4% -> 50.4%" as its best achievable. The
estimate reported under that name must be strictly below the baseline, and
when no allowed change lowers it the response says so instead.
"""

import pytest

from backend.counterfactual_generator import (
    NO_IMPROVEMENT_MESSAGE,
    all_improvements,
    lowest_achievable,
)
from tests.test_whatif_policy import (
    DEMO,
    EDGE,
    _CONFIGS_DIR,
    _RealOmniDiagRouter,
    _validated,
)

POLICY = {
    "BMI": ("decrease", 18.5),
    "PhysActivity": ("to", 1),
    "Fruits": ("to", 1),
}
PATIENT = {"BMI": 33, "PhysActivity": 0, "Fruits": 0, "Age": 9}


def _scorer(effects):
    """Fake model: baseline 0.50 plus a fixed effect per lever that moved."""
    def score(row):
        p = 0.50
        for feat, delta in effects.items():
            if row.get(feat) != PATIENT.get(feat):
                p += delta
        return p
    return score


class TestLowestAchievable:
    def test_harmful_lever_is_left_out(self):
        # PhysActivity raises the estimate; the best combination skips it.
        score = _scorer({"BMI": -0.10, "PhysActivity": +0.30, "Fruits": -0.05})
        best, p = lowest_achievable(PATIENT, POLICY, score, 0.50)
        assert p == pytest.approx(0.35)
        assert best["PhysActivity"] == 0
        assert best["BMI"] == 18.5 and best["Fruits"] == 1

    def test_every_lever_at_once_would_have_exceeded_baseline(self):
        score = _scorer({"BMI": -0.10, "PhysActivity": +0.30, "Fruits": -0.05})
        assert score(all_improvements(PATIENT, POLICY)) > 0.50   # the old behaviour
        _, p = lowest_achievable(PATIENT, POLICY, score, 0.50)
        assert p < 0.50

    def test_nothing_lowers_the_estimate_returns_none(self):
        score = _scorer({"BMI": +0.01, "PhysActivity": +0.30, "Fruits": 0.0})
        assert lowest_achievable(PATIENT, POLICY, score, 0.50) is None

    def test_no_lever_returns_none(self):
        done = {"BMI": 18.5, "PhysActivity": 1, "Fruits": 1}
        assert lowest_achievable(done, POLICY, lambda r: 0.5, 0.5) is None


# Demo patients plus the two diabetes demo cases that are still in the
# deployed June frontend bundle (both positive, both no-crossing).
EXTRA = {
    "diabetes": {
        "layla": {"HighBP": 1, "HighChol": 1, "CholCheck": 1, "BMI": 32.4, "Smoker": 0, "Stroke": 0, "HeartDiseaseorAttack": 0, "PhysActivity": 0, "Fruits": 0, "Veggies": 0, "HvyAlcoholConsump": 0, "AnyHealthcare": 1, "NoDocbcCost": 0, "GenHlth": 3, "MentHlth": 12, "PhysHlth": 18, "DiffWalk": 1, "Sex": 0, "Age": 10, "Education": 3, "Income": 4},
        "mohammed": {"HighBP": 1, "HighChol": 1, "CholCheck": 1, "BMI": 28.7, "Smoker": 1, "Stroke": 0, "HeartDiseaseorAttack": 1, "PhysActivity": 0, "Fruits": 1, "Veggies": 0, "HvyAlcoholConsump": 0, "AnyHealthcare": 1, "NoDocbcCost": 0, "GenHlth": 4, "MentHlth": 8, "PhysHlth": 22, "DiffWalk": 1, "Sex": 1, "Age": 11, "Education": 2, "Income": 3},
    },
}


def _all_cases():
    for disease in ("heart_disease", "diabetes"):
        cases = {**DEMO[disease], **EDGE.get(disease, {}), **EXTRA.get(disease, {})}
        for name, patient in cases.items():
            yield disease, name, patient


@pytest.fixture(scope="module")
def results():
    router = _RealOmniDiagRouter(configs_dir=_CONFIGS_DIR)
    return {
        (d, n): router.counterfactuals(d, _validated(d, p)) for d, n, p in _all_cases()
    }


def _prob(scenario):
    for key in ("new_probability_corrected", "new_probability", "probability"):
        if isinstance(scenario.get(key), (int, float)):
            return float(scenario[key])
    raise AssertionError(f"no probability in {scenario}")


@pytest.mark.parametrize("disease,name,patient", list(_all_cases()),
                         ids=[f"{d}-{n}" for d, n, _ in _all_cases()])
def test_best_achievable_is_below_baseline(results, disease, name, patient):
    r = results[(disease, name)]
    best = r.get("best_achievable")
    if best is None:
        return
    base = float(r["baseline_probability"])
    assert _prob(best) < base, f"{disease}/{name}: best {_prob(best)} >= baseline {base}"
    assert best["risk_reduction_relative_pct"] > 0
    assert best["risk_reduction_absolute_pp"] > 0


@pytest.mark.parametrize("disease,name,patient", list(_all_cases()),
                         ids=[f"{d}-{n}" for d, n, _ in _all_cases()])
def test_no_best_is_explained(results, disease, name, patient):
    r = results[(disease, name)]
    if r.get("counterfactuals") or r.get("best_achievable") or r.get("status") == "not_applicable":
        return
    # Flagged, nothing crosses, nothing lowers: the message must say which.
    assert r["message"], f"{disease}/{name}: no explanation"
    assert "Referral is recommended" in r["message"]
    if "No modifiable factor is available" not in r["message"]:
        assert r["message"] == NO_IMPROVEMENT_MESSAGE
