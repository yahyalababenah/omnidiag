"""
Tests — What-If best_achievable never exceeds the baseline
==========================================================
"Best achievable" used to be every allowed lever pushed at once. A lever can
raise the model's estimate (a physical-activity lever did in the retired BRFSS
module), so on
the live Space Case C showed "49.4% -> 50.4%" as its best achievable. The
estimate reported under that name must be strictly below the baseline, and
when no allowed change lowers it the response says so instead.
"""

import pytest

from backend.counterfactual_generator import (
    NO_IMPROVEMENT_MESSAGE,
    NO_IMPROVEMENT_MESSAGE_DECISION,
    all_improvements,
    lowest_achievable,
)
from tests.test_whatif_policy import (
    DEMO,
    DISEASES,
    EDGE,
    _CONFIGS_DIR,
    _RealOmniDiagRouter,
    _marks,
    _validated,
)

# Neutral lever names: these are unit tests of the shared helpers, not of any
# module's policy. lever_a is the one that raises the estimate in the cases below.
POLICY = {
    "BMI": ("decrease", 18.5),
    "lever_a": ("to", 1),
    "lever_b": ("to", 1),
}
PATIENT = {"BMI": 33, "lever_a": 0, "lever_b": 0, "Age": 9}


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
        # lever_a raises the estimate; the best combination skips it.
        score = _scorer({"BMI": -0.10, "lever_a": +0.30, "lever_b": -0.05})
        best, p = lowest_achievable(PATIENT, POLICY, score, 0.50)
        assert p == pytest.approx(0.35)
        assert best["lever_a"] == 0
        assert best["BMI"] == 18.5 and best["lever_b"] == 1

    def test_every_lever_at_once_would_have_exceeded_baseline(self):
        score = _scorer({"BMI": -0.10, "lever_a": +0.30, "lever_b": -0.05})
        assert score(all_improvements(PATIENT, POLICY)) > 0.50   # the old behaviour
        _, p = lowest_achievable(PATIENT, POLICY, score, 0.50)
        assert p < 0.50

    def test_nothing_lowers_the_estimate_returns_none(self):
        score = _scorer({"BMI": +0.01, "lever_a": +0.30, "lever_b": 0.0})
        assert lowest_achievable(PATIENT, POLICY, score, 0.50) is None

    def test_no_lever_returns_none(self):
        done = {"BMI": 18.5, "lever_a": 1, "lever_b": 1}
        assert lowest_achievable(done, POLICY, lambda r: 0.5, 0.5) is None


# The two extra BRFSS demo cases went with that module (gate B6).
EXTRA = {}


def _patients(disease):
    return {**DEMO[disease], **EDGE.get(disease, {}), **EXTRA.get(disease, {})}


CASE_PARAMS = [
    pytest.param(d, n, p, id=f"{d}-{n}", marks=_marks(d))
    for d in DISEASES for n, p in _patients(d).items()
]


class _PerDisease(dict):
    # Same idea as test_whatif_policy.PerDiseaseResults: a disease is computed
    # on first use, so a heart case never loads the BRFSS model.
    def __init__(self, router):
        super().__init__()
        self._router = router

    def __missing__(self, disease):
        self[disease] = {
            n: self._router.counterfactuals(disease, _validated(disease, p))
            for n, p in _patients(disease).items()
        }
        return self[disease]


@pytest.fixture(scope="module")
def results():
    return _PerDisease(_RealOmniDiagRouter(configs_dir=_CONFIGS_DIR))


def _prob(scenario):
    for key in ("new_probability_corrected", "new_probability", "probability"):
        if isinstance(scenario.get(key), (int, float)):
            return float(scenario[key])
    raise AssertionError(f"no probability in {scenario}")


@pytest.mark.parametrize("disease,name,patient", CASE_PARAMS)
def test_best_achievable_is_below_baseline(results, disease, name, patient):
    r = results[disease][name]
    best = r.get("best_achievable")
    if best is None:
        return
    base = float(r["baseline_probability"])
    assert _prob(best) < base, f"{disease}/{name}: best {_prob(best)} >= baseline {base}"
    assert best["risk_reduction_relative_pct"] > 0
    assert best["risk_reduction_absolute_pp"] > 0


@pytest.mark.parametrize("disease,name,patient", CASE_PARAMS)
def test_no_best_is_explained(results, disease, name, patient):
    r = results[disease][name]
    if r.get("counterfactuals") or r.get("best_achievable") or r.get("status") == "not_applicable":
        return
    # Flagged, nothing crosses, nothing lowers: the message must say which.
    assert r["message"], f"{disease}/{name}: no explanation"
    if disease == "diabetes_nhanes":
        # NHANES words its outcome as an HbA1c test rather than a referral when
        # no lever is left; with levers that all fail it shares the decision text.
        if "No modifiable factor is available" in r["message"]:
            assert "HbA1c test is still recommended" in r["message"]
        else:
            assert r["message"] == NO_IMPROVEMENT_MESSAGE_DECISION
        return
    assert "Referral is recommended" in r["message"]
    if "No modifiable factor is available" not in r["message"]:
        assert r["message"] == NO_IMPROVEMENT_MESSAGE
