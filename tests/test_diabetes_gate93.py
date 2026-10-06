"""
Gate 9.3 — tiered triage, independent What-If levers, side-by-side deployment.

The heart-module isolation tests at the bottom are the ones that matter most:
Task 4 exists because sharing a lever file between two diseases means a change
made for one silently re-scores the other.
"""

import importlib
import inspect

import pytest
import yaml

from backend import diabetes_what_if_levers as levers
from backend.model_backends import get_backend
from backend.schemas_clinical_action import (
    ClinicalActionPlan,
    build_clinical_action_plan,
)

CONFIG_PATH = "configs/diabetes_nhanes.yaml"

PATIENT = {
    "RIDAGEYR": 58, "RIAGENDR": 1.0, "BMXBMI": 31.2, "ADIPOSITY_BAND": 2.0,
    "SBP": 138, "DBP": 84, "BPXPLS": 78, "MCQ300C": 1.0, "CVD_ANY": 0.0,
    "PAQ650": 0.0, "PAQ665": 0.0, "LBDHDD": 41, "LBXSCH": 205, "LBXSTR": 190,
    "LBXSATSI": 28, "LBXSGTSI": 34, "LBXSCR": 0.95, "LBXSBU": 15,
    "LBXSAL": 4.2, "LBXSUA": 6.4,
}
HEALTHY = {**PATIENT, "RIDAGEYR": 28, "BMXBMI": 22.0, "ADIPOSITY_BAND": 0.0,
           "PAQ650": 1.0, "PAQ665": 1.0, "LBDHDD": 68, "SBP": 110, "DBP": 68,
           "LBXSTR": 70}
BORDERLINE = {**PATIENT, "RIDAGEYR": 45, "BMXBMI": 27.0, "ADIPOSITY_BAND": 1.0,
              "LBDHDD": 50}


@pytest.fixture(scope="module")
def config():
    with open(CONFIG_PATH) as handle:
        return yaml.safe_load(handle)


@pytest.fixture(scope="module")
def backend(config):
    return get_backend("ebm_platt_conformal")(config).load()


# ── Task 1: tiered clinical triage ───────────────────────────────────────

@pytest.mark.parametrize("decision,urgency", [
    ("referral", "High"), ("uncertain", "Medium"), ("no_referral", "Low"),
])
def test_each_decision_maps_to_its_tier(decision, urgency):
    plan = build_clinical_action_plan(decision)
    assert isinstance(plan, ClinicalActionPlan)
    assert plan.urgency == urgency
    assert plan.decision == decision


def test_uncertain_orders_a_cheap_test_not_a_workup():
    """The whole point of the Medium tier: break the ambiguity with a fast,
    low-cost test instead of an expensive confirmatory one."""
    plan = build_clinical_action_plan("uncertain")
    assert "glucose" in plan.recommended_test.lower()
    assert "HbA1c" not in plan.recommended_test


def test_referral_orders_the_confirmatory_test():
    assert "HbA1c" in build_clinical_action_plan("referral").recommended_test


def test_no_referral_orders_nothing():
    assert build_clinical_action_plan("no_referral").recommended_test == "None"


def test_every_tier_carries_a_safety_note():
    for decision in ("referral", "uncertain", "no_referral"):
        note = build_clinical_action_plan(decision).safety_note
        assert note and len(note) > 40


def test_no_referral_states_the_missed_positive_rate():
    """14.1% of genuinely dysglycaemic patients land in this tier. A clinician
    reading 'no test indicated' must be able to see that a negative here is not
    a clearance."""
    note = build_clinical_action_plan("no_referral").safety_note
    assert "14.1%" in note
    assert "not a clearance" in note


def test_uncertain_is_not_described_as_moderate_risk():
    note = build_clinical_action_plan("uncertain").safety_note
    assert "not a middle amount of risk" in note


def test_unknown_decision_raises_rather_than_defaulting():
    """A silent fall-through would hand a clinician a 'no test indicated' plan
    derived from a decision nobody recognised."""
    with pytest.raises(ValueError, match="No clinical action plan"):
        build_clinical_action_plan("REFER")  # right idea, wrong casing


def test_predictions_carry_the_plan(backend):
    for patient in (PATIENT, HEALTHY, BORDERLINE):
        result = backend.predict(patient)
        plan = result["clinical_action_plan"]
        assert plan["decision"] == result["decision"]
        assert plan["urgency"] in {"High", "Medium", "Low"}


def test_plan_is_a_plain_dict_not_a_model(backend):
    """The PDF path and the batch CSV writer never import Pydantic models."""
    assert isinstance(backend.predict(PATIENT)["clinical_action_plan"], dict)


# ── Task 2: alpha = 0.20 ─────────────────────────────────────────────────

def test_config_and_bundle_agree_on_alpha(backend, config):
    assert config["model"]["conformal"]["alpha"] == 0.20
    assert backend.bundle["conformal"]["alpha"] == 0.20


def test_config_records_both_sides_of_the_alpha_trade(config):
    """D9-07 moved alpha to reduce over-referral. The cost — more missed
    positives — has to be recorded next to the benefit, or a future reader sees
    only the chosen number."""
    decision = config["model"]["alpha_decision"]
    assert decision["chosen"] == 0.20 and decision["rejected"] == 0.15
    cost = decision["what_it_cost"]["missed_positive_rate"]
    assert cost["at_0_15"] == 0.101 and cost["at_0_20"] == 0.141
    assert config["performance"]["missed_positive_rate"] == 0.141


# ── Task 4: independent levers, zero blast radius ────────────────────────

def test_hdl_lever_only_ever_moves_up():
    """F9-32. The shared generator's `all_improvements` sets a feature to its
    bound unconditionally, which would pull a healthy HDL of 70 DOWN to 40."""
    assert levers.apply_lever(41, "increase", 60) == 60
    assert levers.apply_lever(75, "increase", 60) == 75
    assert levers.apply_lever(60, "increase", 60) == 60


def test_lowering_hdl_is_a_policy_violation():
    problems = levers.policy_violations({"LBDHDD": 70}, {"LBDHDD": 40})
    assert problems and "increase only" in problems[0]


def test_raising_hdl_beyond_the_ceiling_is_a_violation():
    problems = levers.policy_violations({"LBDHDD": 41}, {"LBDHDD": 90})
    assert problems and "ceiling" in problems[0]


def test_all_improvements_does_not_lower_a_healthy_hdl():
    improved = levers.all_improvements({**PATIENT, "LBDHDD": 75})
    assert improved["LBDHDD"] == 75


def test_hdl_simulation_returns_a_curve(backend):
    result = backend.generate_counterfactuals(PATIENT)["hdl_simulation"]
    assert result["applicable"] is True
    assert [step["hdl_mg_dl"] for step in result["steps"]] == [45.0, 50.0, 55.0, 60.0]
    # monotone decreasing risk: the EBM's HDL term is constrained decreasing
    probabilities = [step["probability"] for step in result["steps"]]
    assert probabilities == sorted(probabilities, reverse=True)


def test_hdl_simulation_is_skipped_when_already_high(backend):
    result = backend.generate_counterfactuals({**PATIENT, "LBDHDD": 75})["hdl_simulation"]
    assert result["applicable"] is False
    assert "only moves upward" in result["reason"]


def test_hdl_note_does_not_promise_a_drug():
    assert "No HDL-raising drug" in levers.HDL_LEVER_NOTE


def test_immutable_features_are_never_levers():
    assert not set(levers.IMMUTABLE) & set(levers.DIABETES_LEVERS)
    for feature in ("RIDAGEYR", "RIAGENDR", "MCQ300C", "CVD_ANY"):
        assert feature in levers.IMMUTABLE


# ── blast radius: the heart module must be untouched ─────────────────────

def test_the_heart_module_does_not_import_the_diabetes_levers():
    from backend.model_backends import heart_glm_conformal

    source = inspect.getsource(heart_glm_conformal)
    assert "diabetes_what_if_levers" not in source
    assert "schemas_clinical_action" not in source


def test_the_diabetes_levers_import_nothing_from_the_heart_module():
    """Checked on the import graph, not on the text: the module's docstring
    explains WHY it is separate from the heart path, and grepping for the word
    would flag that explanation."""
    import ast as _ast

    tree = _ast.parse(inspect.getsource(levers))
    imported = set()
    for node in _ast.walk(tree):
        if isinstance(node, _ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, _ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert not any("heart" in name or "backend" in name for name in imported), imported


def test_the_shared_generator_still_has_no_increase_kind():
    """Proof that Task 4 left the shared module alone. If someone later adds an
    `increase` kind there, this test fails and the diabetes levers should be
    reconsidered rather than silently duplicated."""
    from backend import counterfactual_generator

    source = inspect.getsource(counterfactual_generator.policy_violations)
    assert '"increase"' not in source


def test_heart_counterfactual_policy_is_unchanged():
    """The heart module's levers, read out of its own source, are the three it
    has always had."""
    from backend.model_backends import heart_glm_conformal

    source = inspect.getsource(heart_glm_conformal.HeartGlmConformalBackend.generate_counterfactuals)
    assert '"RestingBP": ("decrease", 110)' in source
    assert '"Cholesterol": ("decrease", 150)' in source
    assert '"FastingBS": ("to", 0)' in source


# ── Task 3: side-by-side deployment ──────────────────────────────────────

def test_nhanes_is_the_configured_diabetes_module():
    """Gate B3 retired BRFSS: its config is archived and its name answers 410.
    NHANES is the one diabetes module left in configs/."""
    import os
    from backend.retired_diseases import RETIRED_DISEASES

    with open(CONFIG_PATH) as handle:
        nhanes = yaml.safe_load(handle)
    assert nhanes["disease"]["name"] == "diabetes_nhanes"
    assert nhanes["model"]["family"] == "ebm_platt_conformal"
    assert not os.path.exists("configs/diabetes.yaml")
    assert RETIRED_DISEASES["diabetes"]["replaced_by"] == "diabetes_nhanes"


def test_nhanes_declares_no_threshold_and_no_bands(config):
    assert config["model"]["inference_threshold"] is None
    assert config["model"]["risk_bands"] is None


# ── Task 5: report and metrics scaffolding ───────────────────────────────

def test_report_glossary_describes_every_nhanes_feature(backend):
    from backend.llm.report_generator import _FEATURE_GLOSSARY

    for feature in backend.feature_names:
        assert feature in _FEATURE_GLOSSARY, f"{feature} would appear unexplained in a report"


def test_report_does_not_call_a_platt_interval_venn_abers():
    """Naming the wrong method in a clinical report is a factual error about how
    the number was produced."""
    from backend.llm.report_generator import _decision_block

    block = _decision_block(
        probability=0.66, label="Positive", band=None, threshold_note="",
        output_type="conformal_decision",
        probability_scale="platt_calibrated_nhanes_2015_2016",
        decision="referral", lower=0.62, upper=0.70,
    )
    assert "Venn-Abers" not in block
    assert "calibration interval" in block


def test_heart_report_still_says_venn_abers():
    from backend.llm.report_generator import _decision_block

    block = _decision_block(
        probability=0.66, label="Positive", band=None, threshold_note="",
        output_type="conformal_decision",
        probability_scale="ivap_calibrated_training_mix",
        decision="referral", lower=0.62, upper=0.70,
    )
    assert "Venn-Abers interval" in block


def test_report_states_the_nhanes_calibration_population():
    from backend.llm.report_generator import _decision_block

    block = _decision_block(
        probability=0.66, label="Positive", band=None, threshold_note="",
        output_type="conformal_decision",
        probability_scale="platt_calibrated_nhanes_2015_2016",
        decision="referral", lower=0.62, upper=0.70,
    )
    assert "not a diabetes diagnosis" in block


def test_metrics_counts_conformal_decisions():
    metrics = importlib.import_module("backend.monitoring.metrics")
    assert hasattr(metrics, "record_conformal_decision")
    metrics.record_conformal_decision("diabetes_nhanes", "uncertain")  # must not raise


def test_predicting_records_a_decision(backend):
    """Monitoring must never break serving, so this asserts the call path runs,
    not that prometheus is installed."""
    backend.predict(PATIENT)


def test_what_if_accepts_named_adiposity_band():
    """The schema example carries ADIPOSITY_BAND="high"; What-If must not 500 on it."""
    from backend.diabetes_what_if_levers import DIABETES_LEVERS, all_improvements, is_engaged
    from backend.schemas_diabetes_nhanes import DiabetesNhanesInput

    patient = dict(DiabetesNhanesInput.model_config["json_schema_extra"]["example"])
    kind, bound = DIABETES_LEVERS["ADIPOSITY_BAND"]
    assert is_engaged(patient["ADIPOSITY_BAND"], kind, bound)
    assert all_improvements(patient)["ADIPOSITY_BAND"] == "normal"


def test_band_levels_copy_matches_schema():
    from backend.schemas_diabetes_nhanes import ADIPOSITY_BAND_LEVELS as schema_levels

    assert levers.ADIPOSITY_BAND_LEVELS == schema_levels
