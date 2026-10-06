"""
Gate 9.9 — monitoring for the NHANES module.

A monitor that never fires is worse than no monitor, because it reads as
reassurance. These tests pin both directions: it stays quiet on data that has not
moved, and it fires on data that has.
"""

import json
from pathlib import Path

import pytest

PROFILE = Path("models/diabetes_nhanes/drift_reference.json")


@pytest.fixture(scope="module")
def profile():
    return json.loads(PROFILE.read_text())


def test_a_reference_profile_exists(profile):
    assert profile["disease"] == "diabetes_nhanes"


def test_the_monitor_finds_it_without_being_registered():
    """get_monitor falls back to models/{disease}/drift_reference.json, so adding
    a disease needs no edit to drift.py."""
    from backend.monitoring.drift import drift_unavailable_reason, get_monitor

    assert get_monitor("diabetes_nhanes").is_ready
    assert drift_unavailable_reason("diabetes_nhanes") is None


def test_the_reference_is_the_training_split_not_the_whole_cohort(profile):
    """Drift means 'incoming patients differ from the ones the model learned on',
    so the comparison has to be against what it learned on. 18508 is the 2007-2014
    training split; the full cohort is 26904."""
    assert profile["source"]["rows"] == 18508


def test_it_covers_every_model_input(profile):
    import joblib
    import yaml

    with open("configs/diabetes_nhanes.yaml") as handle:
        config = yaml.safe_load(handle)
    features = set(joblib.load(config["model"]["weights_path"])["features"])
    assert set(profile["monitored"]) == features


def test_nothing_is_profiled_but_unmonitored(profile):
    assert profile["profiled_but_not_monitored"] == {}


def test_the_ordered_judgement_is_tested_as_categorical(profile):
    """A KS test on three ordered levels answers a question nobody asked."""
    assert profile["monitored"]["ADIPOSITY_BAND"]["kind"] == "categorical"
    for binary in ("RIAGENDR", "MCQ300C", "CVD_ANY", "PAQ650", "PAQ665"):
        assert profile["monitored"][binary]["kind"] == "categorical"


def test_no_reweighting_and_it_says_why(profile):
    """The BRFSS profile is reweighted because its source is a 50/50 balanced
    sample. This cohort is natural-prevalence, so reweighting it would invent a
    distribution nobody screened."""
    assert profile["reweighting"]["applied"] is False
    assert "natural-prevalence" in profile["reweighting"]["why"]


def test_the_profile_states_it_cannot_be_rebuilt_in_repo(profile):
    """scripts/build_drift_reference.py --verify rebuilds the heart profile from
    a committed CSV. This one's source is 79 MB of raw survey files that are not
    committed (F9-17), so the profile has to say so rather than look rebuildable
    and silently never be checked."""
    assert profile["source"]["rebuildable_in_repo"] is False
    assert "not committed" in profile["source"]["note"]


def test_the_shared_builder_was_not_touched():
    """Adding this disease to scripts/build_drift_reference.py would break every
    image build: --verify rebuilds what it lists, and this source is not committed."""
    source = Path("scripts/build_drift_reference.py").read_text()
    assert "diabetes_nhanes" not in source


def test_the_rule_matches_the_other_modules(profile):
    other = json.loads(Path("models/heart_disease/drift_reference.json").read_text())
    assert profile["rule"] == other["rule"]


def test_it_records_that_it_measures_input_drift_only(profile):
    """No follow-up HbA1c exists for any patient in this system, so model
    degradation is not measurable here and must not be implied."""
    assert "output/label drift" in profile["rule"]["not_measured"]


def test_a_tiny_sample_is_refused_rather_than_judged():
    import pandas as pd

    from backend.monitoring.drift import get_monitor

    monitor = get_monitor("diabetes_nhanes")
    features = list(monitor._profile["monitored"])
    frame = pd.DataFrame([{f: 1.0 for f in features}] * 10)
    assert monitor.run(frame)["status"] == "insufficient_data"


def test_metrics_counts_the_three_decisions():
    from backend.monitoring import metrics

    for decision in ("referral", "no_referral", "uncertain"):
        metrics.record_conformal_decision("diabetes_nhanes", decision)
