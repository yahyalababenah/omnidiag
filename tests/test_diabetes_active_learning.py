"""
Gate 9.6 — active learning for the NHANES module.

Every test here guards a refusal. The module's job is to decline to retrain in the
situations that would quietly make the model worse, so the tests that matter are
the ones asserting it says no.
"""

import pytest

from backend.active_learning.diabetes_nhanes_candidate import (
    MAX_PREVALENCE_ENRICHMENT,
    Annotation,
    CandidateRefused,
    build_candidate,
    extract_lab_value,
    lab_confirmed_only,
    should_queue_for_review,
    transition_matrix,
    verification_bias,
)

COHORT_PREVALENCE = 0.3288


def ann(label, decision, note="HbA1c 6.1%"):
    return Annotation(features={"RIDAGEYR": 55}, label=label,
                      decision_at_prediction=decision, note=note,
                      lab_value=extract_lab_value(note))


def balanced(n=100, prevalence=COHORT_PREVALENCE):
    """An annotated set that mirrors the cohort: right prevalence, cleared present."""
    out = []
    for i in range(n):
        positive = i < int(round(n * prevalence))
        decision = ("referral" if i % 3 == 0 else
                    "uncertain" if i % 3 == 1 else "no_referral")
        out.append(ann(1 if positive else 0, decision))
    return out


# ── label provenance (hazard 1) ──────────────────────────────────────────

@pytest.mark.parametrize("note,expected", [
    ("HbA1c 6.1%", 6.1),
    ("hba1c was 5.8 on repeat", 5.8),
    ("A1c 7.2", 7.2),
    ("glycated haemoglobin 5.6", 5.6),
    ("OGTT 2h, HbA1c 6.4 confirmed", 6.4),
])
def test_a_lab_value_in_the_note_is_recognised(note, expected):
    assert extract_lab_value(note) == expected


@pytest.mark.parametrize("note", [
    None, "", "confirmed", "patient is clearly diabetic", "agree with the model",
    "looks high", "reviewed by Dr. Ahmad", "HbA1c pending",
])
def test_an_opinion_is_not_a_lab_value(note):
    """The target is an assay result. A reviewer cannot judge HbA1c >= 5.7 by eye,
    so an unconfirmed annotation would teach the model to imitate clinicians."""
    assert extract_lab_value(note) is None


def test_an_implausible_number_is_refused_not_coerced():
    """A number outside the HbA1c range is a glucose value, a date or a typo.
    Guessing which is the silent coercion this module exists to avoid."""
    assert extract_lab_value("glucose 118") is None      # not an HbA1c scale
    assert extract_lab_value("HbA1c 2026") is None
    assert extract_lab_value("A1c 0.5") is None


def test_unconfirmed_annotations_are_dropped_not_used():
    annotations = [ann(1, "referral", "HbA1c 6.3"), ann(0, "uncertain", "looks fine")]
    kept, dropped = lab_confirmed_only(annotations)
    assert len(kept) == 1 and len(dropped) == 1


def test_candidate_refused_when_too_few_are_lab_confirmed():
    annotations = [ann(1, "referral", "agree") for _ in range(50)]
    with pytest.raises(CandidateRefused, match="lab-confirmed"):
        build_candidate(annotations, cohort_prevalence=COHORT_PREVALENCE,
                        min_samples=10, output_dir="/tmp", bundle_sha256="x")


# ── verification bias (hazard 2) ─────────────────────────────────────────

def test_a_referral_only_set_is_refused():
    """In deployment only referred and uncertain patients get an HbA1c. On the
    training cycle that set has prevalence 0.409 against a true 0.321 and discards
    16.7% of all positives. Refitting on it moves the model toward over-referral
    and records that as learning."""
    annotations = [ann(1 if i < 70 else 0, "referral") for i in range(100)]
    with pytest.raises(CandidateRefused, match="not representative"):
        build_candidate(annotations, cohort_prevalence=COHORT_PREVALENCE,
                        min_samples=10, output_dir="/tmp", bundle_sha256="x")


def test_a_set_with_no_cleared_patients_is_refused():
    """A loop that never sees its own clearances cannot learn from the decision it
    is most likely to be getting wrong."""
    annotations = [ann(1 if i < 33 else 0, "uncertain") for i in range(100)]
    report = verification_bias(annotations, COHORT_PREVALENCE)
    assert report["cleared_share"] == 0.0
    assert report["representative"] is False
    assert "CLEARED" in report["reason"]


def test_a_representative_set_passes():
    report = verification_bias(balanced(), COHORT_PREVALENCE)
    assert report["representative"] is True, report["reason"]
    assert report["reason"] is None
    assert abs(report["enrichment"]) <= MAX_PREVALENCE_ENRICHMENT


def test_the_tolerance_sits_below_the_failure_it_catches():
    """The measured enrichment of an uncorrected loop on this cohort is 8.8 points.
    A tolerance at or above that would be decoration."""
    assert MAX_PREVALENCE_ENRICHMENT < 0.088


def test_bias_report_names_the_decision_mix():
    report = verification_bias(balanced(), COHORT_PREVALENCE)
    assert set(report["decision_mix"]) == {"referral", "uncertain", "no_referral"}
    assert report["n"] == 100


def test_empty_annotations_are_not_representative():
    report = verification_bias([], COHORT_PREVALENCE)
    assert report["representative"] is False and report["n"] == 0


# ── sampling ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("decision,queued", [
    ("uncertain", True), ("referral", False), ("no_referral", False),
])
def test_only_uncertain_predictions_are_queued(decision, queued):
    """Not entropy around a threshold: this module configures none, and inventing
    one to compute entropy would reintroduce the number it exists without. The
    conformal layer already states which patients it could not separate."""
    assert should_queue_for_review(decision) is queued


def test_the_sampler_needs_no_threshold():
    import inspect

    source = inspect.getsource(should_queue_for_review)
    assert "threshold" not in source.split('"""')[2]


# ── transition matrix ────────────────────────────────────────────────────

def test_all_nine_cells_are_always_present():
    cells = transition_matrix(["referral"], ["referral"])
    assert len(cells) == 9
    assert cells["referral->referral"] == 1
    assert cells["referral->no_referral"] == 0


def test_a_referral_flipping_straight_to_no_referral_has_its_own_cell():
    cells = transition_matrix(["referral", "referral"], ["no_referral", "uncertain"])
    assert cells["referral->no_referral"] == 1
    assert cells["referral->uncertain"] == 1


def test_length_mismatch_fails():
    with pytest.raises(ValueError, match="matching lengths"):
        transition_matrix(["referral"], ["referral", "uncertain"])


def test_an_unknown_decision_fails():
    with pytest.raises(ValueError, match="unknown decision"):
        transition_matrix(["referral"], ["REFER"])


# ── the promise: never promoted ──────────────────────────────────────────

def test_a_passing_candidate_is_still_not_promoted(tmp_path):
    card = build_candidate(balanced(), cohort_prevalence=COHORT_PREVALENCE,
                           min_samples=10, output_dir=str(tmp_path), bundle_sha256="abc")
    assert card["promoted"] is False
    assert "never promoted automatically" in card["promotion_policy"]
    assert card["parent_bundle_sha256"] == "abc"


def test_the_card_records_what_was_dropped_and_why(tmp_path):
    annotations = balanced() + [ann(1, "referral", "looks diabetic")] * 5
    card = build_candidate(annotations, cohort_prevalence=COHORT_PREVALENCE,
                           min_samples=10, output_dir=str(tmp_path), bundle_sha256="abc")
    assert card["annotations_supplied"] == 105
    assert card["annotations_used"] == 100
    assert card["annotations_dropped_no_lab_value"] == 5


def test_no_model_is_fitted_without_an_explicit_refit(tmp_path):
    card = build_candidate(balanced(), cohort_prevalence=COHORT_PREVALENCE,
                           min_samples=10, output_dir=str(tmp_path), bundle_sha256="abc")
    assert card["model_fitted"] is False


def test_the_live_weights_path_is_never_written(tmp_path):
    """The whole safety property, asserted on the source rather than by trusting it."""
    import inspect

    from backend.active_learning import diabetes_nhanes_candidate as mod

    source = inspect.getsource(mod)
    assert "weights_path" not in source.replace(
        "It never writes to `model.weights_path`", ""
    )
    assert "invalidate()" not in source
    assert "hot-reload" not in source.lower().replace(
        "it never hot-reloads a loader", ""
    ).replace("never hot-reloads a loader", "")


def test_the_shared_retrain_module_is_untouched():
    """`retrain.py` hard-codes an XGBoost path that does not exist for this module,
    and the heart cycle already fails on the same line. It is not edited here."""
    import inspect

    from backend.active_learning import retrain

    source = inspect.getsource(retrain)
    assert "diabetes_nhanes" not in source
    assert "ebm" not in source.lower()
