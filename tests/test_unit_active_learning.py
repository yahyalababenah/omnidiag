"""
Unit Tests — Active Learning Sampler (backend/active_learning/sampler.py)
No database or HTTP client required.

The sampler is threshold-relative: uncertainty is measured around the
decision threshold, not around 0.5. Two regimes are tested explicitly:

  * ARGMAX (t = 0.5) — heart_disease, and any module that configures no
    threshold. Here the threshold-centred view is the identity, so every
    classic 0.5-centred expectation still holds. These tests pass the
    threshold explicitly so that assumption is visible, not implicit.
  * DIABETES (t = 0.059776, prevalence-corrected) — the case the old,
    0.5-centred implementation got wrong: it queued confident Positives
    and skipped every patient near the real decision boundary.
"""

import pytest

from backend.active_learning.sampler import (
    DEFAULT_DECISION_THRESHOLD,
    centre_on_threshold,
    prediction_entropy,
    should_queue_for_review,
    uncertainty_band,
)

ARGMAX = 0.5
DIABETES_T = 0.059776   # configs/diabetes.yaml → model.inference_threshold


def test_default_threshold_is_argmax():
    """The default exists for heart (argmax); diabetes always passes its own."""
    assert DEFAULT_DECISION_THRESHOLD == ARGMAX


# ── Argmax regime (t = 0.5) ───────────────────────────────────────────────────

class TestPredictionEntropyArgmax:
    async def test_entropy_at_maximum_uncertainty(self):
        assert prediction_entropy(0.5, ARGMAX) == pytest.approx(1.0)

    async def test_entropy_at_certain_zero(self):
        assert prediction_entropy(0.0, ARGMAX) == pytest.approx(0.0, abs=1e-6)

    async def test_entropy_at_certain_one(self):
        assert prediction_entropy(1.0, ARGMAX) == pytest.approx(0.0, abs=1e-6)

    async def test_entropy_symmetric(self):
        assert prediction_entropy(0.3, ARGMAX) == pytest.approx(prediction_entropy(0.7, ARGMAX))

    async def test_entropy_monotone_towards_half(self):
        assert (
            prediction_entropy(0.1, ARGMAX)
            < prediction_entropy(0.3, ARGMAX)
            < prediction_entropy(0.5, ARGMAX)
        )

    async def test_entropy_returns_float(self):
        assert isinstance(prediction_entropy(0.4, ARGMAX), float)

    async def test_entropy_never_exceeds_one(self):
        for p in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]:
            assert prediction_entropy(p, ARGMAX) <= 1.0 + 1e-9


class TestShouldQueueForReviewArgmax:
    async def test_maximum_uncertainty_queued(self):
        assert should_queue_for_review(0.5, ARGMAX) is True

    async def test_near_maximum_uncertainty_queued(self):
        assert should_queue_for_review(0.45, ARGMAX) is True

    async def test_high_confidence_positive_not_queued(self):
        assert should_queue_for_review(0.95, ARGMAX) is False

    async def test_high_confidence_negative_not_queued(self):
        assert should_queue_for_review(0.05, ARGMAX) is False

    async def test_custom_entropy_threshold_lower(self):
        assert should_queue_for_review(0.4, ARGMAX, entropy_threshold=0.5) is True

    async def test_custom_entropy_threshold_very_high(self):
        # entropy max is 1.0, so a threshold above it queues nothing
        assert should_queue_for_review(0.5, ARGMAX, entropy_threshold=1.01) is False


class TestUncertaintyBandArgmax:
    @pytest.mark.parametrize("probability,expected_band", [
        (0.95, "CERTAIN"),
        (0.05, "CERTAIN"),
        (0.75, "CONFIDENT"),
        (0.25, "CONFIDENT"),
        (0.62, "BORDERLINE"),
        (0.38, "BORDERLINE"),
        (0.50, "UNCERTAIN"),
    ])
    async def test_uncertainty_band_mapping(self, probability, expected_band):
        assert uncertainty_band(probability, ARGMAX) == expected_band

    async def test_boundary_high_certain(self):
        assert uncertainty_band(0.85, ARGMAX) == "CERTAIN"

    async def test_boundary_low_certain(self):
        assert uncertainty_band(0.15, ARGMAX) == "CERTAIN"

    async def test_band_returns_string(self):
        assert isinstance(uncertainty_band(0.5, ARGMAX), str)

    async def test_all_bands_covered(self):
        bands = {uncertainty_band(p, ARGMAX) for p in [0.5, 0.62, 0.75, 0.95]}
        assert bands == {"UNCERTAIN", "BORDERLINE", "CONFIDENT", "CERTAIN"}


# ── Threshold-centred view ────────────────────────────────────────────────────

class TestCentreOnThreshold:
    async def test_identity_at_argmax(self):
        for p in (0.0, 0.1, 0.37, 0.5, 0.9, 1.0):
            assert centre_on_threshold(p, ARGMAX) == pytest.approx(p, abs=1e-12)

    async def test_threshold_maps_to_half(self):
        assert centre_on_threshold(DIABETES_T, DIABETES_T) == pytest.approx(0.5, abs=1e-12)

    async def test_order_preserving(self):
        ps = [0.0, 0.01, 0.03, DIABETES_T, 0.1, 0.3, 0.65, 1.0]
        qs = [centre_on_threshold(p, DIABETES_T) for p in ps]
        assert qs == sorted(qs)

    async def test_side_of_boundary_is_preserved(self):
        for p in (0.02, 0.05, 0.07, 0.2):
            assert (p >= DIABETES_T) == (centre_on_threshold(p, DIABETES_T) >= 0.5)

    @pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
    async def test_rejects_degenerate_threshold(self, bad):
        with pytest.raises(ValueError):
            centre_on_threshold(0.3, bad)


# ── Diabetes regime (t = 0.059776, corrected scale) ───────────────────────────

class TestDiabetesThreshold:
    async def test_entropy_is_maximal_at_the_decision_threshold(self):
        assert prediction_entropy(DIABETES_T, DIABETES_T) == pytest.approx(1.0)

    async def test_near_threshold_patients_are_queued(self):
        # D-002 (0.111) and D-004 (0.081) sit just above a 0.0598 threshold.
        assert should_queue_for_review(0.1114, DIABETES_T) is True
        assert should_queue_for_review(0.0809, DIABETES_T) is True
        assert should_queue_for_review(0.045, DIABETES_T) is True

    async def test_confident_positive_is_not_queued(self):
        # 0.48 corrected is ~8x the threshold. The 0.5-centred sampler queued
        # exactly this kind of patient and nothing near the boundary.
        assert should_queue_for_review(0.4795, DIABETES_T) is False

    async def test_confident_negative_is_not_queued(self):
        assert should_queue_for_review(0.005, DIABETES_T) is False

    async def test_half_is_certain_not_uncertain(self):
        # The regression this module exists to prevent.
        assert uncertainty_band(0.5, DIABETES_T) == "CERTAIN"
        assert uncertainty_band(DIABETES_T, DIABETES_T) == "UNCERTAIN"

    async def test_zero_point_five_centred_would_have_got_it_backwards(self):
        """Document the old failure mode against the same two patients."""
        near_boundary, confident = 0.0809, 0.4795
        # Old behaviour == passing the argmax threshold on a corrected scale.
        assert should_queue_for_review(near_boundary, ARGMAX) is False
        assert should_queue_for_review(confident, ARGMAX) is True
        # Fixed behaviour.
        assert should_queue_for_review(near_boundary, DIABETES_T) is True
        assert should_queue_for_review(confident, DIABETES_T) is False
