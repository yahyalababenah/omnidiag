"""
Unit Tests — Active Learning Sampler (backend/active_learning/sampler.py)
No database or HTTP client required.
"""

import pytest

from backend.active_learning.sampler import (
    prediction_entropy,
    should_queue_for_review,
    uncertainty_band,
)


class TestPredictionEntropy:
    async def test_entropy_at_maximum_uncertainty(self):
        assert prediction_entropy(0.5) == pytest.approx(1.0)

    async def test_entropy_at_certain_zero(self):
        # p=0 → certain negative → entropy 0
        assert prediction_entropy(0.0) == pytest.approx(0.0, abs=1e-6)

    async def test_entropy_at_certain_one(self):
        # p=1 → certain positive → entropy 0
        assert prediction_entropy(1.0) == pytest.approx(0.0, abs=1e-6)

    async def test_entropy_symmetric(self):
        assert prediction_entropy(0.3) == pytest.approx(prediction_entropy(0.7))

    async def test_entropy_monotone_towards_half(self):
        # Entropy increases as p approaches 0.5 from 0
        assert prediction_entropy(0.1) < prediction_entropy(0.3) < prediction_entropy(0.5)

    async def test_entropy_returns_float(self):
        result = prediction_entropy(0.4)
        assert isinstance(result, float)

    async def test_entropy_never_exceeds_one(self):
        for p in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]:
            assert prediction_entropy(p) <= 1.0 + 1e-9


class TestShouldQueueForReview:
    async def test_maximum_uncertainty_queued(self):
        assert should_queue_for_review(0.5) is True

    async def test_near_maximum_uncertainty_queued(self):
        assert should_queue_for_review(0.45) is True

    async def test_high_confidence_positive_not_queued(self):
        assert should_queue_for_review(0.95) is False

    async def test_high_confidence_negative_not_queued(self):
        assert should_queue_for_review(0.05) is False

    async def test_custom_threshold_lower(self):
        # With a very low threshold, even moderately uncertain predictions queue
        assert should_queue_for_review(0.4, entropy_threshold=0.5) is True

    async def test_custom_threshold_very_high(self):
        # With threshold > 1.0, nothing queues (entropy max is 1.0)
        assert should_queue_for_review(0.5, entropy_threshold=1.01) is False


class TestUncertaintyBand:
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
        assert uncertainty_band(probability) == expected_band

    async def test_boundary_high_certain(self):
        # 0.85 → CERTAIN
        assert uncertainty_band(0.85) == "CERTAIN"

    async def test_boundary_low_certain(self):
        # 0.15 → CERTAIN
        assert uncertainty_band(0.15) == "CERTAIN"

    async def test_band_returns_string(self):
        result = uncertainty_band(0.5)
        assert isinstance(result, str)

    async def test_all_bands_covered(self):
        bands = {uncertainty_band(p) for p in [0.5, 0.62, 0.75, 0.95]}
        assert bands == {"UNCERTAIN", "BORDERLINE", "CONFIDENT", "CERTAIN"}
