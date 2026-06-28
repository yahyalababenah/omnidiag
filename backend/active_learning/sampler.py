"""
OmniDiag — Active Learning Sampler
=====================================
Implements entropy-based uncertainty sampling for the Human-in-the-Loop pipeline.

When a prediction lands in the uncertainty band (default: 35%–65% confidence),
it is flagged as a review candidate and queued in the review_queue table.
Doctors annotate queued items; the annotations feed the next retraining cycle.

Uncertainty metric: prediction entropy
  H(p) = -p*log2(p) - (1-p)*log2(1-p)
  Max entropy = 1.0 at p=0.5 (complete uncertainty)
  High entropy (≥ 0.88) means the model is most uncertain.
"""

import math
import logging
from typing import Optional

log = logging.getLogger("omnidiag.active_learning")

# Predictions with entropy ≥ this threshold are queued for review
_DEFAULT_ENTROPY_THRESHOLD = 0.88  # corresponds to ~35–65% confidence range


def prediction_entropy(probability: float) -> float:
    """
    Binary entropy H(p) = -p*log2(p) - (1-p)*log2(1-p).
    Returns 0 for p=0 or p=1 (certain), 1.0 for p=0.5 (maximally uncertain).
    """
    p = max(1e-9, min(1 - 1e-9, probability))  # avoid log(0)
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


def should_queue_for_review(
    probability: float,
    entropy_threshold: float = _DEFAULT_ENTROPY_THRESHOLD,
) -> bool:
    """Return True if this prediction is uncertain enough to require expert review."""
    return prediction_entropy(probability) >= entropy_threshold


def uncertainty_band(probability: float) -> str:
    """Human-readable band for the prediction certainty."""
    p = probability
    if p >= 0.85 or p <= 0.15:
        return "CERTAIN"
    if p >= 0.70 or p <= 0.30:
        return "CONFIDENT"
    if p >= 0.60 or p <= 0.40:
        return "BORDERLINE"
    return "UNCERTAIN"
