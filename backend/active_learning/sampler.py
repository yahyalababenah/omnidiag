"""
OmniDiag — Active Learning Sampler
=====================================
Uncertainty sampling for the Human-in-the-Loop pipeline: a prediction the
model is unsure about is queued in review_queue, a doctor annotates it, and
the annotation feeds the next retraining cycle.

── Why this file is threshold-relative ────────────────────────────────────
"Unsure" means "close to the decision boundary", not "close to 0.5". Those
coincide only when the decision threshold *is* 0.5.

The diabetes module's threshold is not. It is 0.059776 on the deployment
prior (configs/diabetes.yaml → model.inference_threshold), because the
ensemble was trained on a 50/50 resample and predict() maps its output to the
~14% BRFSS prevalence. Binary entropy centred on 0.5 therefore stopped
sampling the boundary entirely: measured on the 14,139-row test split, every
one of the 4,789 queued rows sat at least 5x above the threshold and was a
confident Positive, while all 849 rows within +-20% of the threshold were
passed over.

The fix keeps entropy — it is the right uncertainty measure — but evaluates it
on a *threshold-centred* view of the probability. The prior-shift map from
backend/prevalence_correction.py is reused with (train=t, deploy=0.5): it is
strictly increasing, fixes 0 and 1, and sends the decision threshold t to
exactly 0.5. So entropy is measured symmetrically in log-odds around the
boundary, whatever the boundary is. When t == 0.5 the map is the identity and
the behaviour is exactly the old one.

All probabilities entering this module are on the DEPLOYMENT (corrected)
scale for diabetes, and on the model's own scale for heart — the same scale
as the `decision_threshold` passed alongside them. See
backend/probability_scale.py for the contract.
"""

import math
import logging

from backend.prevalence_correction import apply_prevalence_correction

log = logging.getLogger("omnidiag.active_learning")

# Predictions whose threshold-centred entropy is >= this are queued for review.
_DEFAULT_ENTROPY_THRESHOLD = 0.88

# Decision threshold to assume when a caller supplies none. 0.5 is sklearn's
# argmax cut-point, which is what heart_disease uses (it configures no
# threshold). Diabetes always passes its configured value explicitly.
DEFAULT_DECISION_THRESHOLD = 0.5

# Band cut-points, stated on the THRESHOLD-CENTRED scale where 0.5 is the
# decision boundary. They are symmetric: a value is CERTAIN when it is at
# least this far above the boundary, or the mirror image below it. Named
# constants rather than inline literals, because a bare 0.85 in a comparison
# against a probability is precisely the defect this module used to have.
_CERTAIN_MARGIN = 0.85
_CONFIDENT_MARGIN = 0.70
_BORDERLINE_MARGIN = 0.60

# The scale every score produced here is stated on, recorded alongside the
# stored uncertainty_score so a later reader can tell.
UNCERTAINTY_SCALE = "corrected"


def centre_on_threshold(probability: float, decision_threshold: float) -> float:
    """
    Map `probability` so that `decision_threshold` lands on 0.5.

    Strictly increasing and order-preserving, so it changes what the number
    means, never which side of the boundary it is on.
    """
    if not 0.0 < decision_threshold < 1.0:
        raise ValueError(
            f"decision_threshold must be strictly between 0 and 1, got {decision_threshold}"
        )
    return float(apply_prevalence_correction(probability, decision_threshold, 0.5))


def prediction_entropy(
    probability_corrected: float,
    decision_threshold: float = DEFAULT_DECISION_THRESHOLD,
) -> float:
    """
    Binary entropy of the threshold-centred probability.

    H(q) = -q*log2(q) - (1-q)*log2(1-q), where q = centre_on_threshold(p, t).
    Returns 1.0 exactly at the decision boundary (maximum uncertainty) and
    falls to 0 at p = 0 or p = 1.
    """
    q = centre_on_threshold(probability_corrected, decision_threshold)
    q = max(1e-9, min(1 - 1e-9, q))  # avoid log(0)
    return -(q * math.log2(q) + (1 - q) * math.log2(1 - q))


def should_queue_for_review(
    probability_corrected: float,
    decision_threshold: float = DEFAULT_DECISION_THRESHOLD,
    entropy_threshold: float = _DEFAULT_ENTROPY_THRESHOLD,
) -> bool:
    """Return True if this prediction sits close enough to the decision
    boundary to be worth a clinician's time."""
    return (
        prediction_entropy(probability_corrected, decision_threshold)
        >= entropy_threshold
    )


def uncertainty_band(
    probability_corrected: float,
    decision_threshold: float = DEFAULT_DECISION_THRESHOLD,
) -> str:
    """
    Human-readable certainty label, relative to the decision boundary.

    CERTAIN / CONFIDENT / BORDERLINE / UNCERTAIN, where UNCERTAIN means
    "hugging the boundary" rather than "near 0.5".
    """
    q = centre_on_threshold(probability_corrected, decision_threshold)
    if q >= _CERTAIN_MARGIN or q <= 1 - _CERTAIN_MARGIN:
        return "CERTAIN"
    if q >= _CONFIDENT_MARGIN or q <= 1 - _CONFIDENT_MARGIN:
        return "CONFIDENT"
    if q >= _BORDERLINE_MARGIN or q <= 1 - _BORDERLINE_MARGIN:
        return "BORDERLINE"
    return "UNCERTAIN"
