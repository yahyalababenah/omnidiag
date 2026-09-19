"""
OmniDiag — Prior-shift (prevalence) correction
===============================================
A classifier trained on a sample whose class balance differs from the
population it is deployed on returns probabilities on the *training* prior.
Bayes' rule moves them to the deployment prior without retraining:

    R      = (pi_deploy / (1 - pi_deploy)) / (pi_train / (1 - pi_train))
    p_real = (R * p) / (1 - p + R * p)

The map is strictly increasing on [0, 1] and fixes 0 and 1, so applying it to
both a probability and its decision threshold leaves every decision unchanged:
    p >= t   <=>   correct(p) >= correct(t)
It changes what the number means, not who is flagged.

The inverse is the same map with the two priors swapped.
"""

from typing import Union

import numpy as np

ArrayLike = Union[float, np.ndarray]


def _validate_prior(name: str, value: float) -> float:
    value = float(value)
    if not 0.0 < value < 1.0:
        raise ValueError(f"{name} must be strictly between 0 and 1, got {value}")
    return value


def prior_odds_ratio(prevalence_train: float, prevalence_deploy: float) -> float:
    """R = odds(pi_deploy) / odds(pi_train)."""
    pi_tr = _validate_prior("prevalence_train", prevalence_train)
    pi_dep = _validate_prior("prevalence_deploy", prevalence_deploy)
    return (pi_dep / (1.0 - pi_dep)) / (pi_tr / (1.0 - pi_tr))


def apply_prevalence_correction(
    p: ArrayLike, prevalence_train: float, prevalence_deploy: float
) -> ArrayLike:
    """
    Map probabilities from the training prior to the deployment prior.

    Accepts a scalar or an array; returns the same kind. Inputs outside
    [0, 1] raise, because they are not probabilities.
    """
    r = prior_odds_ratio(prevalence_train, prevalence_deploy)
    arr = np.asarray(p, dtype=np.float64)
    if np.any(~np.isfinite(arr)) or np.any((arr < 0.0) | (arr > 1.0)):
        raise ValueError("probabilities must be finite and within [0, 1]")
    # Denominator is (1 - p) + R*p, a convex combination of 1 and R > 0,
    # so it is >= min(1, R) > 0 for every p in [0, 1]: no division by zero.
    out = (r * arr) / ((1.0 - arr) + r * arr)
    return float(out) if np.ndim(out) == 0 else out


def invert_prevalence_correction(
    p_corrected: ArrayLike, prevalence_train: float, prevalence_deploy: float
) -> ArrayLike:
    """Map a deployment-prior probability back to the training prior."""
    return apply_prevalence_correction(p_corrected, prevalence_deploy, prevalence_train)
