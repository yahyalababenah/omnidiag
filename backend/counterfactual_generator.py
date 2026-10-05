"""
OmniDiag — What-If policy helpers shared by the live modules
=============================================================
The functions every module's /counterfactuals uses to stay inside its own
mutability policy. A policy maps each lever a patient can actually change to
the one direction it may move:

    ("decrease", floor)   only down, never below floor
    ("to", value)         only to value

The heart backend uses all three functions (backend/model_backends/
heart_glm_conformal.py, and the legacy ModelLoader); the NHANES module has its
own lever module with an "increase" kind (backend/diabetes_what_if_levers.py)
and takes only the decision message from here.

The DiCE-style random-sampling generator and the BRFSS diabetes lever tables
that used to live here went with that module (retired; removed in gate B5).
"""
from typing import Any, Callable, Dict, List, Optional, Tuple


def policy_violations(
    original: Dict[str, Any],
    changes: Dict[str, Any],
    policy: Dict[str, Tuple[str, float]],
) -> List[str]:
    """
    Every way `changes` (feature -> new value) breaks `policy`. Empty list =
    allowed. Shared by the heart backend and the legacy heart loader, and run
    on every scenario immediately before it is returned, so a violating
    scenario cannot be emitted even if candidate generation changes later.
    """
    problems = []
    for feat, new in sorted(changes.items()):
        if feat not in policy:
            problems.append(f"{feat}: immutable")
            continue
        old = original.get(feat)
        if old is None or new is None:
            problems.append(f"{feat}: missing value cannot be a lever")
            continue
        kind, bound = policy[feat]
        old, new = float(old), float(new)
        if new == old:
            continue
        if kind == "decrease" and not (new < old and new >= bound):
            problems.append(f"{feat}: {old} -> {new} (decrease only, floor {bound})")
        elif kind == "to" and new != float(bound):
            problems.append(f"{feat}: {old} -> {new} (may only move to {bound})")
    return problems


def all_improvements(
    patient_data: Dict[str, Any], policy: Dict[str, Tuple[str, float]]
) -> Dict[str, Any]:
    """The patient with every allowed lever pushed to its most favourable
    value (continuous to its floor, binary to its target). Levers the
    patient is missing (None) or already satisfies are left as they are."""
    improved = dict(patient_data)
    for feat, (kind, bound) in sorted(policy.items()):
        value = patient_data.get(feat)
        if value is None:
            continue
        if kind == "decrease":
            if float(value) > bound:
                improved[feat] = bound
        else:
            improved[feat] = bound
    return improved


def lowest_achievable(
    patient_data: Dict[str, Any],
    policy: Dict[str, Tuple[str, float]],
    score_fn: Callable[[Dict[str, Any]], float],
    baseline: float,
) -> Optional[Tuple[Dict[str, Any], float]]:
    """
    The allowed combination of levers with the LOWEST estimate, or None when
    no allowed change lowers it below `baseline`.

    "Every lever at once" is not necessarily the lowest: a lever can raise
    the model's estimate (one did in the retired BRFSS module), and
    reporting that combination as "best achievable" showed an estimate above
    the baseline. Candidates, all pushed to their favourable values:
      - each lever alone,
      - every lever together,
      - only the levers that lowered the estimate on their own.
    That is n + 2 model calls, not 2**n.
    """
    full = all_improvements(patient_data, policy)
    levers = [f for f in sorted(policy) if full.get(f) != patient_data.get(f)]
    if not levers:
        return None

    scored: List[Tuple[float, Dict[str, Any]]] = []
    helpful = []
    for feat in levers:
        cand = dict(patient_data)
        cand[feat] = full[feat]
        p = float(score_fn(cand))
        scored.append((p, cand))
        if p < baseline:
            helpful.append(feat)
    scored.append((float(score_fn(full)), full))
    if len(helpful) > 1 and len(helpful) < len(levers):
        cand = dict(patient_data)
        for feat in helpful:
            cand[feat] = full[feat]
        scored.append((float(score_fn(cand)), cand))

    # Lowest estimate first; on a tie, fewer changes.
    best_p, best = min(
        scored,
        key=lambda t: (t[0], sum(t[1].get(f) != patient_data.get(f) for f in policy)),
    )
    if not best_p < baseline:
        return None
    return best, best_p


NO_IMPROVEMENT_MESSAGE = (
    "No change to the modifiable factors lowers the estimated risk for this "
    "patient. The estimated risk remains above the threshold. Referral is "
    "recommended."
)

#: The same message for a module that decides without a threshold. It has none
#: to remain above, and saying so described a model that has not shipped since
#: Gate 8.1 (found by the manual run in Gate 8.8).
NO_IMPROVEMENT_MESSAGE_DECISION = (
    "No change to the modifiable factors lowers the estimated risk for this "
    "patient, and the decision does not change. Referral is recommended."
)
