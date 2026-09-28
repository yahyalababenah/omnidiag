"""
What-If levers for the NHANES dysglycaemia module (Gate 9.3, F9-32).

A separate module from `backend/counterfactual_generator.py` on purpose, and the
reason is clinical rather than architectural.

In cardiology a lipid lever is read as an absolute target: get LDL down, get HDL
up, each toward a guideline number. In dysglycaemia a low HDL is not a lipid
problem to be corrected on its own — it is a *marker* of insulin resistance, and
it moves together with triglycerides, waist and inactivity. Wiring the same lever
object into both modules would mean a change made for one disease silently
re-scores the other. The heart module's counterfactual path must keep behaving
exactly as it does today, so nothing here is imported by it and no file it reads
is modified.

The shared generator also cannot express this lever. It understands only
`decrease` and `to`. Applying it to HDL would call `all_improvements`, which sets
a feature to its bound UNCONDITIONALLY — pulling a healthy HDL of 70 mg/dL DOWN
to 40 — and `policy_violations` has no branch that would catch it. That is
F9-32, and this module is the fix: an `increase` lever that only ever moves
upward, with its own violation check.

Every lever below moves a value the patient can actually change. Age, sex, family
history and prior cardiovascular disease are facts. Creatinine, urea and albumin
are not levers a patient pulls directly, so they are immutable here even though
the model reads them.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple

# Mirrors backend.schemas_diabetes_nhanes.ADIPOSITY_BAND_LEVELS. Copied, not imported:
# this module imports nothing from backend (enforced by a test); a test also pins the copy.
ADIPOSITY_BAND_LEVELS = {"normal": 0, "increased": 1, "high": 2}

_LEVEL_NAMES = {level: name for name, level in ADIPOSITY_BAND_LEVELS.items()}


def _num(value: Any) -> float:
    """A lever value as a number; named adiposity bands map to their ordinal."""
    if isinstance(value, str) and value in ADIPOSITY_BAND_LEVELS:
        return float(ADIPOSITY_BAND_LEVELS[value])
    return float(value)


def _like(original: Any, number: float) -> Any:
    """`number` in the form the patient supplied it (band name for a band)."""
    if isinstance(original, str) and original in ADIPOSITY_BAND_LEVELS:
        return _LEVEL_NAMES[int(number)]
    return number

#: kind -> meaning
#:   "decrease": may only move DOWN, and never below `bound`
#:   "increase": may only move UP, and never above `bound`
#:   "to":       may only take the value `bound`
LeverKind = str
Lever = Tuple[LeverKind, float]

#: Clinical targets, not model-derived ones. Sourced from ordinary metabolic
#: targets, and capped so a scenario never proposes a physiologically absurd value.
DIABETES_LEVERS: Dict[str, Lever] = {
    "BMXBMI": ("decrease", 24.9),        # kg/m^2, upper end of normal
    "ADIPOSITY_BAND": ("decrease", 0),   # clinician-assessed band, down to normal
    "SBP": ("decrease", 120),            # mmHg
    "DBP": ("decrease", 80),             # mmHg
    "LBXSTR": ("decrease", 150),         # triglycerides, mg/dL
    "LBXSCH": ("decrease", 200),         # total cholesterol, mg/dL
    "LBXSGTSI": ("decrease", 40),        # GGT, U/L
    "LBXSATSI": ("decrease", 40),        # ALT, U/L
    "LBXSUA": ("decrease", 6.0),         # uric acid, mg/dL
    "LBDHDD": ("increase", 60),          # HDL, mg/dL — the F9-32 lever
    "PAQ650": ("to", 1),                 # vigorous activity
    "PAQ665": ("to", 1),                 # moderate activity
}

#: Read by the model, deliberately not levers.
IMMUTABLE: Tuple[str, ...] = (
    "RIDAGEYR", "RIAGENDR", "MCQ300C", "CVD_ANY", "BPXPLS",
    "LBXSCR", "LBXSBU", "LBXSAL",
)

HDL_LEVER_NOTE = (
    "Raising HDL here is a proxy for the behaviour that raises it — sustained "
    "aerobic activity, weight loss, stopping smoking. In this model a low HDL is a "
    "marker of insulin resistance rather than a lipid target in its own right, so "
    "the scenario should be read as 'this metabolic pattern improves', not as "
    "'take a drug that raises HDL'. No HDL-raising drug has been shown to reduce "
    "progression to diabetes."
)


def is_engaged(value: Optional[float], kind: LeverKind, bound: float) -> bool:
    """True when this lever has somewhere to move for this patient."""
    if value is None:
        return False
    value = _num(value)
    if kind == "decrease":
        return value > float(bound)
    if kind == "increase":
        return value < float(bound)
    return value != float(bound)


def apply_lever(value: Optional[float], kind: LeverKind, bound: float) -> Optional[float]:
    """The value this lever moves to, or the value unchanged.

    The `increase` branch is the whole reason this module exists: it moves a value
    UP toward the bound and leaves anything already above it alone. The shared
    generator's equivalent would overwrite it.
    """
    if value is None:
        return None
    original = value
    value = _num(value)
    if kind == "decrease":
        moved = float(bound) if value > float(bound) else value
    elif kind == "increase":
        moved = float(bound) if value < float(bound) else value
    else:
        moved = float(bound)
    return _like(original, moved)


def all_improvements(
    patient: Dict[str, Any], levers: Dict[str, Lever] = DIABETES_LEVERS
) -> Dict[str, Any]:
    """The patient with every engaged lever at its target. Missing values stay missing."""
    improved = dict(patient)
    for feature, (kind, bound) in sorted(levers.items()):
        if feature in patient:
            improved[feature] = apply_lever(patient.get(feature), kind, bound)
    return improved


def policy_violations(
    original: Dict[str, Any],
    changes: Dict[str, Any],
    levers: Dict[str, Lever] = DIABETES_LEVERS,
) -> List[str]:
    """Every way `changes` breaks the policy. Empty list means allowed.

    Run on every scenario immediately before it is returned, so a violating
    scenario cannot escape even if candidate generation changes later.
    """
    problems: List[str] = []
    for feature, new in sorted(changes.items()):
        if feature not in levers:
            problems.append(f"{feature}: immutable")
            continue
        old = original.get(feature)
        if old is None or new is None:
            problems.append(f"{feature}: missing value cannot be a lever")
            continue
        kind, bound = levers[feature]
        old_v, new_v = _num(old), _num(new)
        if new_v == old_v:
            continue
        if kind == "decrease" and not (new_v < old_v and new_v >= float(bound)):
            problems.append(f"{feature}: {old_v} -> {new_v} (decrease only, floor {bound})")
        elif kind == "increase" and not (new_v > old_v and new_v <= float(bound)):
            problems.append(f"{feature}: {old_v} -> {new_v} (increase only, ceiling {bound})")
        elif kind == "to" and new_v != float(bound):
            problems.append(f"{feature}: {old_v} -> {new_v} (may only move to {bound})")
    return problems


def simulate_hdl(
    patient: Dict[str, Any],
    score_fn: Callable[[Dict[str, Any]], float],
    targets: Tuple[float, ...] = (45.0, 50.0, 55.0, 60.0),
) -> Dict[str, Any]:
    """Walk HDL up through clinical targets and report what the model does.

    Returns the curve rather than a single number, because the EBM's HDL shape
    function is not a straight line and a single "what if HDL were 60" hides where
    the benefit actually sits. Targets at or below the patient's current HDL are
    skipped — this lever never moves a value down.
    """
    current = patient.get("LBDHDD")
    if current is None:
        return {
            "feature": "LBDHDD",
            "applicable": False,
            "reason": "HDL was not supplied for this patient.",
            "note": HDL_LEVER_NOTE,
        }

    current = float(current)
    baseline = float(score_fn(patient))
    kind, ceiling = DIABETES_LEVERS["LBDHDD"]

    steps = []
    for target in targets:
        if not is_engaged(current, kind, target) or target > float(ceiling):
            continue
        candidate = dict(patient)
        candidate["LBDHDD"] = float(target)
        if policy_violations(patient, {"LBDHDD": float(target)}):
            continue
        probability = float(score_fn(candidate))
        steps.append({
            "hdl_mg_dl": float(target),
            "probability": probability,
            "absolute_change_pp": round((probability - baseline) * 100, 2),
        })

    return {
        "feature": "LBDHDD",
        "applicable": bool(steps),
        "current_hdl_mg_dl": current,
        "baseline_probability": baseline,
        "steps": steps,
        "reason": None if steps else (
            f"HDL is already at or above every target ({current:g} mg/dL); this lever "
            f"only moves upward."
        ),
        "note": HDL_LEVER_NOTE,
    }


def lowest_achievable(
    patient: Dict[str, Any],
    score_fn: Callable[[Dict[str, Any]], float],
    baseline: float,
    levers: Dict[str, Lever] = DIABETES_LEVERS,
) -> Optional[Tuple[Dict[str, Any], float]]:
    """The allowed combination of levers with the lowest estimate, or None.

    "Every lever at once" is not necessarily the lowest — a lever can raise the
    estimate — so the candidates are each lever alone, all of them together, and
    the subset that helped individually. That is n + 2 model calls, not 2**n.
    """
    full = all_improvements(patient, levers)
    engaged = [f for f in sorted(levers) if full.get(f) != patient.get(f)]
    if not engaged:
        return None

    scored: List[Tuple[float, Dict[str, Any]]] = []
    helpful: List[str] = []
    for feature in engaged:
        candidate = dict(patient)
        candidate[feature] = full[feature]
        probability = float(score_fn(candidate))
        scored.append((probability, candidate))
        if probability < baseline:
            helpful.append(feature)

    scored.append((float(score_fn(full)), full))
    if 1 < len(helpful) < len(engaged):
        candidate = dict(patient)
        for feature in helpful:
            candidate[feature] = full[feature]
        scored.append((float(score_fn(candidate)), candidate))

    best_probability, best = min(
        scored,
        key=lambda pair: (
            pair[0],
            sum(pair[1].get(f) != patient.get(f) for f in levers),
        ),
    )
    if not best_probability < baseline:
        return None
    return best, best_probability
