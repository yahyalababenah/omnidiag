"""
Active learning for the NHANES dysglycaemia module — candidate only, never promoted.

Its own module. `backend/active_learning/retrain.py` is untouched: it hard-codes
`models/{disease}/omni_diag_xgb_optimized.pkl` and an XGBoost refit, neither of
which exists here, and its own docstring already records that the heart cycle
fails on the same line. Nothing the BRFSS or heart paths read is modified.

Two hazards make an ordinary AL loop unsafe for THIS module, and both are enforced
below rather than written down and hoped for.

-------------------------------------------------------------------------------
HAZARD 1 — the review queue's label is an opinion; this model's target is an assay
-------------------------------------------------------------------------------
`ReviewQueue.label` is documented as "expert annotation (0 or 1)". For the heart
module a clinician can meaningfully annotate "should this patient have been
referred". Here the target is `HbA1c >= 5.7%`, a laboratory value that nobody can
eyeball — not from waist, not from lipids, not from a family history.

An annotation that is a clinician's guess teaches the model to imitate clinicians.
An annotation that is a transcribed HbA1c result teaches it the target. These are
different training sets and only one of them is the model's job.

So a row is admitted only when its note carries lab provenance. The check is
deliberately crude and deliberately loud: it is better to refuse a real result that
was badly documented than to silently learn from a guess.

-------------------------------------------------------------------------------
HAZARD 2 — partial verification bias, measured on this cohort
-------------------------------------------------------------------------------
Only patients who are referred or uncertain plausibly get an HbA1c ordered.
Cleared patients are never verified, so they never enter the annotated set. On the
training cycle that means the AL loop would see:

    referral      n=5181  (28.0%)  true prevalence 0.523
    uncertain     n=6908  (37.3%)  true prevalence 0.324
    no_referral   n=6419  (34.7%)  true prevalence 0.154   <- never verified

    annotated set prevalence 0.409  against a true cohort prevalence of 0.321
    an enrichment of +8.8 points, and it discards 990 genuinely dysglycaemic
    patients — 16.7% of all positives — who are precisely the ones the model
    already gets wrong and most needs to learn from.

The enrichment is also NOT uniform by age band (+6.0 / +10.9 / +5.5 points), so it
would distort the group-conditional conformal layer as well as the prevalence.

Refitting on that set without correction moves the model toward over-referral and
calls it learning. `verification_bias` below measures it on the actual annotations
and `build_candidate` refuses past a stated tolerance.

-------------------------------------------------------------------------------
WHAT THIS MODULE WILL NOT DO
-------------------------------------------------------------------------------
It never writes to `model.weights_path`. It never hot-reloads a loader. It writes a
candidate bundle to its own directory with a card recording every refusal it could
have made and did not, and promotion is a human act performed elsewhere.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

log = logging.getLogger("omnidiag.active_learning.diabetes_nhanes")

DECISION_REFERRAL = "referral"
DECISION_NO_REFERRAL = "no_referral"
DECISION_UNCERTAIN = "uncertain"

#: Maximum tolerated prevalence enrichment in the annotated set, in absolute
#: percentage points, before a candidate is refused. Set at 5 points because the
#: measured enrichment of an uncorrected loop on this cohort is 8.8 — the tolerance
#: has to sit below the failure it exists to catch, or it is decoration.
MAX_PREVALENCE_ENRICHMENT = 0.05

#: The annotated set must contain at least this share of cleared patients. Zero
#: cleared patients means the loop has no information about the decision it is most
#: likely to be getting wrong.
MIN_CLEARED_SHARE = 0.10

#: Evidence that a label came from an assay rather than from an opinion.
_LAB_EVIDENCE = re.compile(
    r"\b(hba1c|a1c|glycated|glycosylated|hb\s*a1c|ogtt|fpg|fasting\s+glucose)\b"
    r".{0,40}?(\d+(?:\.\d+)?)",
    re.IGNORECASE | re.DOTALL,
)


class CandidateRefused(RuntimeError):
    """A candidate was not built, and the message says exactly why.

    Raised rather than returning None so that a caller cannot mistake a refusal
    for "nothing to do". Every refusal here is a finding about the data, not an
    error in the code.
    """


@dataclass
class Annotation:
    """One reviewed prediction, with the provenance of its label."""

    features: Dict[str, Any]
    label: int
    decision_at_prediction: Optional[str]
    note: Optional[str] = None
    lab_value: Optional[float] = field(default=None)

    @property
    def lab_confirmed(self) -> bool:
        return self.lab_value is not None


def extract_lab_value(note: Optional[str]) -> Optional[float]:
    """Pull an HbA1c-like number out of a reviewer's note, or return None.

    Deliberately conservative. A note that merely says "confirmed" is not evidence;
    a note that says "HbA1c 6.1" is. Returning None is a refusal, not a default.
    """
    if not note:
        return None
    # Every lab mention is considered, not just the first. "OGTT 2h, HbA1c 6.4"
    # names two tests, and stopping at the first one reads the "2" from "2h" and
    # then discards a perfectly good result.
    for match in _LAB_EVIDENCE.finditer(note):
        try:
            value = float(match.group(2))
        except (TypeError, ValueError):
            continue
        # An HbA1c percentage lives in roughly 3-20. A number outside that is a
        # glucose value, a date, a patient id or a typo, and guessing which is
        # exactly the kind of silent coercion this module exists to avoid.
        if 3.0 <= value <= 20.0:
            return value
    return None


def lab_confirmed_only(annotations: Sequence[Annotation]) -> Tuple[List[Annotation], List[Annotation]]:
    """Split annotations into those with assay provenance and those without."""
    kept = [a for a in annotations if a.lab_confirmed]
    dropped = [a for a in annotations if not a.lab_confirmed]
    return kept, dropped


def verification_bias(
    annotations: Sequence[Annotation], cohort_prevalence: float
) -> Dict[str, Any]:
    """Measure how unrepresentative the annotated set is.

    Reports the decision mix, the prevalence of the annotated set against the
    cohort it claims to represent, and whether cleared patients are present at all.
    The caller decides what to do; this function only measures.
    """
    total = len(annotations)
    if total == 0:
        return {
            "n": 0, "prevalence": None, "cohort_prevalence": cohort_prevalence,
            "enrichment": None, "decision_mix": {}, "cleared_share": 0.0,
            "representative": False,
            "reason": "no annotations",
        }

    labels = np.array([a.label for a in annotations], dtype=float)
    mix: Dict[str, int] = {}
    for a in annotations:
        key = a.decision_at_prediction or "unknown"
        mix[key] = mix.get(key, 0) + 1

    prevalence = float(labels.mean())
    enrichment = prevalence - cohort_prevalence
    cleared_share = mix.get(DECISION_NO_REFERRAL, 0) / total

    reasons = []
    if abs(enrichment) > MAX_PREVALENCE_ENRICHMENT:
        reasons.append(
            f"annotated prevalence {prevalence:.3f} against cohort {cohort_prevalence:.3f} "
            f"({enrichment:+.3f}, tolerance {MAX_PREVALENCE_ENRICHMENT:.3f})"
        )
    if cleared_share < MIN_CLEARED_SHARE:
        reasons.append(
            f"only {cleared_share:.1%} of annotations are patients the model CLEARED "
            f"(minimum {MIN_CLEARED_SHARE:.0%}); a loop that never sees its own clearances "
            f"cannot learn from the decision it is most likely getting wrong"
        )

    return {
        "n": total,
        "prevalence": prevalence,
        "cohort_prevalence": cohort_prevalence,
        "enrichment": enrichment,
        "decision_mix": mix,
        "cleared_share": cleared_share,
        "representative": not reasons,
        "reason": "; ".join(reasons) if reasons else None,
    }


def transition_matrix(before: Sequence[str], after: Sequence[str]) -> Dict[str, int]:
    """Every decision movement, all nine cells always present.

    A candidate that changes no decision is not an improvement, and a candidate
    that flips referral straight to no_referral deserves its own cell rather than
    being averaged into a summary statistic.
    """
    if len(before) != len(after):
        raise ValueError(
            f"transition matrix needs matching lengths, got {len(before)} and {len(after)}"
        )
    states = (DECISION_REFERRAL, DECISION_UNCERTAIN, DECISION_NO_REFERRAL)
    cells = {f"{a}->{b}": 0 for a in states for b in states}
    for a, b in zip(before, after):
        key = f"{a}->{b}"
        if key not in cells:
            raise ValueError(f"unknown decision transition {key!r}")
        cells[key] += 1
    return cells


def should_queue_for_review(decision: str) -> bool:
    """Which predictions go to the review queue for THIS module.

    Not entropy around a threshold: `backend/active_learning/sampler.py` centres
    uncertainty on a decision threshold, and this module configures none
    (`inference_threshold: null`, D-32). Inventing one here to compute entropy
    would reintroduce exactly the number the module exists without.

    The conformal layer already states which patients it could not separate. That
    IS the uncertainty signal, it carries the per-group coverage property, and it
    needs no cut-point.
    """
    return decision == DECISION_UNCERTAIN


def build_candidate(
    annotations: Sequence[Annotation],
    *,
    cohort_prevalence: float,
    min_samples: int,
    output_dir: str,
    bundle_sha256: str,
    refit: Any = None,
) -> Dict[str, Any]:
    """Produce a candidate model card, refusing loudly rather than degrading quietly.

    `refit` is injected so that the guards are testable without fitting anything and
    so that this module never imports the training stack at request time. When it is
    None the guards run and the card records that no model was fitted — which is the
    correct outcome for every refusal path.

    NEVER writes to the live weights path. NEVER hot-reloads a loader. The returned
    card is evidence for a human deciding whether to promote; promotion is not
    performed here and is not performed automatically anywhere.
    """
    kept, dropped = lab_confirmed_only(annotations)

    if dropped:
        log.warning(
            "%d of %d annotations have no laboratory provenance and were dropped. "
            "This module's target is HbA1c >= 5.7%%, which no reviewer can judge by "
            "eye; an unconfirmed label teaches the model to imitate clinicians "
            "instead of predicting the assay.",
            len(dropped), len(annotations),
        )

    if len(kept) < min_samples:
        raise CandidateRefused(
            f"only {len(kept)} lab-confirmed annotations, need {min_samples}. "
            f"{len(dropped)} more were rejected for having no laboratory value in "
            f"the reviewer's note."
        )

    bias = verification_bias(kept, cohort_prevalence)
    if not bias["representative"]:
        raise CandidateRefused(
            "the annotated set is not representative of the population the model "
            f"serves: {bias['reason']}. Refitting on it would move the model toward "
            "over-referral and record that as learning. Collect outcomes for cleared "
            "patients, or apply an explicit reweighting, before retraining."
        )

    card: Dict[str, Any] = {
        "module": "diabetes_nhanes",
        "kind": "candidate",
        "promoted": False,
        "promotion_policy": (
            "This artifact is never promoted automatically. Promotion is a human act "
            "performed outside this module, against this card."
        ),
        "parent_bundle_sha256": bundle_sha256,
        "annotations_supplied": len(annotations),
        "annotations_used": len(kept),
        "annotations_dropped_no_lab_value": len(dropped),
        "verification_bias": bias,
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    if refit is None:
        card["model_fitted"] = False
        card["note"] = "guards passed; no refit function was supplied, so no model was fitted."
        return card

    result = refit(kept)
    card["model_fitted"] = True
    card.update(result or {})

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "candidate_card.json")
    with open(path, "w") as handle:
        json.dump(card, handle, indent=1, default=float)
    card["card_path"] = path
    card["card_sha256"] = hashlib.sha256(
        json.dumps(card, sort_keys=True, default=float).encode()
    ).hexdigest()
    return card
