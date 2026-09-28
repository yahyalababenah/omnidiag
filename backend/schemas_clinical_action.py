"""
Tiered clinical triage for the NHANES dysglycaemia module (Gate 9.3).

Turns the model's three-way conformal decision into the next clinical step, so
that an `uncertain` result leads to a cheap, fast test rather than to either an
expensive workup or a shrug. The point is to stop `uncertain` — 38% of patients
at the shipped alpha — from being read as "moderate risk".

Its own module on purpose. The heart module also returns conformal decisions and
must keep returning exactly what it returns today; nothing here is imported by it,
and no file it reads is touched to add this.

-------------------------------------------------------------------------------
A NUMBER THAT BELONGS NEXT TO THIS CODE
-------------------------------------------------------------------------------
At the shipped alpha = 0.20, measured on the held-out 2017-2018 cycle:

    14.1% of genuinely dysglycaemic patients receive NO_REFER.
    Per age band: 9.8% (20-39), 15.8% (40-59), 14.3% (60+).

So roughly one in seven patients who DO have dysglycaemia is told there is no
indication to test. That is a property of the operating point, not a bug, and it
is why the NO_REFER plan below is worded as "this estimate does not indicate
testing" and carries a safety_note, rather than as "low risk — you are fine".
A clinician reading this field must be able to see that a negative is not a
clearance. At alpha = 0.15 the same figure was 10.1%.
"""

from typing import Dict, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

Urgency = Literal["High", "Medium", "Low"]
Decision = Literal["referral", "uncertain", "no_referral"]


class ClinicalActionPlan(BaseModel):
    """The next clinical step implied by one conformal decision."""

    recommended_test: str = Field(
        ..., description="The test to order next, or 'None' when none is indicated."
    )
    urgency: Urgency = Field(
        ..., description="Triage tier: High, Medium or Low."
    )
    rationale: str = Field(
        ..., description="Why this step follows from the model's decision."
    )
    safety_note: Optional[str] = Field(
        None,
        description=(
            "What this plan does NOT establish. Present on every tier where the "
            "decision can be wrong in a way that matters clinically."
        ),
    )
    decision: Decision = Field(
        ..., description="The conformal decision this plan was derived from."
    )

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "recommended_test": "Fasting Blood Glucose (FBG) or random capillary blood glucose",
            "urgency": "Medium",
            "rationale": (
                "The model could not place this patient confidently in either group. "
                "Order a low-cost, rapid test to break the ambiguity before committing "
                "to confirmatory diagnostics."
            ),
            "safety_note": (
                "UNCERTAIN is not a middle amount of risk. It means the evidence did not "
                "separate the two groups for this patient."
            ),
            "decision": "uncertain",
        }
    })


# Deliberately a module-level constant rather than an if/elif chain inside the
# request path: the three plans are policy, they are reviewed as policy, and a
# reviewer should be able to read all three side by side without following
# control flow.
_PLANS: Dict[Decision, Dict[str, Optional[str]]] = {
    "referral": {
        "recommended_test": "HbA1c (glycated haemoglobin), or OGTT where HbA1c is unreliable",
        "urgency": "High",
        "rationale": (
            "The model places this patient confidently in the dysglycaemic group. "
            "Proceed directly to confirmatory diagnostic testing."
        ),
        "safety_note": (
            "This is a screening estimate, not a diagnosis. Dysglycaemia is established "
            "by the laboratory result, never by this model. HbA1c is itself unreliable in "
            "iron deficiency, haemoglobinopathy and recent transfusion — use OGTT instead "
            "in those patients."
        ),
    },
    "uncertain": {
        "recommended_test": "Fasting Blood Glucose (FBG) or random capillary blood glucose",
        "urgency": "Medium",
        "rationale": (
            "The model could not place this patient confidently in either group. Order a "
            "low-cost, rapid test to break the ambiguity before committing to confirmatory "
            "diagnostics."
        ),
        "safety_note": (
            "UNCERTAIN is not a middle amount of risk, and it is not a mild positive. It "
            "means the evidence did not separate the two groups for this patient. Part of "
            "this rate is irreducible: a quarter of patients sit within the assay's own "
            "measurement error of the 5.7% cut-point."
        ),
    },
    "no_referral": {
        "recommended_test": "None",
        "urgency": "Low",
        "rationale": (
            "This estimate does not indicate testing now. Continue routine lifestyle "
            "advice and re-screen at the usual interval for the patient's age and risk "
            "factors."
        ),
        "safety_note": (
            "A NO_REFER result is not a clearance. At the shipped operating point, 14.1% "
            "of genuinely dysglycaemic patients receive it (9.8% aged 20-39, 15.8% aged "
            "40-59, 14.3% aged 60+). Clinical suspicion, symptoms or a strong family "
            "history override this estimate."
        ),
    },
}


def build_clinical_action_plan(decision: str) -> ClinicalActionPlan:
    """Map one conformal decision to its clinical plan.

    Raises on an unknown decision rather than defaulting. A silent fall-through to
    the Low tier would hand a clinician a "no test indicated" plan derived from a
    decision nobody recognised, which is exactly the failure this module exists to
    prevent.
    """
    plan = _PLANS.get(decision)
    if plan is None:
        raise ValueError(
            f"No clinical action plan for decision {decision!r}. Known decisions: "
            f"{sorted(_PLANS)}. Refusing to fall back to a tier."
        )
    return ClinicalActionPlan(decision=decision, **plan)  # type: ignore[arg-type]
