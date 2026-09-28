"""
Input schema for the NHANES dysglycaemia module (Gate 9.2).

Lives in its own module rather than in backend/schemas.py so that adding this
disease edits no file the heart or BRFSS modules read. backend/router.py picks it
up from `schema: {module, class}` in configs/diabetes_nhanes.yaml and registers it.

Two rules this schema enforces that a generic Pydantic model would not:

1. **D9-06 — six fields are mandatory and cannot be null.** The blank-field audit
   (F9-28) measured what happens when each field is left empty on the shipped
   bundle. Leaving `PAQ650` empty shifts median predicted risk by +0.108 and
   changes 31% of decisions; blanking age changes 38.5%. The model reads a blank
   as a *learned value*, not as "unknown", so a silently-empty field is a silently
   different answer. These six are therefore required with no default: Pydantic
   rejects the request before it reaches the model.

2. **Ranges are the training ranges, stated per field with its unit.** NHANES lab
   values exist in both conventional and SI columns; this model was trained on the
   ones named below and the unit is part of the contract. A value outside the
   range is refused rather than clipped, because clipping would silently move a
   patient to a risk the clinician did not enter.

The remaining fourteen fields are optional, and leaving one empty is allowed but
not free: the backend attaches a `data_completeness_warning` naming them.
"""

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class DiabetesNhanesInput(BaseModel):
    """One patient, as a clinician has them at a first visit.

    Twenty inputs: seven from the room (age, sex, BMI, central adiposity, blood
    pressure, pulse), two questions, and eleven values from one lipid panel plus
    one basic metabolic panel. No complete blood count — see D9-03.
    """

    # ── mandatory (D9-06) ────────────────────────────────────────────────
    RIDAGEYR: float = Field(
        ..., ge=20, le=80, description="Age in years. Adults only; the cohort is 20-80.",
    )
    BMXBMI: float = Field(
        ..., ge=10, le=100, description="Body mass index, kg/m^2.",
    )
    ADIPOSITY_BAND: Literal["normal", "increased", "high"] = Field(
        ...,
        description=(
            "Central adiposity, assessed by the clinician by eye — no tape measure. "
            "normal = no visible central fat; increased = noticeable; high = prominent. "
            "Trained against waist-to-height ratio at the published boundaries 0.50 "
            "and 0.60. Declared as named levels rather than 0/1/2 so the form renders "
            "a labelled dropdown instead of an unlabelled slider, and so the value is "
            "self-describing in a stored record. There is no 'unknown' level on "
            "purpose: F9-25 measured that blanking this field LOWERS predicted risk, "
            "which is a false reassurance the clinician did not intend."
        ),
    )
    LBDHDD: float = Field(
        ..., ge=10, le=150, description="HDL cholesterol, mg/dL.",
    )
    PAQ650: int = Field(
        ..., ge=0, le=1,
        description="Vigorous recreational activity in a typical week: 1 = yes, 0 = no.",
    )
    PAQ665: int = Field(
        ..., ge=0, le=1,
        description="Moderate recreational activity in a typical week: 1 = yes, 0 = no.",
    )

    # ── optional ─────────────────────────────────────────────────────────
    RIAGENDR: Optional[int] = Field(
        None, ge=0, le=1, description="Sex: 1 = male, 0 = female.",
    )
    SBP: Optional[float] = Field(
        None, ge=60, le=260, description="Systolic blood pressure, mmHg (mean of readings).",
    )
    DBP: Optional[float] = Field(
        None, ge=30, le=150, description="Diastolic blood pressure, mmHg (mean of readings).",
    )
    BPXPLS: Optional[float] = Field(
        None, ge=30, le=200, description="Resting pulse, beats per minute.",
    )
    MCQ300C: Optional[int] = Field(
        None, ge=0, le=1,
        description="Close blood relative with diabetes: 1 = yes, 0 = no.",
    )
    CVD_ANY: Optional[int] = Field(
        None, ge=0, le=1,
        description=(
            "Any prior cardiovascular disease — heart failure, coronary heart disease, "
            "heart attack or stroke: 1 = yes, 0 = no."
        ),
    )
    LBXSCH: Optional[float] = Field(
        None, ge=50, le=500, description="Total cholesterol, mg/dL.",
    )
    LBXSTR: Optional[float] = Field(
        None, ge=10, le=3000, description="Triglycerides, mg/dL.",
    )
    LBXSATSI: Optional[float] = Field(
        None, ge=1, le=1000, description="ALT (alanine aminotransferase), U/L.",
    )
    LBXSGTSI: Optional[float] = Field(
        None, ge=1, le=1000, description="GGT (gamma-glutamyl transferase), U/L.",
    )
    LBXSCR: Optional[float] = Field(
        None, ge=0.1, le=20, description="Serum creatinine, mg/dL.",
    )
    LBXSBU: Optional[float] = Field(
        None, ge=1, le=200, description="Blood urea nitrogen, mg/dL.",
    )
    LBXSAL: Optional[float] = Field(
        None, ge=1, le=6, description="Serum albumin, g/dL.",
    )
    LBXSUA: Optional[float] = Field(
        None, ge=0.5, le=20, description="Uric acid, mg/dL.",
    )

    model_config = ConfigDict(json_schema_extra={
        # F9-38 / D9-06. The form pre-fills from `example`, so without this the six
        # mandatory fields arrive already carrying the example patient's values and
        # the firewall is defeated: it rejects a NULL, but the form never sends one.
        # A clinician who skips the adiposity question would submit "high" because
        # that is what the example patient had. A blank is refused; a wrong value is
        # scored. Fields listed here are never pre-filled and must be answered.
        # Same mechanism as `x_unused_by_model` (Gate 8.9): a schema-level hint the
        # parser reads, absent from every other module's schema, so nothing else
        # changes behaviour.
        "x_requires_deliberate_entry": [
            "RIDAGEYR", "BMXBMI", "ADIPOSITY_BAND", "LBDHDD", "PAQ650", "PAQ665",
        ],
        "example": {
            "RIDAGEYR": 58, "BMXBMI": 31.2, "ADIPOSITY_BAND": "high",
            "LBDHDD": 41, "PAQ650": 0, "PAQ665": 0,
            "RIAGENDR": 1, "SBP": 138, "DBP": 84, "BPXPLS": 78,
            "MCQ300C": 1, "CVD_ANY": 0, "LBXSCH": 205, "LBXSTR": 190,
            "LBXSATSI": 28, "LBXSGTSI": 34, "LBXSCR": 0.95, "LBXSBU": 15,
            "LBXSAL": 4.2, "LBXSUA": 6.4,
        }
    })


#: The six D9-06 fields, exported so a consumer can mark them in a form without
#: re-deriving the list. It is asserted against the bundle in the test suite.
MANDATORY_FIELDS = [
    "RIDAGEYR", "BMXBMI", "ADIPOSITY_BAND", "LBDHDD", "PAQ650", "PAQ665",
]


#: The clinician-facing levels, and the integer the model reads. Exported so the
#: backend maps them in exactly one place and a test can assert the two agree.
ADIPOSITY_BAND_LEVELS = {"normal": 0, "increased": 1, "high": 2}
