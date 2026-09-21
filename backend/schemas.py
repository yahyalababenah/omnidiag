"""
OmniDiag — Pydantic Schemas
============================
Dynamic Pydantic model generation for disease-specific patient input schemas.
Each disease can define its own input fields, validation rules, and example data.

Currently supports:
    - Heart Disease (CAD): 12 clinical features → 16 total after auto-engineering.
      Auto-computed features: RPP (RestingBP × MaxHR), Exercise_Risk_Index (Oldpeak × Angina),
      Age_BP_Interaction, HR_Age_Ratio, Chol_Age_Ratio.
    
    Future diseases will add their own schema definitions here.
"""

from typing import Dict, Type, List, Optional, Literal
from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field


# =============================================================================
# Heart Disease Schema
# =============================================================================

class HeartDiseaseInput(BaseModel):
    """
    Patient input schema for Coronary Artery Disease (CAD) risk assessment.
    
    All fields correspond to the 12 harmonized clinical features from the
    UCI Cleveland + Z-Alizadeh Sani merged dataset.
    
    **Auto-computed features (no input needed):**
    When a prediction is made, the system automatically computes:
        - RPP (Rate-Pressure Product): RestingBP × MaxHR
        - Exercise_Risk_Index: Oldpeak × ExerciseAngina (encoded)
        - Age_BP_Interaction: Age × RestingBP
        - HR_Age_Ratio: MaxHR / Age
        - Chol_Age_Ratio: Cholesterol / Age
    
    These are calculated server-side; you only need to provide the 12 base fields.
    
    Categorical fields (Sex, ChestPainType, RestingECG, ExerciseAngina,
    ST_Slope) accept both raw strings (e.g., 'M', 'ATA', 'Normal') and
    pre-encoded integers. The ModelLoader applies label encoding internally.
    """
    # Required: no legitimate way to triage a patient without these, and
    # ChestPainType/ExerciseAngina rank #1/#3 in the shipped model's SHAP
    # importance (evaluation_evidence/heart/shap_importance.json) -- missing
    # values for those are rejected outright, not silently imputed.
    Age: int = Field(..., description="Age in years", ge=20, le=100)
    Sex: Literal['M', 'F'] = Field(..., description="Sex: 'M' or 'F' (or encoded 0/1)")
    ChestPainType: Literal['TA', 'ATA', 'NAP', 'ASY'] = Field(..., description="Chest pain type: 'TA', 'ATA', 'NAP', or 'ASY' (or encoded 0-3)")
    RestingECG: Literal['Normal', 'ST', 'LVH'] = Field(..., description="Resting ECG: 'Normal', 'ST', or 'LVH' (or encoded 0-2)")
    ExerciseAngina: Literal['Y', 'N'] = Field(..., description="Exercise-induced angina: 'Y' or 'N' (or encoded 0/1)")

    # Optional: the shipped Pipeline's ColumnTransformer imputes every one of
    # these (IterativeImputer for the numeric four, SimpleImputer
    # most-frequent for ST_Slope) -- it was built to accept incomplete raw
    # UCI-site data (e.g. Hungarian rows are commonly missing several of
    # these), so pydantic requiring them outright rejected real, usable
    # patient records before they ever reached the model. See
    # WEAKNESS_REGISTER.md HM-5. A missing value among these that also ranks
    # high in SHAP importance (Oldpeak, Cholesterol) surfaces as
    # data_completeness_warning in the prediction response instead of being
    # silently imputed — see ModelLoader._completeness_warning().
    RestingBP: Optional[int] = Field(None, description="Resting blood pressure (mm Hg)", ge=80, le=220)
    Cholesterol: Optional[int] = Field(None, description="Serum cholesterol (mg/dl)", ge=100, le=600)
    FastingBS: Optional[int] = Field(None, description="Fasting blood sugar > 120 mg/dl (1=True, 0=False)", ge=0, le=1)
    MaxHR: Optional[int] = Field(None, description="Maximum heart rate achieved", ge=60, le=220)
    # Bounds verified against the real training data (data/heart_disease/processed/
    # uci_heart_by_site.csv, 920 rows, 4 UCI sites): observed range is -2.6 to 6.2.
    # ge/le with a margin above/below that range also rejects inf and nan outright
    # (any comparison against nan is False in Python, so nan fails both bounds).
    Oldpeak: Optional[float] = Field(None, description="ST depression induced by exercise relative to rest", ge=-3.0, le=10.0)
    ST_Slope: Optional[Literal['Up', 'Flat', 'Down']] = Field(None, description="ST slope: 'Up', 'Flat', or 'Down' (or encoded 0-2)")

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "Age": 54,
            "Sex": "M",
            "ChestPainType": "ATA",
            "RestingBP": 140,
            "Cholesterol": 289,
            "FastingBS": 0,
            "RestingECG": "Normal",
            "MaxHR": 122,
            "ExerciseAngina": "N",
            "Oldpeak": 0.0,
            "ST_Slope": "Flat",
        }
    })


# =============================================================================
# SHAP Explanation Response Models
# =============================================================================

class FeatureImpact(BaseModel):
    """
    A single feature's SHAP value contribution.
    
    Attributes:
        feature: The name of the feature (e.g., "Age", "ST_Slope").
        shap_value: The SHAP value for this feature. Positive values indicate
                    increased risk; negative values indicate decreased risk.
    """
    feature: str = Field(..., description="Feature name")
    shap_value: float = Field(..., description="SHAP value (positive = increased risk, negative = decreased risk)")


class Counterfactual(BaseModel):
    """
    A single 'what-if' scenario showing actionable changes to reduce risk.
    
    Generated by the DiCE-inspired CounterfactualGenerator. Each counterfactual
    proposes a set of feature changes that would flip a Positive prediction to
    Negative (high risk → low risk).
    
    Attributes:
        scenario: Human-readable description of the proposed changes.
        changes: Dict of feature_name → new_value for changed features only.
        new_probability: Predicted probability of positive class after changes.
        risk_reduction: Percentage reduction text (e.g., "44%").
        feasibility: Clinical feasibility level:
                     - "high": Lifestyle changes only (BMI, diet, exercise)
                     - "medium": Requires medical intervention (medication)
                     - "low": Multiple medical changes or unrealistic BMI drop
    """
    scenario: str = Field(
        ...,
        description="Human-readable description of the proposed changes"
    )
    changes: Dict[str, float] = Field(
        ...,
        description="Modified feature values (only changed features)"
    )
    new_probability: float = Field(
        ...,
        description=(
            "Predicted probability of the positive class after the changes, on "
            "the CORRECTED scale. Kept under its original name; "
            "`new_probability_corrected` is the same value, explicitly named."
        ),
        ge=0.0,
        le=1.0
    )
    risk_reduction: str = Field(
        ...,
        description="RELATIVE reduction in risk as a string (e.g., '74%')"
    )
    risk_reduction_relative_pct: Optional[float] = Field(
        None,
        description="(baseline - after) / baseline * 100 — share of risk removed"
    )
    risk_reduction_absolute_pp: Optional[float] = Field(
        None,
        description=(
            "(baseline - after) * 100 — percentage POINTS removed. Much smaller "
            "than the relative figure on a corrected scale: 0.4795 -> 0.0395 is "
            "92% relative but 44.0 points."
        )
    )
    new_probability_corrected: Optional[float] = Field(
        None,
        description="Same value as new_probability, named for its scale",
        ge=0.0,
        le=1.0
    )
    baseline_probability_corrected: Optional[float] = Field(
        None,
        description="Probability before the changes, corrected scale",
        ge=0.0,
        le=1.0
    )
    probability_scale: Optional[str] = Field(
        None,
        description="'corrected' for every probability in this object"
    )
    feasibility: Literal["high", "medium", "low"] = Field(
        ...,
        description="Clinical feasibility: high=lifestyle, medium=medical, low=unrealistic"
    )


class PredictResponse(BaseModel):
    """
    Response model for POST /api/v4/{disease}/predict.

    Two shapes share this model:

    * heart_disease returns only prediction / confidence / diagnosis, on the
      model's own scale with an argmax 0.5 cut-point.
    * diabetes additionally returns the prevalence-correction audit fields.
      Its `confidence` is on the DEPLOYMENT prior and must be compared with
      `inference_threshold` (same scale), never with 0.5.

    The route uses response_model_exclude_unset=True, so a field a module
    does not send is absent from the JSON rather than present as null —
    heart's response stays exactly the three keys it has always had.

    Naming contract (backend/probability_scale.py): an unsuffixed probability
    or threshold is on the scale the module reports (corrected for diabetes);
    a `_raw` suffix means the model's own training-prior scale.
    """

    model_config = ConfigDict(extra="allow")

    prediction: int = Field(..., description="1 = Positive, 0 = Negative")
    confidence: float = Field(
        ...,
        description=(
            "Probability of the positive class on the reported scale — "
            "prevalence-corrected for diabetes. Same value as probability_corrected."
        ),
        ge=0.0,
        le=1.0,
    )
    diagnosis: str = Field(..., description="'Positive' or 'Negative'")

    probability_raw: Optional[float] = Field(
        None, description="Ensemble output on the training prior (50/50 resample)", ge=0.0, le=1.0
    )
    probability_corrected: Optional[float] = Field(
        None, description="Bayes prior-shift corrected probability; equals confidence", ge=0.0, le=1.0
    )
    prevalence_correction_applied: Optional[bool] = Field(
        None, description="True whenever the two fields above differ in meaning"
    )
    prevalence_train: Optional[float] = Field(None, description="Class prior of the training sample")
    prevalence_deploy: Optional[float] = Field(None, description="Class prior of the deployment population")
    inference_threshold: Optional[float] = Field(
        None, description="Decision threshold on the same scale as confidence"
    )
    inference_threshold_raw: Optional[float] = Field(
        None, description="The same threshold on the training-prior scale, for auditing"
    )
    risk_bands: Optional[Dict[str, float]] = Field(
        None, description="Display bands (high / moderate) on the same scale as confidence"
    )
    risk_bands_raw: Optional[Dict[str, float]] = Field(
        None, description="The same bands on the training-prior scale"
    )
    model_contributions: Optional[Dict[str, float]] = Field(
        None, description="Per-base-model probability, training-prior scale (uncorrected)"
    )
    ensemble_type: Optional[str] = Field(None, description="'stacking' or 'voting'")
    ensemble_variance: Optional[float] = Field(
        None, description="Std of base-model probabilities (training-prior scale)", ge=0.0
    )
    model_agreement: Optional[str] = Field(None, description="'high' | 'moderate' | 'low'")
    data_completeness_warning: Optional[str] = Field(
        None,
        description=(
            "Present only when a high-SHAP-importance input feature was missing and the "
            "model imputed it automatically instead of using the patient's actual value. "
            "Names the missing feature(s); the prediction should be treated with extra "
            "caution. See WEAKNESS_REGISTER.md HM-5."
        ),
    )


class ExplainResponse(BaseModel):
    """
    Response model for the SHAP explanation endpoint.

    Scale note: for a module with a prevalence correction (diabetes) both
    `confidence` and `base_value` are on the DEPLOYMENT scale. The per-feature
    SHAP values are unchanged by the correction — it is a constant additive
    term in log-odds, absorbed entirely into `base_value`.
    """
    prediction: Optional[int] = Field(
        None,
        description="Binary prediction (1 = Positive / disease, 0 = Negative)"
    )
    confidence: Optional[float] = Field(
        None,
        description=(
            "Probability of the positive class, on the scale this module "
            "reports — prevalence-corrected for diabetes, the model's own "
            "scale for heart. Compare it against inference_threshold, never "
            "against 0.5."
        ),
        ge=0.0,
        le=1.0
    )
    diagnosis: Optional[str] = Field(
        None,
        description="Human-readable diagnosis label ('Positive' or 'Negative')"
    )
    chart_data: List[FeatureImpact] = Field(
        ...,
        description="Sorted list of feature SHAP impacts for chart rendering"
    )
    text_explanation: str = Field(
        ...,
        description="Human-readable summary of top 3 impactful features"
    )
    base_value: float = Field(
        ...,
        description=(
            "Base (expected) value from the SHAP explainer, in log-odds on the "
            "same scale as `confidence`"
        )
    )
    base_value_raw: Optional[float] = Field(
        None,
        description=(
            "`base_value` before the prior shift, i.e. on the model's own "
            "training prior. Absent for modules with no correction."
        )
    )
    shap_scale: Optional[str] = Field(
        None,
        description="Scale of base_value and the SHAP sum, e.g. 'corrected_log_odds'"
    )
    shap_reconstructed_probability_corrected: Optional[float] = Field(
        None,
        description=(
            "sigmoid(base_value + sum(shap_values)) — the probability the "
            "explanation alone implies"
        ),
        ge=0.0,
        le=1.0
    )
    shap_additivity_gap: Optional[float] = Field(
        None,
        description=(
            "|reconstruction - confidence|. Non-zero by construction for a "
            "stacking ensemble: the SHAP values are averaged over the base "
            "models while the probability comes from the meta-learner above "
            "them. Reported rather than hidden."
        ),
        ge=0.0
    )
    ensemble_variance: Optional[float] = Field(
        None,
        description="Standard deviation of base model probabilities (model disagreement)",
        ge=0.0
    )
    model_agreement: Optional[str] = Field(
        None,
        description="Qualitative model agreement: 'high', 'moderate', or 'low'"
    )
    counterfactuals: Optional[List[Counterfactual]] = Field(
        None,
        description="Diverse counterfactual scenarios for clinical actionability"
    )


# =============================================================================
# Diabetes Schema
# =============================================================================

class DiabetesInput(BaseModel):
    """
    Patient input schema for Diabetes Risk Assessment.

    All 21 fields correspond to the CDC BRFSS 2015 Health Indicators dataset.
    Binary fields (0/1) and continuous/ordinal fields are validated with
    appropriate ranges.

    **Auto-computed features (no input needed):**
    When a prediction is made, the system automatically computes:
        - BMI_Age_Interaction, Health_Index, Lifestyle_Score, SES_Composite
        - Diabetes_Clinical_Risk

    These are calculated server-side; you only need to provide the 21 base fields.
    """
    # Binary (0/1) risk factors
    HighBP: int = Field(..., description="High blood pressure (1=Yes, 0=No)", ge=0, le=1)
    HighChol: int = Field(..., description="High cholesterol (1=Yes, 0=No)", ge=0, le=1)
    CholCheck: int = Field(..., description="Cholesterol check in past 5 years (1=Yes, 0=No)", ge=0, le=1)
    Smoker: int = Field(..., description="Smoked at least 100 cigarettes in life (1=Yes, 0=No)", ge=0, le=1)
    Stroke: int = Field(..., description="Ever told you had a stroke (1=Yes, 0=No)", ge=0, le=1)
    HeartDiseaseorAttack: int = Field(..., description="Coronary heart disease or MI (1=Yes, 0=No)", ge=0, le=1)
    PhysActivity: int = Field(..., description="Physical activity in past 30 days (1=Yes, 0=No)", ge=0, le=1)
    Fruits: int = Field(..., description="Consume fruit 1+ times per day (1=Yes, 0=No)", ge=0, le=1)
    Veggies: int = Field(..., description="Consume vegetables 1+ times per day (1=Yes, 0=No)", ge=0, le=1)
    HvyAlcoholConsump: int = Field(..., description="Heavy drinkers: adult men >14, women >7 per week (1=Yes, 0=No)", ge=0, le=1)
    AnyHealthcare: int = Field(..., description="Have any form of health insurance (1=Yes, 0=No)", ge=0, le=1)
    NoDocbcCost: int = Field(..., description="Could not see doctor due to cost (1=Yes, 0=No)", ge=0, le=1)
    DiffWalk: int = Field(..., description="Serious difficulty walking/climbing stairs (1=Yes, 0=No)", ge=0, le=1)
    Sex: int = Field(..., description="Sex: 1=Male, 0=Female", ge=0, le=1)

    # Continuous / ordinal health indicators
    BMI: float = Field(..., description="Body Mass Index", ge=10.0, le=100.0)
    MentHlth: int = Field(..., description="Days of poor mental health in past 30", ge=0, le=30)
    PhysHlth: int = Field(..., description="Days of poor physical health in past 30", ge=0, le=30)
    GenHlth: int = Field(..., description="General health: 1=Excellent, 2=Very Good, 3=Good, 4=Fair, 5=Poor", ge=1, le=5)
    Age: int = Field(..., description="Age category: 1=18-24 ... 13=80+ (BRFSS coding)", ge=1, le=13)
    Education: int = Field(..., description="Education level: 1=None to 6=College graduate", ge=1, le=6)
    Income: int = Field(..., description="Income scale: 1=<$10K to 8=>$75K", ge=1, le=8)

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "HighBP": 1,
            "HighChol": 1,
            "CholCheck": 1,
            "BMI": 30.0,
            "Smoker": 1,
            "Stroke": 0,
            "HeartDiseaseorAttack": 0,
            "PhysActivity": 0,
            "Fruits": 0,
            "Veggies": 0,
            "HvyAlcoholConsump": 0,
            "AnyHealthcare": 1,
            "NoDocbcCost": 0,
            "GenHlth": 3,
            "MentHlth": 10,
            "PhysHlth": 5,
            "DiffWalk": 0,
            "Sex": 1,
            "Age": 7,
            "Education": 4,
            "Income": 4,
        }
    })


# =============================================================================
# Schema Registry
# =============================================================================
# Maps disease names (from config) to their Pydantic input schemas.
# A new disease can instead declare `schema: {module, class}` in its YAML;
# backend/router.py then calls register_schema() for it.

DISEASE_SCHEMA_REGISTRY: Dict[str, Type[BaseModel]] = {
    "heart_disease": HeartDiseaseInput,
    "diabetes": DiabetesInput,
}


def register_schema(disease_name: str, schema: Type[BaseModel]) -> None:
    """Register (or replace) the input schema for `disease_name`."""
    if not (isinstance(schema, type) and issubclass(schema, BaseModel)):
        raise TypeError(f"Schema for '{disease_name}' must be a pydantic BaseModel, got {schema!r}")
    DISEASE_SCHEMA_REGISTRY[disease_name] = schema


def get_schema_for_disease(disease_name: str) -> Type[BaseModel]:
    """
    Get the Pydantic input schema for a given disease.
    
    Args:
        disease_name: The disease name as defined in its YAML config.
    
    Returns:
        The Pydantic BaseModel class for that disease.
    
    Raises:
        HTTPException 404: If no schema is registered for the disease.
    """
    if disease_name not in DISEASE_SCHEMA_REGISTRY:
        raise HTTPException(
            status_code=404,
            detail=f"No schema registered for disease '{disease_name}'. "
                   f"Available: {list(DISEASE_SCHEMA_REGISTRY.keys())}"
        )
    return DISEASE_SCHEMA_REGISTRY[disease_name]
