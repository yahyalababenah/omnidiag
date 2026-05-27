"""
OmniDiag — Pydantic Schemas
============================
Dynamic Pydantic model generation for disease-specific patient input schemas.
Each disease can define its own input fields, validation rules, and example data.

Currently supports:
    - Heart Disease: 11 clinical features with medical validation ranges.
    
Future diseases will add their own schema definitions here.
"""

from typing import Dict, Type, List
from fastapi import HTTPException
from pydantic import BaseModel, Field


# =============================================================================
# Heart Disease Schema
# =============================================================================

class HeartDiseaseInput(BaseModel):
    """
    Patient input schema for Heart Disease diagnosis.
    
    All fields correspond to the 11 clinical features used in the
    UCI Heart Disease dataset + Z-Alizadeh Sani harmonized dataset.
    
    Categorical fields (Sex, ChestPainType, RestingECG, ExerciseAngina,
    ST_Slope) accept both raw strings (e.g., 'M', 'ATA', 'Normal') and
    pre-encoded integers. The ModelLoader applies label encoding internally.
    """
    Age: int = Field(..., description="Age in years", ge=20, le=100)
    Sex: str = Field(..., description="Sex: 'M' or 'F' (or encoded 0/1)")
    ChestPainType: str = Field(..., description="Chest pain type: 'TA', 'ATA', 'NAP', or 'ASY' (or encoded 0-3)")
    RestingBP: int = Field(..., description="Resting blood pressure (mm Hg)", ge=80, le=220)
    Cholesterol: int = Field(..., description="Serum cholesterol (mg/dl)", ge=100, le=600)
    FastingBS: int = Field(..., description="Fasting blood sugar > 120 mg/dl (1=True, 0=False)", ge=0, le=1)
    RestingECG: str = Field(..., description="Resting ECG: 'Normal', 'ST', or 'LVH' (or encoded 0-2)")
    MaxHR: int = Field(..., description="Maximum heart rate achieved", ge=60, le=220)
    ExerciseAngina: str = Field(..., description="Exercise-induced angina: 'Y' or 'N' (or encoded 0/1)")
    Oldpeak: float = Field(..., description="ST depression induced by exercise relative to rest")
    ST_Slope: str = Field(..., description="ST slope: 'Up', 'Flat', or 'Down' (or encoded 0-2)")

    class Config:
        json_schema_extra = {
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
                "ST_Slope": "Flat"
            }
        }


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


class ExplainResponse(BaseModel):
    """
    Response model for the SHAP explanation endpoint.
    
    Attributes:
        chart_data: List of feature impacts sorted by absolute SHAP value
                    descending, ready for direct consumption by the React
                    ShapBarChart component.
        text_explanation: Human-readable summary of the top 3 most impactful
                          features with direction labels.
        base_value: The base (expected) value from the SHAP explainer, useful
                    for advanced charting.
    """
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
        description="Base (expected) value from the SHAP explainer"
    )


# =============================================================================
# Schema Registry
# =============================================================================
# Maps disease names (from config) to their Pydantic input schemas.
# When a new disease is added, register its schema here.

DISEASE_SCHEMA_REGISTRY: Dict[str, Type[BaseModel]] = {
    "heart_disease": HeartDiseaseInput,
}


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
