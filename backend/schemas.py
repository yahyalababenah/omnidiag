"""
OmniDiag — Pydantic Schemas
============================
Dynamic Pydantic model generation for disease-specific patient input schemas.
Each disease can define its own input fields, validation rules, and example data.

Currently supports:
    - Heart Disease: 11 clinical features with medical validation ranges.
    
Future diseases will add their own schema definitions here.
"""

from typing import Dict, Type
from pydantic import BaseModel, Field


# =============================================================================
# Heart Disease Schema
# =============================================================================

class HeartDiseaseInput(BaseModel):
    """
    Patient input schema for Heart Disease diagnosis.
    
    All fields correspond to the 11 clinical features used in the
    UCI Heart Disease dataset + Z-Alizadeh Sani harmonized dataset.
    """
    Age: int = Field(..., description="Age in years", ge=20, le=100)
    Sex: int = Field(..., description="Sex (1=Male, 0=Female)", ge=0, le=1)
    ChestPainType: int = Field(..., description="Chest pain type (0=TA, 1=ATA, 2=NAP, 3=ASY)", ge=0, le=3)
    RestingBP: int = Field(..., description="Resting blood pressure (mm Hg)", ge=80, le=220)
    Cholesterol: int = Field(..., description="Serum cholesterol (mg/dl)", ge=100, le=600)
    FastingBS: int = Field(..., description="Fasting blood sugar > 120 mg/dl (1=True, 0=False)", ge=0, le=1)
    RestingECG: int = Field(..., description="Resting ECG results (0=Normal, 1=ST, 2=LVH)", ge=0, le=2)
    MaxHR: int = Field(..., description="Maximum heart rate achieved", ge=60, le=220)
    ExerciseAngina: int = Field(..., description="Exercise-induced angina (1=Yes, 0=No)", ge=0, le=1)
    Oldpeak: float = Field(..., description="ST depression induced by exercise relative to rest")
    ST_Slope: int = Field(..., description="Slope of peak exercise ST segment (0=Up, 1=Flat, 2=Down)", ge=0, le=2)

    class Config:
        json_schema_extra = {
            "example": {
                "Age": 54,
                "Sex": 1,
                "ChestPainType": 0,
                "RestingBP": 140,
                "Cholesterol": 289,
                "FastingBS": 0,
                "RestingECG": 1,
                "MaxHR": 122,
                "ExerciseAngina": 0,
                "Oldpeak": 0.0,
                "ST_Slope": 1
            }
        }


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
        ValueError: If no schema is registered for the disease.
    """
    if disease_name not in DISEASE_SCHEMA_REGISTRY:
        raise ValueError(
            f"No schema registered for disease '{disease_name}'. "
            f"Available: {list(DISEASE_SCHEMA_REGISTRY.keys())}"
        )
    return DISEASE_SCHEMA_REGISTRY[disease_name]
