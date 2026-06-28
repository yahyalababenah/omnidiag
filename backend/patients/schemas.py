"""
OmniDiag — Patient Pydantic Schemas
=====================================
Request/response models for /api/v4/patients/* endpoints.
"""

from datetime import date, datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ── Requests ─────────────────────────────────────────────────────────────────

class PatientCreate(BaseModel):
    mrn: str = Field(..., min_length=1, max_length=50, description="Medical Record Number — must be unique")
    full_name: str = Field(..., min_length=2, max_length=255)
    date_of_birth: Optional[date] = Field(None, description="ISO date, e.g. 1970-05-15")
    gender: Optional[str] = Field(None, max_length=10, description="M | F | Other")
    contact_email: Optional[str] = Field(None, max_length=255)


class PatientUpdate(BaseModel):
    full_name: Optional[str] = Field(None, min_length=2, max_length=255)
    date_of_birth: Optional[date] = None
    gender: Optional[str] = Field(None, max_length=10)
    contact_email: Optional[str] = Field(None, max_length=255)


# ── Responses ────────────────────────────────────────────────────────────────

class PatientOut(BaseModel):
    id: str
    mrn: str
    full_name: str
    date_of_birth: Optional[date]
    gender: Optional[str]
    contact_email: Optional[str]
    created_at: datetime
    deleted_at: Optional[datetime]

    model_config = {"from_attributes": True}


class PatientPage(BaseModel):
    total: int
    page: int
    limit: int
    pages: int
    items: List[PatientOut]


class PredictionOut(BaseModel):
    id: str
    disease: str
    prediction: int
    confidence: float
    diagnosis: Optional[str]
    input_features: Dict[str, Any]
    shap_chart_data: Optional[Any]
    created_at: datetime

    model_config = {"from_attributes": True}


class PredictionPage(BaseModel):
    total: int
    page: int
    limit: int
    pages: int
    items: List[PredictionOut]


class PatientExportBundle(BaseModel):
    exported_at: datetime
    patient: PatientOut
    predictions: List[PredictionOut]
    total_predictions: int
