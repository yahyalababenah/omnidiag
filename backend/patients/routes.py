"""
OmniDiag — Patient Records Endpoints
======================================
Mounted at /api/v4/patients in main.py.
All endpoints require at minimum the doctor/nurse/super_admin role.
Soft-delete ensures GDPR compliance — no physical row deletion.

Endpoints:
    POST   /api/v4/patients                        — Create patient record
    GET    /api/v4/patients                        — Paginated list + search
    GET    /api/v4/patients/{patient_id}           — Single patient
    DELETE /api/v4/patients/{patient_id}           — Soft-delete
    GET    /api/v4/patients/{patient_id}/predictions — Prediction history
    GET    /api/v4/patients/{patient_id}/export    — Full data export bundle
"""

import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import and_, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.auth.rbac import CLINICAL_ROLES, ADMIN_ROLES, require_role
from backend.database import get_db
from backend.db_models.patient import Patient
from backend.db_models.prediction import Prediction
from backend.db_models.user import User
from backend.patients.schemas import (
    PatientCreate,
    PatientExportBundle,
    PatientOut,
    PatientPage,
    PatientUpdate,
    PredictionOut,
    PredictionPage,
)

router = APIRouter()

# ── Helpers ───────────────────────────────────────────────────────────────────

async def _get_active_patient(patient_id: str, db: AsyncSession) -> Patient:
    """Fetch a non-deleted patient or raise 404."""
    result = await db.execute(
        select(Patient).where(
            Patient.id == patient_id,
            Patient.deleted_at.is_(None),
        )
    )
    patient = result.scalar_one_or_none()
    if patient is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Patient '{patient_id}' not found",
        )
    return patient


# ── POST /api/v4/patients ─────────────────────────────────────────────────────

@router.post(
    "/",
    response_model=PatientOut,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new patient record",
)
async def create_patient(
    payload: PatientCreate,
    current_user: User = Depends(require_role(*CLINICAL_ROLES)),
    db: AsyncSession = Depends(get_db),
) -> PatientOut:
    """
    Register a new patient in the system.

    - MRN (Medical Record Number) must be unique across all active patients.
    - The creating user's ID is stored in `created_by` for audit purposes.
    """
    # Enforce MRN uniqueness among non-deleted patients
    existing = await db.execute(
        select(Patient).where(
            Patient.mrn == payload.mrn,
            Patient.deleted_at.is_(None),
        )
    )
    if existing.scalar_one_or_none() is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"A patient with MRN '{payload.mrn}' already exists",
        )

    patient = Patient(
        id=str(uuid.uuid4()),
        mrn=payload.mrn,
        full_name=payload.full_name,
        date_of_birth=payload.date_of_birth,
        gender=payload.gender,
        contact_email=payload.contact_email,
        created_by=current_user.id,
    )
    db.add(patient)
    await db.commit()
    await db.refresh(patient)
    return patient


# ── GET /api/v4/patients ──────────────────────────────────────────────────────

@router.get(
    "/",
    response_model=PatientPage,
    summary="List patients (paginated, searchable)",
)
async def list_patients(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=200),
    search: Optional[str] = Query(None, description="Search by name or MRN (case-insensitive)"),
    include_deleted: bool = Query(False, description="Include soft-deleted records (super_admin only)"),
    _user: User = Depends(require_role(*CLINICAL_ROLES)),
    db: AsyncSession = Depends(get_db),
) -> PatientPage:
    """Return a paginated list of patients, optionally filtered by name or MRN."""
    filters = []

    if not include_deleted:
        filters.append(Patient.deleted_at.is_(None))

    if search:
        pattern = f"%{search}%"
        filters.append(
            or_(
                Patient.full_name.ilike(pattern),
                Patient.mrn.ilike(pattern),
            )
        )

    where_clause = and_(*filters) if filters else True

    total: int = (
        await db.execute(select(func.count()).select_from(Patient).where(where_clause))
    ).scalar_one()

    offset = (page - 1) * limit
    rows = (
        await db.execute(
            select(Patient)
            .where(where_clause)
            .order_by(Patient.created_at.desc())
            .offset(offset)
            .limit(limit)
        )
    ).scalars().all()

    return PatientPage(
        total=total,
        page=page,
        limit=limit,
        pages=max(1, -(-total // limit)),
        items=rows,
    )


# ── GET /api/v4/patients/{patient_id} ────────────────────────────────────────

@router.get(
    "/{patient_id}",
    response_model=PatientOut,
    summary="Get a single patient by ID",
)
async def get_patient(
    patient_id: str,
    _user: User = Depends(require_role(*CLINICAL_ROLES)),
    db: AsyncSession = Depends(get_db),
) -> PatientOut:
    return await _get_active_patient(patient_id, db)


# ── DELETE /api/v4/patients/{patient_id} ─────────────────────────────────────

@router.delete(
    "/{patient_id}",
    response_model=PatientOut,
    summary="Soft-delete a patient record (GDPR-compliant)",
)
async def delete_patient(
    patient_id: str,
    current_user: User = Depends(require_role(*ADMIN_ROLES)),
    db: AsyncSession = Depends(get_db),
) -> PatientOut:
    """
    Marks the patient as deleted by setting `deleted_at = now()`.
    The record and all linked predictions are retained for compliance.
    Only super_admin may delete patient records.
    """
    patient = await _get_active_patient(patient_id, db)
    patient.deleted_at = datetime.now(timezone.utc)
    await db.commit()
    await db.refresh(patient)
    return patient


# ── GET /api/v4/patients/{patient_id}/predictions ────────────────────────────

@router.get(
    "/{patient_id}/predictions",
    response_model=PredictionPage,
    summary="Prediction history for a patient",
)
async def get_patient_predictions(
    patient_id: str,
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=200),
    disease: Optional[str] = Query(None, description="Filter by disease name"),
    _user: User = Depends(require_role(*CLINICAL_ROLES)),
    db: AsyncSession = Depends(get_db),
) -> PredictionPage:
    """
    Returns all predictions linked to this patient, newest first.
    Optionally filtered by disease (e.g. ?disease=heart_disease).
    """
    await _get_active_patient(patient_id, db)  # 404 if not found

    filters = [Prediction.patient_id == patient_id]
    if disease:
        filters.append(Prediction.disease == disease)

    where_clause = and_(*filters)

    total: int = (
        await db.execute(
            select(func.count()).select_from(Prediction).where(where_clause)
        )
    ).scalar_one()

    offset = (page - 1) * limit
    rows = (
        await db.execute(
            select(Prediction)
            .where(where_clause)
            .order_by(Prediction.created_at.desc())
            .offset(offset)
            .limit(limit)
        )
    ).scalars().all()

    return PredictionPage(
        total=total,
        page=page,
        limit=limit,
        pages=max(1, -(-total // limit)),
        items=rows,
    )


# ── GET /api/v4/patients/{patient_id}/export ─────────────────────────────────

@router.get(
    "/{patient_id}/export",
    response_model=PatientExportBundle,
    summary="Export all patient data and predictions as JSON bundle",
)
async def export_patient(
    patient_id: str,
    _user: User = Depends(require_role(*CLINICAL_ROLES)),
    db: AsyncSession = Depends(get_db),
) -> PatientExportBundle:
    """
    Returns a complete data portability bundle for the patient:
      - Demographic record
      - All linked predictions (including SHAP data)

    Designed for GDPR data subject access requests and EMR integration.
    """
    patient = await _get_active_patient(patient_id, db)

    predictions = (
        await db.execute(
            select(Prediction)
            .where(Prediction.patient_id == patient_id)
            .order_by(Prediction.created_at.desc())
        )
    ).scalars().all()

    return PatientExportBundle(
        exported_at=datetime.now(timezone.utc),
        patient=patient,
        predictions=predictions,
        total_predictions=len(predictions),
    )
