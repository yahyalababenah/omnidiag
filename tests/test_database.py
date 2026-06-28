"""
Database-level Tests — D-1 through D-5
Tests constraints, soft delete, FK behavior, and JSON storage using
db_session directly (no HTTP client).
"""

import uuid
from datetime import datetime, timezone

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from backend.db_models.patient import Patient
from backend.db_models.prediction import Prediction
from backend.db_models.audit_log import AuditLog


def _unique_mrn(prefix="MRN-DB"):
    return f"{prefix}-{uuid.uuid4().hex[:8].upper()}"


class TestUniqueConstraints:
    async def test_duplicate_mrn_raises_integrity_error(self, db_session):
        """D-1: Two patients with the same MRN must violate the unique constraint."""
        mrn = _unique_mrn()
        p1 = Patient(mrn=mrn, full_name="Patient A")
        p2 = Patient(mrn=mrn, full_name="Patient B")
        db_session.add(p1)
        db_session.add(p2)
        with pytest.raises(IntegrityError):
            await db_session.commit()
        # Roll back the failed transaction so the session stays usable
        await db_session.rollback()

    async def test_unique_mrns_do_not_conflict(self, db_session):
        """D-1b: Two patients with distinct MRNs both persist."""
        p1 = Patient(mrn=_unique_mrn("A"), full_name="Patient X")
        p2 = Patient(mrn=_unique_mrn("B"), full_name="Patient Y")
        db_session.add(p1)
        db_session.add(p2)
        await db_session.commit()

        result = await db_session.execute(select(Patient).where(Patient.id.in_([p1.id, p2.id])))
        found = result.scalars().all()
        assert len(found) == 2


class TestSoftDelete:
    async def test_soft_delete_sets_deleted_at(self, db_session):
        """D-2: Setting deleted_at marks the record without physically removing it."""
        patient = Patient(mrn=_unique_mrn(), full_name="Soft Delete Test")
        db_session.add(patient)
        await db_session.commit()
        await db_session.refresh(patient)

        patient.deleted_at = datetime.now(timezone.utc)
        await db_session.commit()
        await db_session.refresh(patient)

        # Row still exists in DB
        result = await db_session.execute(select(Patient).where(Patient.id == patient.id))
        found = result.scalar_one_or_none()
        assert found is not None
        assert found.deleted_at is not None

    async def test_soft_deleted_row_persists(self, db_session):
        """D-2b: A soft-deleted patient is physically present in the database."""
        patient = Patient(mrn=_unique_mrn(), full_name="Persists After Delete")
        db_session.add(patient)
        await db_session.commit()
        patient_id = patient.id

        patient.deleted_at = datetime.now(timezone.utc)
        await db_session.commit()

        # Re-fetch directly — no soft-delete filter at the DB level
        result = await db_session.execute(select(Patient).where(Patient.id == patient_id))
        assert result.scalar_one_or_none() is not None


class TestForeignKeyBehavior:
    async def test_prediction_survives_patient_deletion(self, db_session):
        """D-3: Predictions have ondelete='SET NULL' — they are not cascade-deleted."""
        patient = Patient(mrn=_unique_mrn(), full_name="FK Test Patient")
        db_session.add(patient)
        await db_session.commit()
        await db_session.refresh(patient)

        prediction = Prediction(
            patient_id=patient.id,
            disease="heart_disease",
            input_features={"Age": 50},
            prediction=1,
            confidence=0.8,
            diagnosis="Positive",
        )
        db_session.add(prediction)
        await db_session.commit()
        prediction_id = prediction.id

        # Delete the patient record at SQL level (bypasses ORM cascade)
        await db_session.delete(patient)
        await db_session.commit()

        # Prediction should still exist (SET NULL, not CASCADE DELETE)
        result = await db_session.execute(
            select(Prediction).where(Prediction.id == prediction_id)
        )
        found = result.scalar_one_or_none()
        assert found is not None, "Prediction must survive patient deletion (ondelete='SET NULL')"

    async def test_anonymous_prediction_allowed(self, db_session):
        """D-3b: Predictions can be created without a patient (nullable FK)."""
        prediction = Prediction(
            patient_id=None,
            disease="diabetes",
            input_features={"HighBP": 1},
            prediction=0,
            confidence=0.3,
            diagnosis="Negative",
        )
        db_session.add(prediction)
        await db_session.commit()
        await db_session.refresh(prediction)
        assert prediction.patient_id is None


class TestJSONStorage:
    async def test_json_features_persist_and_retrieve(self, db_session):
        """D-4: input_features JSON column stores and retrieves nested data correctly."""
        features = {"Age": 55, "Sex": "M", "Cholesterol": 250, "nested": {"key": "val"}}
        pred = Prediction(
            disease="heart_disease",
            input_features=features,
            prediction=1,
            confidence=0.85,
        )
        db_session.add(pred)
        await db_session.commit()
        await db_session.refresh(pred)

        assert pred.input_features["Age"] == 55
        assert pred.input_features["Sex"] == "M"
        assert pred.input_features["nested"]["key"] == "val"

    async def test_json_features_mutable(self, db_session):
        """D-4b: MutableDict.as_mutable allows in-place updates to be tracked."""
        pred = Prediction(
            disease="diabetes",
            input_features={"BMI": 30.0},
            prediction=0,
            confidence=0.4,
        )
        db_session.add(pred)
        await db_session.commit()

        pred.input_features["BMI"] = 28.5
        await db_session.commit()
        await db_session.refresh(pred)

        assert pred.input_features["BMI"] == 28.5


class TestAuditLogNullUser:
    async def test_audit_log_allows_null_user_id(self, db_session):
        """D-5: AuditLog must accept null user_id (unauthenticated requests)."""
        entry = AuditLog(
            user_id=None,
            endpoint="/auth/login",
            method="POST",
            status_code=401,
            ip_address="127.0.0.1",
            duration_ms=5.0,
        )
        db_session.add(entry)
        await db_session.commit()
        await db_session.refresh(entry)

        assert entry.id is not None
        assert entry.user_id is None
        assert entry.status_code == 401

    async def test_audit_log_persists_with_all_fields(self, db_session):
        """D-5b: AuditLog with all fields set persists correctly."""
        entry = AuditLog(
            user_id=None,
            endpoint="/api/v4/heart_disease/predict",
            method="POST",
            status_code=200,
            ip_address="::1",
            duration_ms=42.0,
        )
        db_session.add(entry)
        await db_session.commit()

        result = await db_session.execute(
            select(AuditLog).where(AuditLog.endpoint == "/api/v4/heart_disease/predict")
        )
        found = result.scalars().first()
        assert found is not None
        assert found.duration_ms == 42.0
        assert found.method == "POST"
