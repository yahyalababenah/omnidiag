"""
Integration Tests — AuditMiddleware (I-7 through I-11)
Verifies that every API request is recorded in the audit_logs table.

The AuditMiddleware used to open AsyncSessionLocal from backend.database directly,
which bypassed the get_db override and wrote to the real database, so this module
patched it locally. Every OTHER test module was left writing to omnidiag_dev.db
(WEAKNESS_REGISTER.md P-13). The middleware now resolves its session through
backend.database.app_session, which honours app.dependency_overrides[get_db], so
no patch is needed and every module is isolated the same way.
"""

import pytest
from sqlalchemy import select

from backend.db_models.audit_log import AuditLog

# TestSessionLocal is the in-memory session factory defined in conftest.py
from tests.conftest import TestSessionLocal

HEART_PAYLOAD = {
    "Age": 55, "Sex": "M", "ChestPainType": "ATA", "RestingBP": 130,
    "Cholesterol": 250, "FastingBS": 0, "RestingECG": "Normal",
    "MaxHR": 150, "ExerciseAngina": "N", "Oldpeak": 1.5, "ST_Slope": "Up",
}


@pytest.fixture(autouse=True)
def audit_tables(db_tables):
    """Ensure the test tables exist; the middleware reaches them via the get_db override."""
    yield


class TestAuditMiddlewareLogsRequests:
    async def test_authenticated_predict_creates_audit_log(self, client, doctor_token, db_session):
        resp = await client.post(
            "/api/v4/heart_disease/predict",
            json=HEART_PAYLOAD,
            headers={"Authorization": f"Bearer {doctor_token}"},
        )
        assert resp.status_code == 200

        result = await db_session.execute(
            select(AuditLog).where(AuditLog.endpoint == "/api/v4/heart_disease/predict")
        )
        logs = result.scalars().all()
        assert len(logs) >= 1, "Expected at least one audit log entry for /predict"

    async def test_audit_log_has_correct_status_code(self, client, doctor_token, db_session):
        resp = await client.post(
            "/api/v4/heart_disease/predict",
            json=HEART_PAYLOAD,
            headers={"Authorization": f"Bearer {doctor_token}"},
        )
        result = await db_session.execute(
            select(AuditLog).where(AuditLog.endpoint == "/api/v4/heart_disease/predict")
        )
        logs = result.scalars().all()
        assert any(log.status_code == 200 for log in logs)

    async def test_audit_log_duration_ms_positive(self, client, doctor_token, db_session):
        await client.post(
            "/api/v4/heart_disease/predict",
            json=HEART_PAYLOAD,
            headers={"Authorization": f"Bearer {doctor_token}"},
        )
        result = await db_session.execute(
            select(AuditLog).where(AuditLog.endpoint == "/api/v4/heart_disease/predict")
        )
        logs = result.scalars().all()
        assert all(log.duration_ms is not None and log.duration_ms > 0 for log in logs)

    async def test_audit_log_records_user_id_for_authenticated_request(
        self, client, doctor_token, db_session
    ):
        await client.post(
            "/api/v4/heart_disease/predict",
            json=HEART_PAYLOAD,
            headers={"Authorization": f"Bearer {doctor_token}"},
        )
        result = await db_session.execute(
            select(AuditLog).where(AuditLog.endpoint == "/api/v4/heart_disease/predict")
        )
        logs = result.scalars().all()
        authenticated_logs = [log for log in logs if log.user_id is not None]
        assert len(authenticated_logs) >= 1

    async def test_unauthenticated_request_logs_null_user_id(self, client, db_session):
        # Login endpoint is always logged; unauthenticated POST with invalid creds
        await client.post(
            "/auth/login",
            json={"email": "nonexistent@test.com", "password": "wrongpassword"},
        )
        result = await db_session.execute(
            select(AuditLog).where(
                AuditLog.endpoint == "/auth/login",
                AuditLog.user_id.is_(None),
            )
        )
        logs = result.scalars().all()
        assert len(logs) >= 1, "Expected audit log with null user_id for failed login"

    async def test_audit_log_records_http_method(self, client, doctor_token, db_session):
        await client.post(
            "/api/v4/heart_disease/predict",
            json=HEART_PAYLOAD,
            headers={"Authorization": f"Bearer {doctor_token}"},
        )
        result = await db_session.execute(
            select(AuditLog).where(AuditLog.endpoint == "/api/v4/heart_disease/predict")
        )
        logs = result.scalars().all()
        assert any(log.method == "POST" for log in logs)
