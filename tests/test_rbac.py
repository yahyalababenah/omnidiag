"""
Tests — Feature 2.3: RBAC
Verifies that clinical endpoints enforce role requirements.
"""

import pytest

HEART_PAYLOAD = {
    "Age": 55,
    "Sex": "M",
    "ChestPainType": "ATA",
    "RestingBP": 130,
    "Cholesterol": 250,
    "FastingBS": 0,
    "RestingECG": "Normal",
    "MaxHR": 150,
    "ExerciseAngina": "N",
    "Oldpeak": 1.5,
    "ST_Slope": "Up",
}


class TestUnauthenticated:
    async def test_predict_requires_auth(self, client, db_tables):
        resp = await client.post("/api/v4/heart_disease/predict", json=HEART_PAYLOAD)
        assert resp.status_code in (401, 403)

    async def test_explain_requires_auth(self, client, db_tables):
        resp = await client.post("/api/v4/heart_disease/explain", json=HEART_PAYLOAD)
        assert resp.status_code in (401, 403)

    async def test_counterfactuals_requires_auth(self, client, db_tables):
        resp = await client.post("/api/v4/heart_disease/counterfactuals", json=HEART_PAYLOAD)
        assert resp.status_code in (401, 403)

    async def test_admin_audit_logs_requires_auth(self, client, db_tables):
        resp = await client.get("/admin/audit-logs")
        assert resp.status_code in (401, 403)

    async def test_patients_requires_auth(self, client, db_tables):
        resp = await client.get("/api/v4/patients/")
        assert resp.status_code in (401, 403)


class TestViewerRole:
    """Viewer role must NOT access clinical or admin routes."""

    async def test_viewer_cannot_predict(self, client, viewer_token):
        resp = await client.post(
            "/api/v4/heart_disease/predict",
            json=HEART_PAYLOAD,
            headers={"Authorization": f"Bearer {viewer_token}"},
        )
        assert resp.status_code == 403

    async def test_viewer_cannot_explain(self, client, viewer_token):
        resp = await client.post(
            "/api/v4/heart_disease/explain",
            json=HEART_PAYLOAD,
            headers={"Authorization": f"Bearer {viewer_token}"},
        )
        assert resp.status_code == 403

    async def test_viewer_cannot_access_admin(self, client, viewer_token):
        resp = await client.get(
            "/admin/audit-logs",
            headers={"Authorization": f"Bearer {viewer_token}"},
        )
        assert resp.status_code == 403


class TestDoctorRole:
    """Doctor role should have clinical access but NOT admin access."""

    async def test_doctor_can_predict(self, client, doctor_token):
        resp = await client.post(
            "/api/v4/heart_disease/predict",
            json=HEART_PAYLOAD,
            headers={"Authorization": f"Bearer {doctor_token}"},
        )
        assert resp.status_code == 200

    async def test_doctor_can_explain(self, client, doctor_token):
        resp = await client.post(
            "/api/v4/heart_disease/explain",
            json=HEART_PAYLOAD,
            headers={"Authorization": f"Bearer {doctor_token}"},
        )
        assert resp.status_code == 200

    async def test_doctor_cannot_access_admin(self, client, doctor_token):
        resp = await client.get(
            "/admin/audit-logs",
            headers={"Authorization": f"Bearer {doctor_token}"},
        )
        assert resp.status_code == 403

    async def test_doctor_cannot_flush_cache(self, client, doctor_token):
        resp = await client.post(
            "/admin/cache/flush",
            headers={"Authorization": f"Bearer {doctor_token}"},
        )
        assert resp.status_code == 403


class TestAdminRole:
    """super_admin should have access to everything."""

    async def test_admin_can_predict(self, client, admin_token):
        resp = await client.post(
            "/api/v4/heart_disease/predict",
            json=HEART_PAYLOAD,
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert resp.status_code == 200

    async def test_admin_can_access_audit_logs(self, client, admin_token):
        resp = await client.get(
            "/admin/audit-logs",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert resp.status_code == 200

    async def test_admin_can_flush_cache(self, client, admin_token):
        resp = await client.post(
            "/admin/cache/flush",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert resp.status_code == 200
