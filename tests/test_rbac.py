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
    """
    predict / explain / counterfactuals are intentionally open to anonymous
    callers so the public demo can be tried without an account — they use
    Depends(get_optional_user), and results are only persisted for
    authenticated users. Everything that touches stored data (batch, admin,
    patients) still requires a role.
    """

    async def test_predict_allows_anonymous(self, client, db_tables):
        resp = await client.post("/api/v4/heart_disease/predict", json=HEART_PAYLOAD)
        assert resp.status_code == 200

    async def test_explain_allows_anonymous(self, client, db_tables):
        resp = await client.post("/api/v4/heart_disease/explain", json=HEART_PAYLOAD)
        assert resp.status_code == 200

    async def test_counterfactuals_allows_anonymous(self, client, db_tables):
        resp = await client.post("/api/v4/heart_disease/counterfactuals", json=HEART_PAYLOAD)
        # 200 with scenarios, or 501 when the loader has no CF generator —
        # either way it must not be an auth rejection.
        assert resp.status_code not in (401, 403)

    async def test_batch_requires_auth(self, client, db_tables):
        resp = await client.post("/api/v4/heart_disease/batch")
        assert resp.status_code in (401, 403)

    async def test_admin_audit_logs_requires_auth(self, client, db_tables):
        resp = await client.get("/admin/audit-logs")
        assert resp.status_code in (401, 403)

    async def test_patients_requires_auth(self, client, db_tables):
        resp = await client.get("/api/v4/patients/")
        assert resp.status_code in (401, 403)


class TestViewerRole:
    """
    Viewer role must NOT access admin routes or anything that writes stored
    data. predict/explain are open to everyone (including anonymous), so a
    viewer token is no more restricted there than no token at all.
    """

    async def test_viewer_can_predict_like_anonymous(self, client, viewer_token):
        resp = await client.post(
            "/api/v4/heart_disease/predict",
            json=HEART_PAYLOAD,
            headers={"Authorization": f"Bearer {viewer_token}"},
        )
        assert resp.status_code == 200

    async def test_viewer_can_explain_like_anonymous(self, client, viewer_token):
        resp = await client.post(
            "/api/v4/heart_disease/explain",
            json=HEART_PAYLOAD,
            headers={"Authorization": f"Bearer {viewer_token}"},
        )
        assert resp.status_code == 200

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
