"""
Tests — Feature 2.4: Patient Records & Prediction History
Uses module-scoped DB (data persists across tests), so every test uses a unique MRN.
"""

import uuid
import pytest


def _patient(suffix: str = None):
    """Return a unique patient payload."""
    uid = suffix or uuid.uuid4().hex[:8]
    return {
        "mrn": f"MRN-{uid}",
        "full_name": f"Patient {uid}",
        "date_of_birth": "1980-05-15",
        "gender": "female",
        "contact_email": f"patient-{uid}@example.com",
    }


class TestCreatePatient:
    async def test_create_patient_success(self, client, doctor_token):
        resp = await client.post(
            "/api/v4/patients/",
            json=_patient("A001"),
            headers={"Authorization": f"Bearer {doctor_token}"},
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["mrn"] == "MRN-A001"
        assert "id" in data

    async def test_create_patient_duplicate_mrn(self, client, doctor_token):
        p = _patient("DUP1")
        await client.post(
            "/api/v4/patients/",
            json=p,
            headers={"Authorization": f"Bearer {doctor_token}"},
        )
        resp = await client.post(
            "/api/v4/patients/",
            json=p,
            headers={"Authorization": f"Bearer {doctor_token}"},
        )
        assert resp.status_code == 409

    async def test_create_patient_requires_auth(self, client, db_tables):
        resp = await client.post("/api/v4/patients/", json=_patient("NOAUTH"))
        assert resp.status_code in (401, 403)


class TestListPatients:
    async def test_list_patients(self, client, doctor_token):
        for uid in ["LIST1", "LIST2"]:
            await client.post(
                "/api/v4/patients/",
                json=_patient(uid),
                headers={"Authorization": f"Bearer {doctor_token}"},
            )

        resp = await client.get(
            "/api/v4/patients/",
            headers={"Authorization": f"Bearer {doctor_token}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "items" in data
        assert data["total"] >= 2

    async def test_list_patients_search_by_name(self, client, doctor_token):
        uid = "SRCHNAME"
        await client.post(
            "/api/v4/patients/",
            json={**_patient(uid), "full_name": f"UniqueSearchable {uid}"},
            headers={"Authorization": f"Bearer {doctor_token}"},
        )
        resp = await client.get(
            f"/api/v4/patients/?search=UniqueSearchable",
            headers={"Authorization": f"Bearer {doctor_token}"},
        )
        assert resp.status_code == 200
        items = resp.json()["items"]
        assert any("UniqueSearchable" in p["full_name"] for p in items)

    async def test_list_patients_search_by_mrn(self, client, doctor_token):
        uid = "SRCHMRN"
        p = _patient(uid)
        await client.post(
            "/api/v4/patients/",
            json=p,
            headers={"Authorization": f"Bearer {doctor_token}"},
        )
        resp = await client.get(
            f"/api/v4/patients/?search={p['mrn']}",
            headers={"Authorization": f"Bearer {doctor_token}"},
        )
        assert resp.status_code == 200
        items = resp.json()["items"]
        assert any(x["mrn"] == p["mrn"] for x in items)


class TestGetPatient:
    async def test_get_patient_by_id(self, client, doctor_token):
        p = _patient("GET1")
        create_resp = await client.post(
            "/api/v4/patients/",
            json=p,
            headers={"Authorization": f"Bearer {doctor_token}"},
        )
        assert create_resp.status_code == 201
        patient_id = create_resp.json()["id"]

        resp = await client.get(
            f"/api/v4/patients/{patient_id}",
            headers={"Authorization": f"Bearer {doctor_token}"},
        )
        assert resp.status_code == 200
        assert resp.json()["mrn"] == p["mrn"]

    async def test_get_patient_not_found(self, client, doctor_token):
        resp = await client.get(
            "/api/v4/patients/00000000-0000-0000-0000-000000000000",
            headers={"Authorization": f"Bearer {doctor_token}"},
        )
        assert resp.status_code == 404


class TestSoftDeletePatient:
    async def test_soft_delete_requires_admin(self, client, doctor_token):
        create_resp = await client.post(
            "/api/v4/patients/",
            json=_patient("DEL0"),
            headers={"Authorization": f"Bearer {doctor_token}"},
        )
        patient_id = create_resp.json()["id"]

        resp = await client.delete(
            f"/api/v4/patients/{patient_id}",
            headers={"Authorization": f"Bearer {doctor_token}"},
        )
        assert resp.status_code == 403

    async def test_soft_delete_by_admin(self, client, admin_token, doctor_token):
        create_resp = await client.post(
            "/api/v4/patients/",
            json=_patient("DEL1"),
            headers={"Authorization": f"Bearer {doctor_token}"},
        )
        patient_id = create_resp.json()["id"]

        del_resp = await client.delete(
            f"/api/v4/patients/{patient_id}",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert del_resp.status_code == 200

        get_resp = await client.get(
            f"/api/v4/patients/{patient_id}",
            headers={"Authorization": f"Bearer {doctor_token}"},
        )
        assert get_resp.status_code == 404

    async def test_deleted_patient_not_in_list(self, client, admin_token, doctor_token):
        create_resp = await client.post(
            "/api/v4/patients/",
            json=_patient("DEL2"),
            headers={"Authorization": f"Bearer {doctor_token}"},
        )
        patient_id = create_resp.json()["id"]

        await client.delete(
            f"/api/v4/patients/{patient_id}",
            headers={"Authorization": f"Bearer {admin_token}"},
        )

        list_resp = await client.get(
            "/api/v4/patients/",
            headers={"Authorization": f"Bearer {doctor_token}"},
        )
        ids = [p["id"] for p in list_resp.json()["items"]]
        assert patient_id not in ids


class TestPredictionHistory:
    async def test_patient_predictions_empty(self, client, doctor_token):
        create_resp = await client.post(
            "/api/v4/patients/",
            json=_patient("PRED1"),
            headers={"Authorization": f"Bearer {doctor_token}"},
        )
        patient_id = create_resp.json()["id"]

        resp = await client.get(
            f"/api/v4/patients/{patient_id}/predictions",
            headers={"Authorization": f"Bearer {doctor_token}"},
        )
        assert resp.status_code == 200
        assert resp.json()["total"] == 0

    async def test_patient_export_bundle(self, client, doctor_token):
        create_resp = await client.post(
            "/api/v4/patients/",
            json=_patient("EXP1"),
            headers={"Authorization": f"Bearer {doctor_token}"},
        )
        patient_id = create_resp.json()["id"]

        resp = await client.get(
            f"/api/v4/patients/{patient_id}/export",
            headers={"Authorization": f"Bearer {doctor_token}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "patient" in data
        assert "predictions" in data
        assert "exported_at" in data
