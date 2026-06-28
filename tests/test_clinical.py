"""
Tests — Clinical Endpoints: /predict, /explain, /counterfactuals
Uses the mocked OmniDiagRouter — no ML models loaded.
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


class TestPredict:
    async def test_predict_returns_result(self, client, doctor_token):
        resp = await client.post(
            "/api/v4/heart_disease/predict",
            json=HEART_PAYLOAD,
            headers={"Authorization": f"Bearer {doctor_token}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "prediction" in data
        assert "confidence" in data
        assert "diagnosis" in data

    async def test_predict_positive_case(self, client, doctor_token):
        resp = await client.post(
            "/api/v4/heart_disease/predict",
            json=HEART_PAYLOAD,
            headers={"Authorization": f"Bearer {doctor_token}"},
        )
        data = resp.json()
        # Mock router always returns prediction=1, confidence=0.82
        assert data["prediction"] == 1
        assert data["confidence"] == pytest.approx(0.82, abs=0.01)

    async def test_predict_sets_cache_hit_header(self, client, doctor_token):
        resp = await client.post(
            "/api/v4/heart_disease/predict",
            json=HEART_PAYLOAD,
            headers={"Authorization": f"Bearer {doctor_token}"},
        )
        assert resp.status_code == 200
        assert "cache-hit" in resp.headers

    async def test_predict_cache_hit_header_present(self, client, doctor_token):
        # Cache-Hit header is always set (either "true" or "false")
        resp = await client.post(
            "/api/v4/heart_disease/predict",
            json=HEART_PAYLOAD,
            headers={"Authorization": f"Bearer {doctor_token}"},
        )
        assert resp.headers.get("cache-hit") in ("true", "false")

    async def test_predict_with_patient_id_skips_cache(self, client, doctor_token, seeded_db):
        # Create a patient first
        create_resp = await client.post(
            "/api/v4/patients/",
            json={
                "mrn": "MRN-PRED-001",
                "full_name": "Cache Test",
                "date_of_birth": "1985-01-01",
                "gender": "male",
                "contact_email": "cache@test.com",
            },
            headers={"Authorization": f"Bearer {doctor_token}"},
        )
        patient_id = create_resp.json()["id"]

        resp = await client.post(
            f"/api/v4/heart_disease/predict?patient_id={patient_id}",
            json=HEART_PAYLOAD,
            headers={"Authorization": f"Bearer {doctor_token}"},
        )
        assert resp.status_code == 200
        # Cache is always bypassed when patient_id is provided
        assert resp.headers.get("cache-hit") == "false"


class TestExplain:
    async def test_explain_returns_shap_data(self, client, doctor_token):
        resp = await client.post(
            "/api/v4/heart_disease/explain",
            json=HEART_PAYLOAD,
            headers={"Authorization": f"Bearer {doctor_token}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        # ExplainResponse schema: chart_data, text_explanation, base_value
        assert "chart_data" in data
        assert "text_explanation" in data
        assert "base_value" in data

    async def test_explain_chart_data_structure(self, client, doctor_token):
        resp = await client.post(
            "/api/v4/heart_disease/explain",
            json=HEART_PAYLOAD,
            headers={"Authorization": f"Bearer {doctor_token}"},
        )
        chart_data = resp.json()["chart_data"]
        assert isinstance(chart_data, list)
        if chart_data:
            entry = chart_data[0]
            assert "feature" in entry
            assert "shap_value" in entry


class TestCounterfactuals:
    async def test_counterfactuals_returns_scenarios(self, client, doctor_token):
        resp = await client.post(
            "/api/v4/heart_disease/counterfactuals",
            json=HEART_PAYLOAD,
            headers={"Authorization": f"Bearer {doctor_token}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert "counterfactuals" in data

    async def test_counterfactuals_baseline_probability(self, client, doctor_token):
        resp = await client.post(
            "/api/v4/heart_disease/counterfactuals",
            json=HEART_PAYLOAD,
            headers={"Authorization": f"Bearer {doctor_token}"},
        )
        data = resp.json()
        assert "baseline_probability" in data
        assert 0.0 <= data["baseline_probability"] <= 1.0


class TestDiseaseSchema:
    async def test_get_schema_returns_json_schema(self, client, db_tables):
        resp = await client.get("/api/v4/heart_disease/schema")
        assert resp.status_code == 200
        data = resp.json()
        assert "properties" in data or "title" in data

    async def test_get_schema_sets_cache_header(self, client, db_tables):
        resp = await client.get("/api/v4/heart_disease/schema")
        assert resp.status_code == 200
        assert "cache-hit" in resp.headers

    async def test_list_diseases(self, client, db_tables):
        resp = await client.get("/api/v4/diseases")
        assert resp.status_code == 200
        diseases = resp.json()["diseases"]
        names = [d["name"] for d in diseases]
        assert "heart_disease" in names
        assert "diabetes" in names
