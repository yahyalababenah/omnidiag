"""
Gate 8.7c — POST /admin/drift/{disease}/run over real HTTP.

Before this, `/status` had one HTTP-level test and `/run` had none at all: the
route that actually builds the DataFrame from stored predictions, computes
drift and updates the gauge was covered only by direct calls to the monitor.
This exercises the real route: real DB rows, real auth, real JSON response.
"""
from __future__ import annotations

import pytest
from sqlalchemy import select

from backend.db_models.patient import Patient
from backend.db_models.prediction import Prediction
from tests.conftest import TestSessionLocal

# Zurich-like inputs: cholesterol almost never recorded, fasting blood sugar
# skewed toward positive, resting ECG mostly abnormal -- a real shift the seeded
# drift test in test_drift_stats.py already showed the monitor catches, seeded
# here as stored predictions instead of read straight from the training CSV.
_ZURICH_LIKE = [
    {"Age": 55 + i % 20, "Sex": "M" if i % 3 else "F", "ChestPainType": "ASY",
     "RestingBP": 130 + i % 15, "Cholesterol": None, "FastingBS": 1,
     "RestingECG": "ST", "MaxHR": 140, "ExerciseAngina": "N", "Oldpeak": 1.0,
     "ST_Slope": "Flat"}
    for i in range(60)
]


_counter = iter(range(10_000_000))


async def _seed_predictions(rows, disease="heart_disease"):
    """Each call gets fresh patient/prediction ids -- the DB is shared across
    tests in this module (found by running them together: reusing 'DRIFT-P0'
    across calls hit predictions.id's UNIQUE constraint)."""
    async with TestSessionLocal() as s:
        for features in rows:
            n = next(_counter)
            pid = f"DRIFT-P{n}"
            s.add(Patient(id=pid, mrn=pid, full_name="Drift Test"))
            await s.flush()
            s.add(Prediction(
                id=f"pred-drift-{disease}-{n}", patient_id=pid, disease=disease,
                input_features=features, prediction=1, confidence=0.6,
                diagnosis="Positive", probability_scale="corrected",
            ))
        await s.commit()


@pytest.mark.asyncio
class TestDriftRunRoute:
    async def test_run_returns_the_seven_features_ks_chi2_and_psi(
        self, client, admin_token, db_tables
    ):
        await _seed_predictions(_ZURICH_LIKE)
        resp = await client.post(
            "/api/v4/admin/drift/heart_disease/run",
            headers={"Authorization": f"Bearer {admin_token}"},
            json={"sample_size": 100},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["monitor_ready"] is True
        report = body["report"]
        assert report["status"] == "ok"
        feature_names = {f["feature"] for f in report["features"]}
        assert feature_names == {"Age", "RestingBP", "Cholesterol", "Sex_m",
                                 "cp_anginal", "FastingBS_cat", "RestingECG"}
        for f in report["features"]:
            assert "psi" in f and "p_value" in f and "drifted" in f
        assert report["metrics"]["dataset_drift"]["number_of_columns"] == 7
        # This population's cholesterol is never recorded -- the seeded fault
        # that a missingness-blind test would have missed (found while building
        # this gate: KS alone cannot see a feature that stopped being measured).
        assert "Cholesterol" in report["drifted_features"]

    async def test_run_states_a_named_status_not_only_a_number(
        self, client, admin_token, db_tables
    ):
        """A drift_share of 0.0 must never be the answer for 'too few rows'.

        `sample_size` bounds the query itself (10 is the route's own floor,
        below the monitor's 50-row threshold), so this holds regardless of how
        many rows other tests in this module have already put in the shared DB.
        """
        await _seed_predictions(_ZURICH_LIKE[:3])
        resp = await client.post(
            "/api/v4/admin/drift/heart_disease/run",
            headers={"Authorization": f"Bearer {admin_token}"},
            json={"sample_size": 10},
        )
        assert resp.status_code == 200, resp.text
        report = resp.json()["report"]
        assert report["status"] == "insufficient_data"
        assert report["metrics"]["dataset_drift"]["drift_share"] is None

    async def test_run_requires_admin(self, client, doctor_token, db_tables):
        resp = await client.post(
            "/api/v4/admin/drift/heart_disease/run",
            headers={"Authorization": f"Bearer {doctor_token}"},
            json={"sample_size": 100},
        )
        assert resp.status_code == 403

    async def test_run_updates_the_prometheus_gauge(
        self, client, admin_token, db_tables
    ):
        from backend.monitoring.metrics import get_metrics_response
        await _seed_predictions(_ZURICH_LIKE)
        resp = await client.post(
            "/api/v4/admin/drift/heart_disease/run",
            headers={"Authorization": f"Bearer {admin_token}"},
            json={"sample_size": 100},
        )
        share = resp.json()["report"]["metrics"]["dataset_drift"]["drift_share"]
        body, _ = get_metrics_response()
        text = body.decode()
        assert 'omnidiag_drift_share{disease="heart_disease"}' in text
        line = next(l for l in text.splitlines()
                   if l.startswith('omnidiag_drift_share{disease="heart_disease"}'))
        assert float(line.rsplit(" ", 1)[1]) == pytest.approx(share)
