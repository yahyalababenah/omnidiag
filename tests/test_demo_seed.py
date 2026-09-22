"""
X-5 — demo patients and their History must survive a container restart.

The live DB is SQLite inside the Space, wiped by every restart and rebuild.
Nothing in the product ever created a Patient row, so
GET /api/v4/patients/P-001/predictions answered 404 and History could not
work at all -- not "was empty", could not work.

These tests cover the three things that made it fail: the patients exist,
they have history, and re-running the seeder does not pile up duplicates.
"""

import json
import os
import subprocess
from unittest.mock import MagicMock

import pytest
from sqlalchemy import func, select

from backend.db_models.patient import Patient
from backend.db_models.patient_visit import PatientVisit
from backend.db_models.prediction import Prediction
from backend import demo_seed
from tests.conftest import TestSessionLocal

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _stub_router():
    """A router whose probability depends on the features it is handed."""
    router = MagicMock()

    def _predict(disease, features):
        # Make the score move with the lever field, so a flat timeline (the
        # seeder ignoring steps_back) is detectable.
        lever = demo_seed.HISTORY_LEVERS[disease]["field"]
        value = float(features.get(lever, 0) or 0)
        confidence = min(0.99, value / 1000.0)
        return {
            "prediction": 1 if confidence > 0.1 else 0,
            "confidence": confidence,
            "diagnosis": "Positive" if confidence > 0.1 else "Negative",
        }

    router.predict.side_effect = _predict
    return router


class TestDemoPatientSource:
    def test_json_matches_the_frontends_own_mock_patients(self):
        """
        backend/demo_patients.json is generated from
        frontend/src/mockPatients.js. If the two drift, the History timeline
        stops matching the patient card on screen.
        """
        src = os.path.join(ROOT, "frontend", "src", "mockPatients.js")
        out = subprocess.run(
            ["node", "-e",
             "import(process.argv[1]).then(m => console.log(JSON.stringify(m.default)))",
             src],
            capture_output=True, text=True, check=True,
        )
        live = json.loads(out.stdout)
        seeded = demo_seed.load_demo_patients()

        assert set(seeded) == set(live), "disease keys differ from mockPatients.js"
        for disease, patients in live.items():
            assert [p["id"] for p in patients] == [p["id"] for p in seeded[disease]]
            for actual, stored in zip(patients, seeded[disease]):
                assert stored["name"] == actual["name"]
                assert stored["data"] == actual["data"], (
                    f"{disease}/{actual['id']} feature values have drifted from "
                    f"mockPatients.js — regenerate backend/demo_patients.json"
                )


class TestHistoricalFeatures:
    def test_todays_entry_is_the_patients_current_record_untouched(self):
        current = {"Cholesterol": 289, "Age": 54}
        assert demo_seed.historical_features(current, "heart_disease", 0) == current

    def test_earlier_visits_move_only_the_documented_lever(self):
        current = {"Cholesterol": 289, "Age": 54, "RestingBP": 140}
        past = demo_seed.historical_features(current, "heart_disease", 2)
        assert past["Cholesterol"] != current["Cholesterol"]
        assert past["Age"] == 54 and past["RestingBP"] == 140

    def test_the_lever_stays_inside_its_clinical_range(self):
        current = {"Cholesterol": 395}
        past = demo_seed.historical_features(current, "heart_disease", 10)
        assert past["Cholesterol"] <= demo_seed.HISTORY_LEVERS["heart_disease"]["maximum"]

    def test_an_unknown_disease_is_left_alone_rather_than_guessed(self):
        current = {"Whatever": 1}
        assert demo_seed.historical_features(current, "not_a_disease", 3) == current


@pytest.mark.asyncio
class TestSeeding:
    async def _counts(self):
        async with TestSessionLocal() as s:
            return (
                (await s.execute(select(func.count()).select_from(Patient))).scalar_one(),
                (await s.execute(select(func.count()).select_from(Prediction))).scalar_one(),
                (await s.execute(select(func.count()).select_from(PatientVisit))).scalar_one(),
            )

    async def test_seeding_creates_every_demo_patient_with_history(
        self, db_tables, monkeypatch
    ):
        monkeypatch.setattr(demo_seed, "AsyncSessionLocal", TestSessionLocal)
        demo = demo_seed.load_demo_patients()
        expected_patients = sum(len(p) for p in demo.values())

        patients_before, predictions_before, visits_before = await self._counts()
        written = await demo_seed.seed_demo_history(_stub_router())
        patients_after, predictions_after, visits_after = await self._counts()

        assert written == expected_patients * demo_seed.VISITS_PER_PATIENT
        assert patients_after == patients_before + expected_patients
        assert predictions_after == predictions_before + written
        assert visits_after == visits_before + written

    async def test_re_running_on_a_seeded_db_writes_nothing(self, db_tables, monkeypatch):
        """
        Startup runs this every time. A second run must not double the
        timeline.
        """
        monkeypatch.setattr(demo_seed, "AsyncSessionLocal", TestSessionLocal)
        await demo_seed.seed_demo_history(_stub_router())
        before = await self._counts()

        written = await demo_seed.seed_demo_history(_stub_router())

        assert written == 0
        assert await self._counts() == before

    async def test_history_is_a_trajectory_not_a_flat_line(self, db_tables, monkeypatch):
        monkeypatch.setattr(demo_seed, "AsyncSessionLocal", TestSessionLocal)
        await demo_seed.seed_demo_history(_stub_router())

        async with TestSessionLocal() as s:
            rows = (await s.execute(
                select(Prediction)
                .where(Prediction.patient_id == "P-001")
                .order_by(Prediction.created_at)
            )).scalars().all()

        assert len(rows) == demo_seed.VISITS_PER_PATIENT
        assert len({r.confidence for r in rows}) > 1, (
            "every seeded visit scored identically — History would draw a flat "
            "line and tell a judge nothing"
        )
        # Oldest first, and today's entry really is today.
        assert rows == sorted(rows, key=lambda r: r.created_at)

    async def test_todays_entry_matches_the_patient_card_on_screen(
        self, db_tables, monkeypatch
    ):
        monkeypatch.setattr(demo_seed, "AsyncSessionLocal", TestSessionLocal)
        await demo_seed.seed_demo_history(_stub_router())

        expected = next(
            p for p in demo_seed.load_demo_patients()["heart_disease"]
            if p["id"] == "P-001"
        )["data"]

        async with TestSessionLocal() as s:
            newest = (await s.execute(
                select(Prediction)
                .where(Prediction.patient_id == "P-001")
                .order_by(Prediction.created_at.desc())
                .limit(1)
            )).scalar_one()

        assert newest.input_features == expected, (
            "the newest History entry does not match the patient's current "
            "record — the timeline would contradict the EMR screen"
        )


@pytest.mark.asyncio
class TestHistoryEndpoint:
    async def test_history_endpoint_answers_for_a_demo_id(
        self, client, doctor_token, db_tables, monkeypatch
    ):
        """
        The actual failure a judge saw: 404 Patient 'P-001' not found.
        """
        monkeypatch.setattr(demo_seed, "AsyncSessionLocal", TestSessionLocal)
        await demo_seed.seed_demo_history(_stub_router())

        resp = await client.get(
            "/api/v4/patients/P-001/predictions",
            headers={"Authorization": f"Bearer {doctor_token}"},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["total"] >= demo_seed.VISITS_PER_PATIENT
        assert body["items"], "History returned an empty timeline"
