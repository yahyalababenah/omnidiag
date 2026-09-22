"""
7 + 11 — the clinician's own words must be kept.

Two places asked a human to type free text and then threw it away:

  * the notes-parser box was the only free-text field in the UI, and nothing
    stored it — History and the exported PDF had no record of what the doctor
    thought, only of what the regex extracted; and
  * POST /api/v4/review/{id}/annotate accepted a `notes` field and dropped it
    for want of a column, discarding the reviewer's reasoning and keeping
    only the bare 0/1.

Both are now columns. Neither is ever a model input, and the tests below pin
that down as well as the storage.
"""

import pytest
from sqlalchemy import select

from backend.db_models.prediction import Prediction
from backend.db_models.review_queue import ReviewQueue
from tests.conftest import TestSessionLocal

NOTE = "Patient reports exertional dyspnoea not captured by the form. Repeat ECG ordered."

PATIENT = {
    "Age": 57, "Sex": "M", "ChestPainType": "ASY", "RestingBP": 138,
    "Cholesterol": 240, "FastingBS": 0, "RestingECG": "Normal",
    "MaxHR": 141, "ExerciseAngina": "N", "Oldpeak": 0.8, "ST_Slope": "Flat",
}


async def _seed_patient_with_prediction(disease="heart_disease", patient_id="NOTE-P1"):
    from backend.db_models.patient import Patient
    async with TestSessionLocal() as s:
        if (await s.execute(select(Patient).where(Patient.id == patient_id))).scalar_one_or_none() is None:
            s.add(Patient(id=patient_id, mrn=patient_id, full_name="Note Test"))
            await s.flush()
        s.add(Prediction(
            id=f"pred-{patient_id}-{disease}",
            patient_id=patient_id,
            disease=disease,
            input_features=PATIENT,
            prediction=1,
            confidence=0.61,
            diagnosis="Positive",
        ))
        await s.commit()


@pytest.mark.asyncio
class TestDoctorNotesOnAPrediction:
    async def test_note_is_stored_against_the_latest_screening(
        self, client, doctor_token, db_tables
    ):
        await _seed_patient_with_prediction()
        resp = await client.post(
            "/api/v4/patients/NOTE-P1/notes",
            json={"disease": "heart_disease", "notes": NOTE},
            headers={"Authorization": f"Bearer {doctor_token}"},
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["notes"] == NOTE

        async with TestSessionLocal() as s:
            row = (await s.execute(
                select(Prediction).where(Prediction.id == "pred-NOTE-P1-heart_disease")
            )).scalar_one()
        assert row.notes == NOTE

    async def test_the_note_comes_back_in_history(self, client, doctor_token, db_tables):
        """History is one of the two places the note has to show up."""
        await _seed_patient_with_prediction(patient_id="NOTE-P2")
        await client.post(
            "/api/v4/patients/NOTE-P2/notes",
            json={"disease": "heart_disease", "notes": NOTE},
            headers={"Authorization": f"Bearer {doctor_token}"},
        )
        resp = await client.get(
            "/api/v4/patients/NOTE-P2/predictions",
            headers={"Authorization": f"Bearer {doctor_token}"},
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["items"][0]["notes"] == NOTE

    async def test_the_note_never_becomes_a_feature(self, client, doctor_token, db_tables):
        """
        The whole point of storing it is that it is inert. It must not end up
        in input_features, where it would reach retraining.
        """
        await _seed_patient_with_prediction(patient_id="NOTE-P3")
        await client.post(
            "/api/v4/patients/NOTE-P3/notes",
            json={"disease": "heart_disease", "notes": NOTE},
            headers={"Authorization": f"Bearer {doctor_token}"},
        )
        async with TestSessionLocal() as s:
            row = (await s.execute(
                select(Prediction).where(Prediction.id == "pred-NOTE-P3-heart_disease")
            )).scalar_one()
        assert row.input_features == PATIENT
        assert "notes" not in row.input_features
        assert NOTE not in str(row.input_features)

    async def test_saving_a_note_with_no_screening_says_so(
        self, client, doctor_token, db_tables
    ):
        from backend.db_models.patient import Patient
        async with TestSessionLocal() as s:
            s.add(Patient(id="NOTE-EMPTY", mrn="NOTE-EMPTY", full_name="No Screening"))
            await s.commit()

        resp = await client.post(
            "/api/v4/patients/NOTE-EMPTY/notes",
            json={"disease": "heart_disease", "notes": NOTE},
            headers={"Authorization": f"Bearer {doctor_token}"},
        )
        assert resp.status_code == 404
        assert "NO_PREDICTION_TO_ANNOTATE" in resp.text

    async def test_an_anonymous_visitor_cannot_write_notes(self, client, db_tables):
        await _seed_patient_with_prediction(patient_id="NOTE-P4")
        resp = await client.post(
            "/api/v4/patients/NOTE-P4/notes",
            json={"disease": "heart_disease", "notes": NOTE},
        )
        assert resp.status_code == 401


@pytest.mark.asyncio
class TestReviewAnnotationNotes:
    async def _queued_item(self, item_id="rq-notes-1"):
        from backend.db_models.patient import Patient
        async with TestSessionLocal() as s:
            if (await s.execute(select(Patient).where(Patient.id == "RQ-P1"))).scalar_one_or_none() is None:
                s.add(Patient(id="RQ-P1", mrn="RQ-P1", full_name="Queue Test"))
                await s.flush()
            s.add(Prediction(
                id=f"pred-{item_id}", patient_id="RQ-P1", disease="heart_disease",
                input_features=PATIENT, prediction=1, confidence=0.4,
                diagnosis="Positive",
            ))
            await s.flush()
            s.add(ReviewQueue(
                id=item_id, prediction_id=f"pred-{item_id}",
                uncertainty_score=0.97, status="pending",
            ))
            await s.commit()
        return item_id

    async def test_label_note_is_persisted(self, client, doctor_token, db_tables):
        item_id = await self._queued_item("rq-notes-1")
        resp = await client.post(
            f"/api/v4/review/{item_id}/annotate",
            json={"label": 1, "notes": NOTE},
            headers={"Authorization": f"Bearer {doctor_token}"},
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["notes"] == NOTE

        async with TestSessionLocal() as s:
            row = (await s.execute(
                select(ReviewQueue).where(ReviewQueue.id == item_id)
            )).scalar_one()
        assert row.label == 1
        assert row.notes == NOTE, "the reviewer's note was accepted and discarded (11)"
        assert row.status == "reviewed"

    async def test_a_blank_note_is_stored_as_nothing_not_as_empty_text(
        self, client, doctor_token, db_tables
    ):
        item_id = await self._queued_item("rq-notes-2")
        await client.post(
            f"/api/v4/review/{item_id}/annotate",
            json={"label": 0, "notes": "   "},
            headers={"Authorization": f"Bearer {doctor_token}"},
        )
        async with TestSessionLocal() as s:
            row = (await s.execute(
                select(ReviewQueue).where(ReviewQueue.id == item_id)
            )).scalar_one()
        assert row.notes is None

    async def test_annotating_without_a_note_still_works(
        self, client, doctor_token, db_tables
    ):
        item_id = await self._queued_item("rq-notes-3")
        resp = await client.post(
            f"/api/v4/review/{item_id}/annotate",
            json={"label": 1},
            headers={"Authorization": f"Bearer {doctor_token}"},
        )
        assert resp.status_code == 200
        assert resp.json()["notes"] is None

    async def test_the_queue_listing_carries_the_note_back(
        self, client, doctor_token, db_tables
    ):
        """A reviewed item's reasoning has to be readable afterwards."""
        item_id = await self._queued_item("rq-notes-4")
        resp = await client.get(
            "/api/v4/review/queue",
            headers={"Authorization": f"Bearer {doctor_token}"},
        )
        assert resp.status_code == 200
        listed = {i["id"]: i for i in resp.json()["items"]}
        assert item_id in listed
        # Pending items carry the field (null) so the UI can bind to it.
        assert "notes" in listed[item_id]
        assert "label" in listed[item_id]

    async def test_a_queue_row_is_not_empty(self, client, doctor_token, db_tables):
        """
        The Admin labelling table rendered "— — — —" because it read a nested
        `item.prediction` the endpoint never sends. The flat fields it now
        reads must actually be populated.
        """
        await self._queued_item("rq-notes-5")
        resp = await client.get(
            "/api/v4/review/queue",
            headers={"Authorization": f"Bearer {doctor_token}"},
        )
        row = next(i for i in resp.json()["items"] if i["id"] == "rq-notes-5")
        assert row["disease"] == "heart_disease"
        assert row["model_prediction"] == 1
        assert row["confidence"] == pytest.approx(0.4)
        assert row["uncertainty_score"] == pytest.approx(0.97)
        assert row["features"], "the queue row carries no feature values"
