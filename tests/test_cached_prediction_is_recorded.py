"""
X-7 — a cached /predict must still be recorded.

The cache used to short-circuit the whole endpoint: on a hit it returned the
payload before the predictions row was written, before the Prometheus counter
moved, and before the uncertainty check that feeds the review queue. So the
second clinician to score the same patient left no trace at all — the
prediction count stayed put and an uncertain case that should have gone for
human review never did.

The cache is supposed to skip the model call, not the recording of what was
served.
"""

import pytest
from sqlalchemy import func, select

from backend.db_models.prediction import Prediction
from backend.db_models.review_queue import ReviewQueue
from tests.conftest import TestSessionLocal

PATIENT = {
    "Age": 61, "Sex": "F", "ChestPainType": "ATA", "RestingBP": 142,
    "Cholesterol": 268, "FastingBS": 1, "RestingECG": "ST",
    "MaxHR": 122, "ExerciseAngina": "Y", "Oldpeak": 2.1, "ST_Slope": "Flat",
}


def patient_at_age(age: int) -> dict:
    """A distinct payload per test, so tests never share a cache entry."""
    return {**PATIENT, "Age": age}


async def count_rows(model) -> int:
    async with TestSessionLocal() as session:
        return (await session.execute(select(func.count()).select_from(model))).scalar_one()


async def predict(client, token, payload):
    return await client.post(
        "/api/v4/heart_disease/predict",
        json=payload,
        headers={"Authorization": f"Bearer {token}"},
    )


@pytest.mark.asyncio
async def test_second_identical_prediction_is_a_cache_hit(client, doctor_token, db_tables):
    payload = patient_at_age(61)
    first = await predict(client, doctor_token, payload)
    second = await predict(client, doctor_token, payload)

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.headers["Cache-Hit"] == "false"
    assert second.headers["Cache-Hit"] == "true", (
        "the cache is not being used at all — this test would then prove nothing"
    )
    # Skipping the model call must not change the answer.
    assert first.json() == second.json()


@pytest.mark.asyncio
async def test_cache_hit_still_writes_a_prediction_row(client, doctor_token, db_tables):
    payload = patient_at_age(62)

    before = await count_rows(Prediction)
    await predict(client, doctor_token, payload)
    after_miss = await count_rows(Prediction)
    hit = await predict(client, doctor_token, payload)
    after_hit = await count_rows(Prediction)

    assert hit.headers["Cache-Hit"] == "true"
    assert after_miss == before + 1
    assert after_hit == after_miss + 1, (
        "a cached prediction was served without being recorded (X-7)"
    )


@pytest.mark.asyncio
async def test_cache_hit_still_queues_an_uncertain_case_for_review(
    app, client, doctor_token, db_tables
):
    """
    The consequence that matters clinically: an uncertain prediction served
    from cache was never offered to a human.
    """
    payload = patient_at_age(63)
    router = _router()
    original = router.predict.return_value
    # Sat on the decision boundary, so should_queue_for_review() fires.
    router.predict.return_value = {
        "prediction": 1, "confidence": 0.5, "diagnosis": "Positive",
    }
    try:
        before = await count_rows(ReviewQueue)
        miss = await predict(client, doctor_token, payload)
        after_miss = await count_rows(ReviewQueue)
        hit = await predict(client, doctor_token, payload)
        after_hit = await count_rows(ReviewQueue)
    finally:
        router.predict.return_value = original

    assert miss.headers["Cache-Hit"] == "false"
    assert hit.headers["Cache-Hit"] == "true"
    assert after_miss == before + 1, "the uncomputed case was not queued at all"
    assert after_hit == after_miss + 1, (
        "an uncertain prediction served from cache never reached the review "
        "queue (X-7)"
    )


@pytest.mark.asyncio
async def test_anonymous_cache_hit_still_writes_nothing(client, db_tables):
    """The fix must not start persisting predictions for signed-out visitors."""
    payload = patient_at_age(64)

    before = await count_rows(Prediction)
    await client.post("/api/v4/heart_disease/predict", json=payload)
    hit = await client.post("/api/v4/heart_disease/predict", json=payload)
    after = await count_rows(Prediction)

    assert hit.headers["Cache-Hit"] == "true"
    assert after == before, "an anonymous prediction was persisted"


def _router():
    import backend.main as main_module
    return main_module.router
