"""
HITL from the Clinical EMR view: a doctor (not an admin) gets a review_id on an
uncertain prediction and labels it through the endpoint ReviewLabelBox calls.
"""
import pytest
from sqlalchemy import select

from backend.db_models.review_queue import ReviewQueue
from tests.conftest import TestSessionLocal
from tests.test_cached_prediction_is_recorded import _router, patient_at_age, predict


def _auth(t):
    return {"Authorization": f"Bearer {t}"}


@pytest.mark.asyncio
async def test_doctor_labels_uncertain_case_without_admin(client, doctor_token, viewer_token, db_tables):
    router = _router()
    original = router.predict.return_value
    router.predict.return_value = {"prediction": 1, "confidence": 0.5, "diagnosis": "Positive"}
    try:
        r = await predict(client, doctor_token, patient_at_age(71))
        confident = None
        router.predict.return_value = {"prediction": 1, "confidence": 0.99, "diagnosis": "Positive"}
        confident = await predict(client, doctor_token, patient_at_age(72))
    finally:
        router.predict.return_value = original

    rid = r.json().get("review_id")
    assert rid, "uncertain case returned no review_id -> UI would show no buttons"
    assert "review_id" not in confident.json()

    # a viewer must not be able to label
    denied = await client.post(f"/api/v4/review/{rid}/annotate", json={"label": 1}, headers=_auth(viewer_token))
    assert denied.status_code in (401, 403)

    ok = await client.post(f"/api/v4/review/{rid}/annotate",
                           json={"label": 1, "notes": "confirmed by cath"}, headers=_auth(doctor_token))
    assert ok.status_code == 200, ok.text

    async with TestSessionLocal() as s:
        row = (await s.execute(select(ReviewQueue).where(ReviewQueue.id == rid))).scalar_one()
    assert (row.status, row.label, row.notes) == ("reviewed", 1, "confirmed by cath")
    assert row.reviewer_id

    again = await client.post(f"/api/v4/review/{rid}/annotate", json={"label": 0}, headers=_auth(doctor_token))
    assert again.status_code == 409


@pytest.mark.asyncio
async def test_admin_can_list_reviewed_items(client, doctor_token, admin_token, db_tables):
    router = _router()
    original = router.predict.return_value
    router.predict.return_value = {"prediction": 1, "confidence": 0.5, "diagnosis": "Positive"}
    try:
        r = await predict(client, doctor_token, patient_at_age(73))
    finally:
        router.predict.return_value = original
    rid = r.json()["review_id"]
    await client.post(f"/api/v4/review/{rid}/annotate", json={"label": 0, "notes": "normal cath"}, headers=_auth(doctor_token))

    pending = await client.get("/api/v4/review/queue?status=pending&limit=100", headers=_auth(admin_token))
    assert rid not in [i["id"] for i in pending.json()["items"]]
    done = await client.get("/api/v4/review/queue?status=reviewed&limit=100", headers=_auth(admin_token))
    row = next(i for i in done.json()["items"] if i["id"] == rid)
    assert (row["label"], row["notes"], row["status"]) == (0, "normal cath", "reviewed")
    assert row["reviewer"] and row["reviewed_at"]
    bad = await client.get("/api/v4/review/queue?status=bogus", headers=_auth(admin_token))
    assert bad.status_code == 422


@pytest.mark.asyncio
async def test_retrain_route_keeps_candidate_fields_and_is_super_admin_only(client, admin_token, doctor_token, db_tables, monkeypatch):
    fake = {"status": "success", "disease": "heart_disease", "samples_used": 7,
            "outcome": "candidate_built_not_promoted", "mlflow_run_id": "abc123",
            "rows_total": 927, "rows_rejected": [{"index": 1, "reason": "missing Age"}],
            "moved_out_of_uncertain": 3, "moved_into_uncertain": 1,
            "decisions_changed_total": 4, "limit": "not validated"}

    async def _fake(db, disease, min_samples):
        return fake
    monkeypatch.setattr("backend.active_learning.retrain.run_retrain_pipeline", _fake)

    body = {"disease": "heart_disease", "min_samples": 5}
    ok = await client.post("/admin/retrain", json=body, headers=_auth(admin_token))
    assert ok.status_code == 200, ok.text
    got = ok.json()
    assert got["mlflow_run_id"] == "abc123" and got["decisions_changed_total"] == 4
    assert got["outcome"] == "candidate_built_not_promoted" and got["rows_total"] == 927

    denied = await client.post("/admin/retrain", json=body, headers=_auth(doctor_token))
    assert denied.status_code == 403
