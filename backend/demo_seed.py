"""
OmniDiag — Demo patient + history seeding (X-5)
================================================

Why this exists
---------------
The live database is SQLite inside the Space container, so it is wiped by
every restart and rebuild. Two things vanished with it:

  * the demo patients themselves — nothing in the product ever created a
    Patient row, so `GET /api/v4/patients/P-001/predictions` answered 404
    "Patient 'P-001' not found" and History could not work at all; and
  * any prediction history, so even once a patient existed the timeline was
    empty until someone manually re-ran scans.

This module recreates both on startup, idempotently, so History shows a real
timeline after any restart.

What is real and what is synthetic
----------------------------------
Every risk score written here is REAL model output: the seeder calls the same
router the API calls, on the feature values it stores alongside the score. No
probability is invented.

What IS synthetic is the earlier *visits*. A demo patient has one set of
current values, so a trajectory has to come from somewhere. Each earlier
visit takes the patient's current data and moves ONE documented modifiable
field (see HISTORY_LEVERS), then asks the model what that patient would have
scored. The most recent entry is the patient's current data unchanged, so the
last point on the timeline always equals what the EMR screen shows.

Rows written here are marked in `diagnosis` history via the seeded prediction
being linked to a patient whose MRN starts with the demo prefix, and the
whole set is recognisable by its backdated created_at. Nothing here is
presented as a real clinical record.

Source of truth
---------------
backend/demo_patients.json is generated from frontend/src/mockPatients.js —
the same file the UI renders. tests/test_demo_seed.py re-extracts that file
and fails if the two drift apart.
"""

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import select

from backend.retired_diseases import is_retired
from backend.active_learning.sampler import (
    DEFAULT_DECISION_THRESHOLD,
    prediction_entropy,
    should_queue_for_review,
)
from backend.database import AsyncSessionLocal
from backend.db_models.patient import Patient
from backend.db_models.patient_visit import PatientVisit
from backend.db_models.prediction import Prediction
from backend.db_models.review_queue import ReviewQueue
from backend.probability_scale import scale_of_result

log = logging.getLogger("omnidiag.demo_seed")

_DEMO_PATIENTS_PATH = os.path.join(os.path.dirname(__file__), "demo_patients.json")

# How many timeline points each demo patient gets, including today's.
VISITS_PER_PATIENT = 3

# Spacing between seeded visits.
VISIT_INTERVAL_DAYS = 90

# The one field each disease's synthetic history moves, and by how much per
# step back in time. Chosen to be modifiable and clinically legible: a
# patient whose BMI or cholesterol was higher at the last visit reads as a
# plausible trajectory rather than noise. Values are clamped to the schema's
# own range by the validators these feed into.
HISTORY_LEVERS: Dict[str, Dict[str, Any]] = {
    # Total cholesterol, mg/dL — higher at older visits.
    "heart_disease": {"field": "Cholesterol", "step": 18, "minimum": 100, "maximum": 400},
    # BMI — higher at older visits.
    "diabetes": {"field": "BMI", "step": 2, "minimum": 18, "maximum": 60},
    # NHANES dysglycaemia module: BMI in kg/m2, higher at older visits. It moves
    # only BMXBMI, not ADIPOSITY_BAND, so the earlier visits are synthetic history
    # in the same sense as the other two modules'.
    "diabetes_nhanes": {"field": "BMXBMI", "step": 2, "minimum": 15, "maximum": 60},
}


def load_demo_patients() -> Dict[str, List[Dict[str, Any]]]:
    """{disease: [patient, ...]} as exported from the frontend's mock data."""
    with open(_DEMO_PATIENTS_PATH) as f:
        return json.load(f)


def active_demo_patients() -> Dict[str, List[Dict[str, Any]]]:
    """The demo patients of modules that are still live.

    A retired module (backend/retired_diseases.py) cannot score, so seeding it
    would only log failures and leave empty patient rows. Filtered here rather
    than in load_demo_patients(), which mirrors the frontend's file exactly and
    is tested against it.
    """
    return {d: p for d, p in load_demo_patients().items() if not is_retired(d)}


def historical_features(
    current: Dict[str, Any], disease: str, steps_back: int
) -> Dict[str, Any]:
    """
    The patient's feature values *steps_back* visits ago.

    steps_back == 0 returns the current values untouched, so today's entry on
    the timeline is exactly the record the EMR screen shows.
    """
    if steps_back == 0:
        return dict(current)

    lever = HISTORY_LEVERS.get(disease)
    if not lever:
        return dict(current)

    field = lever["field"]
    value = current.get(field)
    if not isinstance(value, (int, float)):
        return dict(current)

    shifted = value + lever["step"] * steps_back
    shifted = max(lever["minimum"], min(lever["maximum"], shifted))
    return {**current, field: type(value)(shifted)}


async def _already_seeded(db) -> bool:
    """True when a previous run already created the demo patients."""
    # The same list seed_demo_history() writes, or a retired module's patients
    # would never exist and every startup would believe seeding was unfinished.
    demo_ids = [
        patient["id"]
        for patients in active_demo_patients().values()
        for patient in patients
    ]
    if not demo_ids:
        return True
    existing = (
        await db.execute(select(Patient.id).where(Patient.id.in_(demo_ids)))
    ).scalars().all()
    return len(existing) == len(demo_ids)


async def seed_demo_history(router, *, force: bool = False) -> int:
    """
    Create the demo patients and their prediction history if missing.

    Idempotent: a run that finds every demo patient already present does
    nothing and returns 0. Safe to call on every startup.

    Returns the number of prediction rows written.

    *router* is the live OmniDiagRouter — the seeded probabilities are its
    own output, not stored constants.
    """
    demo = active_demo_patients()

    async with AsyncSessionLocal() as db:
        if not force and await _already_seeded(db):
            log.info("Demo history already present — nothing to seed")
            return 0

        written = 0
        queued = 0
        now = datetime.now(timezone.utc)

        for disease, patients in demo.items():
            for entry in patients:
                patient_id = entry["id"]

                patient = (
                    await db.execute(select(Patient).where(Patient.id == patient_id))
                ).scalar_one_or_none()
                if patient is None:
                    patient = Patient(
                        id=patient_id,
                        # The demo id doubles as the MRN: it is what the UI
                        # shows on the patient card, so History, the PDF and
                        # the screen all name the patient the same way.
                        mrn=patient_id,
                        full_name=entry["name"],
                        gender={"M": "male", "F": "female"}.get(entry.get("sex")),
                    )
                    db.add(patient)
                    await db.flush()

                # Oldest first, so created_at ordering matches the story.
                for steps_back in range(VISITS_PER_PATIENT - 1, -1, -1):
                    features = historical_features(entry["data"], disease, steps_back)
                    visit_date = now - timedelta(days=VISIT_INTERVAL_DAYS * steps_back)

                    try:
                        result = await asyncio.to_thread(router.predict, disease, features)
                    except Exception as exc:
                        log.warning(
                            "Demo seed: %s/%s (t-%d) failed to score — %s",
                            disease, patient_id, steps_back, exc,
                        )
                        continue

                    confidence = float(result.get("confidence", 0.0))
                    record = Prediction(
                        id=str(uuid.uuid4()),
                        patient_id=patient.id,
                        disease=disease,
                        input_features=features,
                        prediction=int(result.get("prediction", 0)),
                        confidence=confidence,
                        probability_scale=scale_of_result(result).value,
                        diagnosis=result.get("diagnosis"),
                        # The conformal decision and its interval, for a module
                        # that reports them (Gate 8.4). Without these the demo's
                        # own history was the only place in the system that still
                        # stored a heart prediction with no decision attached.
                        decision=result.get("decision"),
                        probability_lower=result.get("probability_lower"),
                        probability_upper=result.get("probability_upper"),
                        created_at=visit_date,
                    )
                    db.add(record)
                    await db.flush()

                    # Queue the CURRENT screening for review when it is
                    # genuinely uncertain, by the same rule /predict applies
                    # to a live one. Without this the Admin annotation queue
                    # is empty after every restart and there is nothing to
                    # demonstrate the human-in-the-loop step with — the
                    # feature exists but cannot be shown. Only today's visit
                    # is considered; backdated ones are history, not work.
                    #
                    # "The same rule as /predict" has to be kept true, and until
                    # Gate 8.8 it was not: this branch read
                    # `inference_threshold` with a 0.5 default, so every heart
                    # row was judged by entropy around a threshold the model does
                    # not have AND recorded 0.5 as the threshold it used. The demo
                    # queue -- the one thing shown to demonstrate
                    # human-in-the-loop -- was the last place still doing it.
                    raw_threshold = result.get("inference_threshold")
                    decision_threshold = (
                        float(raw_threshold) if raw_threshold is not None else None
                    )
                    if result.get("decision") is not None:
                        is_uncertain = result.get("decision") == "uncertain"
                        uncertainty_score = 1.0 if is_uncertain else 0.0
                    else:
                        is_uncertain = should_queue_for_review(
                            confidence,
                            decision_threshold=decision_threshold or DEFAULT_DECISION_THRESHOLD,
                        )
                        uncertainty_score = prediction_entropy(
                            confidence,
                            decision_threshold=decision_threshold or DEFAULT_DECISION_THRESHOLD,
                        )
                    if steps_back == 0 and is_uncertain:
                        db.add(ReviewQueue(
                            id=str(uuid.uuid4()),
                            prediction_id=record.id,
                            uncertainty_score=uncertainty_score,
                            uncertainty_scale=scale_of_result(result).value,
                            decision_threshold=decision_threshold,
                            created_at=visit_date,
                        ))
                        queued += 1
                    db.add(PatientVisit(
                        id=str(uuid.uuid4()),
                        patient_id=patient.id,
                        disease=disease,
                        visit_date=visit_date,
                        features=features,
                        risk_score=float(result.get("confidence", 0.0)),
                        prediction=int(result.get("prediction", 0)),
                        notes=(
                            "Seeded demo visit — current record"
                            if steps_back == 0
                            else f"Seeded demo visit — synthetic history, "
                                 f"{HISTORY_LEVERS.get(disease, {}).get('field', 'n/a')} "
                                 f"as recorded {steps_back * VISIT_INTERVAL_DAYS} days ago"
                        ),
                    ))
                    written += 1

        await db.commit()
        log.info(
            "Demo history seeded — %d prediction rows across %d patients, "
            "%d queued for review",
            written, sum(len(p) for p in demo.values()), queued,
        )
        return written


async def seed_demo_history_in_background(router) -> Optional[asyncio.Task]:
    """
    Schedule seeding without delaying startup.

    Scoring 21 rows takes a few seconds, and the first call also loads the
    models — worth doing, but not worth making the Space unreachable while it
    happens. The task runs as soon as the event loop starts serving, so
    History is populated a few seconds after the first request can be
    answered, and the models are warm by the time a judge clicks anything.
    """
    async def _run():
        try:
            await seed_demo_history(router)
        except Exception as exc:
            # Seeding is a convenience, never a reason to fail startup.
            log.warning("Demo history seeding failed — History will be empty: %s", exc)

    return asyncio.create_task(_run())
