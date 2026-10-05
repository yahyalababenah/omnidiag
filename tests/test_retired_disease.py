"""
Retired disease modules (gate B3) — one test per route.

The rule (backend/retired_diseases.py): reading an existing record of a retired
disease is allowed; scoring, writing or retraining for it answers 410.

BRFSS diabetes is the retired module here. Its config is archived, so the real
router no longer registers it; the records below stand in for the rows it left
in the database while it was live.
"""

import os
import uuid

import pytest
import pytest_asyncio
from sqlalchemy import func, select

from backend.router import OmniDiagRouter as _RealOmniDiagRouter
from backend.retired_diseases import RETIRED_DISEASES
from tests.conftest import TestSessionLocal

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RETIRED = "diabetes"

BRFSS_ROW = {"HighBP": 1, "HighChol": 0, "CholCheck": 1, "BMI": 29, "Smoker": 0, "Stroke": 0,
             "HeartDiseaseorAttack": 0, "PhysActivity": 1, "Fruits": 1, "Veggies": 1,
             "HvyAlcoholConsump": 0, "AnyHealthcare": 1, "NoDocbcCost": 0, "GenHlth": 3,
             "MentHlth": 0, "PhysHlth": 2, "DiffWalk": 0, "Sex": 1, "Age": 6, "Education": 5,
             "Income": 6}
HEART_ROW = {"Age": 54, "Sex": "M", "ChestPainType": "ATA", "RestingBP": 140, "Cholesterol": 289,
             "FastingBS": 0, "RestingECG": "Normal", "MaxHR": 122, "ExerciseAngina": "N",
             "Oldpeak": 0, "ST_Slope": "Flat"}


def _assert_retired(resp):
    assert resp.status_code == 410, resp.text
    body = resp.json()
    assert body["code"] == "DISEASE_RETIRED"
    assert body["replaced_by"] == "diabetes_nhanes"
    assert body["reason"]


@pytest.fixture(scope="module")
def live_app(app):
    """The conftest app with the REAL router: the 410s must come from the real
    registration path (no config -> not registered -> retired), not a mock."""
    import backend.main as main_module

    previous_router, limiter = main_module.router, main_module.app.state.limiter
    previous_enabled = limiter.enabled
    main_module.router = _RealOmniDiagRouter(configs_dir=os.path.join(_ROOT, "configs"))
    limiter.enabled = False
    yield main_module.app
    main_module.router = previous_router
    limiter.enabled = previous_enabled


@pytest_asyncio.fixture(scope="module")
async def records(db_tables):
    """A patient whose only screening is a retired-module row, with a visit and a
    pending review item; and a heart patient with a pending item for contrast."""
    from backend.db_models.patient import Patient
    from backend.db_models.patient_visit import PatientVisit
    from backend.db_models.prediction import Prediction
    from backend.db_models.review_queue import ReviewQueue

    ids = {k: str(uuid.uuid4()) for k in ("pt", "pred", "rq", "hpt", "hpred", "hrq")}
    async with TestSessionLocal() as s:
        s.add_all([
            Patient(id=ids["pt"], mrn=f"RET-{ids['pt'][:8]}", full_name="Retired Module Patient"),
            Patient(id=ids["hpt"], mrn=f"HRT-{ids['hpt'][:8]}", full_name="Heart Patient"),
        ])
        await s.flush()
        s.add_all([
            Prediction(id=ids["pred"], patient_id=ids["pt"], disease=RETIRED,
                       input_features=BRFSS_ROW, prediction=1, confidence=0.41,
                       diagnosis="Positive", probability_scale="corrected",
                       shap_chart_data=[{"feature": "BMI", "shap_value": 0.12}]),
            Prediction(id=ids["hpred"], patient_id=ids["hpt"], disease="heart_disease",
                       input_features=HEART_ROW, prediction=1, confidence=0.55,
                       diagnosis="Positive", probability_scale="raw"),
        ])
        await s.flush()
        s.add_all([
            PatientVisit(patient_id=ids["pt"], disease=RETIRED, features=BRFSS_ROW,
                         risk_score=0.41, prediction=1),
            ReviewQueue(id=ids["rq"], prediction_id=ids["pred"], uncertainty_score=0.97,
                        status="pending"),
            ReviewQueue(id=ids["hrq"], prediction_id=ids["hpred"], uncertainty_score=1.0,
                        status="pending"),
        ])
        await s.commit()
    return ids


@pytest_asyncio.fixture
async def c(live_app, db_tables):
    from httpx import ASGITransport, AsyncClient

    async with AsyncClient(transport=ASGITransport(app=live_app), base_url="http://test") as cl:
        yield cl


def _auth(token):
    return {"Authorization": f"Bearer {token}"}


# ── The registry ─────────────────────────────────────────────────────────────

def test_no_retired_disease_has_a_config():
    for disease in RETIRED_DISEASES:
        for ext in (".yaml", ".yml"):
            assert not os.path.exists(os.path.join(_ROOT, "configs", disease + ext)), disease


async def test_diseases_lists_exactly_the_live_modules(c):
    body = (await c.get("/api/v4/diseases")).json()
    assert {d["name"] for d in body["diseases"]} == {"heart_disease", "diabetes_nhanes"}
    assert set(body["all_registered"]) == {"heart_disease", "diabetes_nhanes"}


# ── Scoring a retired disease: 410 ───────────────────────────────────────────

@pytest.mark.parametrize("route", ["predict", "explain", "counterfactuals"])
async def test_scoring_routes_answer_410(c, doctor_token, route):
    _assert_retired(await c.post(f"/api/v4/{RETIRED}/{route}", json=BRFSS_ROW,
                                 headers=_auth(doctor_token)))


async def test_batch_answers_410(c, doctor_token):
    csv = ",".join(BRFSS_ROW) + "\n" + ",".join(str(v) for v in BRFSS_ROW.values()) + "\n"
    _assert_retired(await c.post(f"/api/v4/{RETIRED}/batch", headers=_auth(doctor_token),
                                 files={"file": ("b.csv", csv, "text/csv")}))


async def test_schema_answers_410_even_when_cached(c, monkeypatch):
    # A schema cached while the module was live must not be served afterwards.
    # The test client does not run startup, so the cache is switched on here;
    # otherwise cache_set is a no-op and this would pass without testing anything.
    from fastapi_cache import FastAPICache
    from fastapi_cache.backends.inmemory import InMemoryBackend

    import backend.cache as cache_module

    mem = InMemoryBackend()
    monkeypatch.setattr(cache_module, "_backend", mem)
    FastAPICache.init(mem, prefix="retired-test")
    key = cache_module.schema_cache_key(RETIRED)
    await cache_module.cache_set(key, {"stale": True}, 3600)
    assert await cache_module.cache_get(key) == {"stale": True}
    _assert_retired(await c.get(f"/api/v4/{RETIRED}/schema"))


async def test_parse_notes_answers_410_for_a_retired_disease(c):
    _assert_retired(await c.post("/api/v4/parse-notes",
                                 json={"note": "52 year old woman, BMI 31", "disease": RETIRED}))
    assert (await c.post("/api/v4/parse-notes", json={"note": "52 year old woman"})).status_code == 200


# ── Retraining: 410, and W-08 proven closed ─────────────────────────────────

async def test_retrain_answers_410(c, admin_token):
    _assert_retired(await c.post("/admin/retrain", headers=_auth(admin_token),
                                 json={"disease": RETIRED, "min_samples": 1}))


async def test_retrain_never_reaches_the_legacy_xgboost_writer(monkeypatch):
    """W-08: with no config the pipeline used to fall through to retrain_xgb, which
    overwrites models/diabetes/omni_diag_xgb_optimized.pkl in an image that still
    downloads it. It must stop before reading a single sample."""
    from fastapi import HTTPException

    from backend.active_learning import retrain

    def boom(*a, **k):
        raise AssertionError("reached past the retired-disease guard")

    monkeypatch.setattr(retrain, "get_annotated_samples", boom)
    monkeypatch.setattr(retrain, "retrain_candidate", boom)
    # retrain_xgb itself was removed in gate B4; nothing that writes is reachable.
    assert not hasattr(retrain, "retrain_xgb")
    with pytest.raises(HTTPException) as err:
        await retrain.run_retrain_pipeline(None, RETIRED, 1)
    assert err.value.status_code == 410


# ── Drift: no live model to monitor ─────────────────────────────────────────

@pytest.mark.parametrize("method,path", [("get", "status"), ("post", "run"), ("get", "report")])
async def test_drift_answers_410(c, admin_token, method, path):
    resp = await getattr(c, method)(f"/api/v4/admin/drift/{RETIRED}/{path}",
                                    headers=_auth(admin_token))
    _assert_retired(resp)


# ── Reading existing records: allowed ───────────────────────────────────────

async def test_history_lists_the_retired_rows(c, doctor_token, records):
    for params in ({}, {"disease": RETIRED}):
        resp = await c.get(f"/api/v4/patients/{records['pt']}/predictions",
                           params=params, headers=_auth(doctor_token))
        assert resp.status_code == 200, resp.text
        assert [i["disease"] for i in resp.json()["items"]] == [RETIRED]


async def test_export_bundle_includes_the_retired_rows(c, doctor_token, records):
    resp = await c.get(f"/api/v4/patients/{records['pt']}/export", headers=_auth(doctor_token))
    assert resp.status_code == 200, resp.text
    assert [p["disease"] for p in resp.json()["predictions"]] == [RETIRED]


async def test_visits_of_a_retired_disease_stay_readable(c, doctor_token, records):
    resp = await c.get(f"/api/v4/patients/{records['pt']}/visits",
                       params={"disease": RETIRED}, headers=_auth(doctor_token))
    assert resp.status_code == 200, resp.text
    assert [v["disease"] for v in resp.json()["visits"]] == [RETIRED]


async def test_a_report_renders_from_a_stored_retired_row(c, records):
    resp = await c.post("/api/v4/generate-report", json={
        "disease": RETIRED, "label": "Positive", "probability_corrected": 0.41,
        "shap_values": [{"feature": "BMI", "shap_value": 0.12}], "features": BRFSS_ROW,
    })
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["source"] == "archived" and body["archived"] is True
    assert body["risk_band"] is None
    assert body["report"].startswith("**ARCHIVED MODULE — replaced by NHANES dysglycaemia module**")
    assert "41.0%" in body["report"]


async def test_a_retired_row_never_reaches_the_llm(c, monkeypatch, records):
    """Even with an API key configured, a retired module's row is rendered by
    the rule-based archived template; no model is asked about it."""
    import backend.llm.report_generator as rg
    import backend.main as main_module

    async def boom(**kwargs):
        raise AssertionError("a retired row reached the LLM path")

    monkeypatch.setattr(rg, "_get_api_key", lambda: "test-key")
    monkeypatch.setattr(main_module, "_generate_report", boom)
    resp = await c.post("/api/v4/generate-report", json={
        "disease": RETIRED, "label": "Negative", "probability_corrected": 0.08,
        "shap_values": [], "features": BRFSS_ROW,
    })
    assert resp.status_code == 200, resp.text
    assert resp.json()["source"] == "archived"


async def test_export_marks_a_retired_row_as_archived(c, doctor_token, records):
    rows = (await c.get(f"/api/v4/patients/{records['pt']}/export",
                        headers=_auth(doctor_token))).json()["predictions"]
    assert [r["disease"] for r in rows] == [RETIRED]
    note = rows[0]["archived_note"]
    assert note.startswith("Archived module, replaced by the NHANES dysglycaemia module")
    heart = (await c.get(f"/api/v4/patients/{records['hpt']}/export",
                         headers=_auth(doctor_token))).json()["predictions"]
    assert heart and all(r["archived_note"] is None for r in heart)


# ── New writes for a retired disease: 410 ────────────────────────────────────

async def _count(model, **where):
    async with TestSessionLocal() as s:
        q = select(func.count()).select_from(model)
        for k, v in where.items():
            q = q.where(getattr(model, k) == v)
        return (await s.execute(q)).scalar_one()


async def test_a_new_visit_is_refused(c, doctor_token, records):
    from backend.db_models.patient_visit import PatientVisit

    before = await _count(PatientVisit, disease=RETIRED)
    _assert_retired(await c.post(
        f"/api/v4/patients/{records['pt']}/visits", headers=_auth(doctor_token),
        json={"disease": RETIRED, "features": BRFSS_ROW, "risk_score": 0.3, "prediction": 1}))
    assert await _count(PatientVisit, disease=RETIRED) == before


@pytest.mark.parametrize("body", [{"disease": RETIRED}, {}], ids=["named", "latest-row"])
async def test_a_note_on_a_retired_row_is_refused(c, doctor_token, records, body):
    # "latest-row": no disease given, so the note would land on the patient's
    # latest screening -- which is the retired one.
    _assert_retired(await c.post(f"/api/v4/patients/{records['pt']}/notes",
                                 headers=_auth(doctor_token), json={"notes": "x", **body}))


# ── The review queue ────────────────────────────────────────────────────────

async def test_queue_lists_retired_items_flagged(c, doctor_token, records):
    resp = await c.get("/api/v4/review/queue", params={"limit": 100}, headers=_auth(doctor_token))
    assert resp.status_code == 200, resp.text
    flag = {i["id"]: i["retired"] for i in resp.json()["items"]}
    assert flag[records["rq"]] is True
    assert flag[records["hrq"]] is False


@pytest.mark.parametrize("action,body", [("annotate", {"label": 1}), ("skip", None)])
async def test_labelling_a_retired_item_is_refused(c, doctor_token, records, action, body):
    from backend.db_models.review_queue import ReviewQueue

    kwargs = {"json": body} if body else {}
    _assert_retired(await c.post(f"/api/v4/review/{records['rq']}/{action}",
                                 headers=_auth(doctor_token), **kwargs))
    async with TestSessionLocal() as s:
        rq = (await s.execute(select(ReviewQueue).where(ReviewQueue.id == records["rq"]))).scalar_one()
    assert rq.status == "pending" and rq.label is None


async def test_stats_count_retired_items_apart(c, doctor_token, records):
    from backend.db_models.review_queue import ReviewQueue

    stats = (await c.get("/api/v4/review/stats", headers=_auth(doctor_token))).json()
    all_pending = await _count(ReviewQueue, status="pending")
    assert stats["retired_pending"] >= 1
    assert stats["pending"] + stats["retired_pending"] == all_pending
    assert stats["total"] == stats["pending"] + stats["retired_pending"] + stats["reviewed"] + stats["skipped"]


# ── Seeding ─────────────────────────────────────────────────────────────────

def test_demo_seed_skips_retired_modules():
    from backend import demo_seed

    # Gate B6 removed the retired module's demo patients from mockPatients.js and
    # its mirror together; active_demo_patients() still filters, for any later one.
    assert RETIRED not in demo_seed.load_demo_patients()
    assert RETIRED not in demo_seed.active_demo_patients()


async def test_history_marks_a_retired_row_as_archived(c, doctor_token, records):
    items = (await c.get(f"/api/v4/patients/{records['pt']}/predictions",
                         headers=_auth(doctor_token))).json()["items"]
    assert [i["disease"] for i in items] == [RETIRED]
    assert items[0]["archived_note"].startswith("Archived module, replaced by the NHANES")
    heart = (await c.get(f"/api/v4/patients/{records['hpt']}/predictions",
                         headers=_auth(doctor_token))).json()["items"]
    assert heart and all(i["archived_note"] is None for i in heart)
