"""Cross-layer smoke check for heart_disease and diabetes_nhanes against the real app.

Run from a repo root with DATABASE_URL pointing at a throwaway sqlite file.
In-memory session DB, super_admin user, no network (LLM report forced to rule-based),
retrain called with an unreachable min_samples so it dispatches but never trains.
Prints one line per check plus a digest of the model outputs, for diffing two trees.
"""
import asyncio, hashlib, json, os, sys, uuid

ROOT = os.getcwd()
sys.path.insert(0, ROOT)


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:12]


async def main():
    from fastapi_cache import FastAPICache
    from fastapi_cache.backends.inmemory import InMemoryBackend
    from httpx import ASGITransport, AsyncClient
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
    from sqlalchemy.pool import StaticPool

    import backend.main as main_module
    from backend.database import Base, get_db
    from backend.auth.hashing import hash_password
    from backend.db_models.user import User, user_roles
    from backend.db_models.role import Role
    import backend.db_models  # noqa: F401
    import backend.cache as cache_module

    main_module._generate_report = None

    engine = create_async_engine("sqlite+aiosqlite:///:memory:",
                                 connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async def _get_db():
        async with Session() as s:
            yield s

    app = main_module.app
    app.dependency_overrides[get_db] = _get_db
    app.state.limiter.enabled = False
    mem = InMemoryBackend()
    cache_module._backend = mem
    FastAPICache.init(mem, prefix="xcheck")

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    async with Session() as s:
        roles = [Role(name=n, description=n) for n in ("super_admin", "admin", "doctor")]
        s.add_all(roles)
        await s.flush()
        user = User(id=str(uuid.uuid4()), email="x@test.com", hashed_password=hash_password("Xcheck1234"),
                    full_name="X", is_active=True)
        s.add(user)
        await s.flush()
        for r in roles:
            await s.execute(user_roles.insert().values(user_id=user.id, role_id=r.id))
        await s.commit()

    demos = json.load(open(os.path.join(ROOT, "backend/demo_patients.json")))
    failures, outputs = [], {}

    def check(name, resp, ok=(200,)):
        good = resp.status_code in ok
        print(f"{'OK  ' if good else 'FAIL'} {resp.status_code} {name}")
        if not good:
            failures.append(name)
            print("     ", resp.text[:300])
        return resp

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t", timeout=600) as c:
        tok = (await c.post("/auth/login", json={"email": "x@test.com", "password": "Xcheck1234"})).json()["access_token"]
        auth = {"Authorization": f"Bearer {tok}"}
        diseases = check("GET /diseases", await c.get("/api/v4/diseases")).json()
        print("      registered:", sorted(d["name"] for d in diseases["diseases"]))

        for disease in ("heart_disease", "diabetes_nhanes"):
            for p in demos[disease]:
                tag = f"{disease}/{p['id']}"
                payload = p["data"]
                pr = check(f"predict  {tag}", await c.post(f"/api/v4/{disease}/predict", json=payload, headers=auth,
                                                          params={"patient_id": "00000000-0000-0000-0000-000000000000"}))
                ex = check(f"explain  {tag}", await c.post(f"/api/v4/{disease}/explain", json=payload))
                cf = check(f"cf       {tag}", await c.post(f"/api/v4/{disease}/counterfactuals", json=payload))
                pj, ej = pr.json(), ex.json()
                shap = ej.get("shap_values") or ej.get("contributions") or []
                rep = check(f"report   {tag}", await c.post("/api/v4/generate-report", json={
                    "disease": disease, "label": str(pj.get("label", pj.get("prediction", ""))),
                    "probability_corrected": pj.get("confidence"),
                    "decision": pj.get("decision"), "probability_lower": pj.get("probability_lower"),
                    "probability_upper": pj.get("probability_upper"),
                    "shap_values": shap if isinstance(shap, list) else [], "features": payload}))
                drop = {"review_id", "prediction_id", "timestamp", "request_id", "latency_ms", "cached", "created_at"}
                cfj = cf.json()
                scen = cfj.get("counterfactuals") if isinstance(cfj, dict) else None
                outputs[tag] = {
                    "predict": {k: v for k, v in pj.items() if k not in drop},
                    "explain": ej,
                    "cf_shape": {"status": cfj.get("status") if isinstance(cfj, dict) else None,
                                 "n": len(scen) if isinstance(scen, list) else None},
                    "report_len_bucket": len(rep.json().get("report", "")) // 200,
                }
            for _ in range(3):  # drift run needs >= 10 stored predictions
                for p in demos[disease]:
                    await c.post(f"/api/v4/{disease}/predict", json=p["data"], headers=auth,
                                 params={"patient_id": str(uuid.uuid4())})
            check(f"drift status {disease}", await c.get(f"/api/v4/admin/drift/{disease}/status", headers=auth))
            check(f"drift run    {disease}", await c.post(f"/api/v4/admin/drift/{disease}/run", headers=auth))
            rt = check(f"retrain      {disease} (min_samples unreachable)",
                       await c.post("/admin/retrain", headers=auth, json={"disease": disease, "min_samples": 10000}),
                       ok=(200, 400, 409, 422))
            print("      retrain body:", rt.text[:220])
        q = check("AL queue", await c.get("/api/v4/review/queue", headers=auth))
        qj = q.json()
        items = qj.get("items", qj) if isinstance(qj, dict) else qj
        print("      queued:", sorted({i.get("disease") for i in items}) if isinstance(items, list) else qj)
        check("AL stats", await c.get("/api/v4/review/stats", headers=auth))

    for tag, o in outputs.items():
        print(f"DIGEST {tag} predict={digest(o['predict'])} explain={digest(o['explain'])} "
              f"cf={o['cf_shape']} report_bucket={o['report_len_bucket']}")
    print("RESULT", "PASS" if not failures else f"FAIL {failures}")


asyncio.run(main())
