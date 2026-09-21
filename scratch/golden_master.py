"""
Golden master for the model-family registry refactor.

Captures every observable output of /predict, /explain, /counterfactuals,
/batch, /schema (plus /diseases and the legacy /api/v3/predict) for heart and
diabetes, through the REAL router and the REAL FastAPI app, in-process.

    backend/.venv/bin/python scratch/golden_master.py capture scratch/golden_before.json
    backend/.venv/bin/python scratch/golden_master.py capture scratch/golden_after.json
    backend/.venv/bin/python scratch/golden_master.py diff scratch/golden_before.json scratch/golden_after.json

Run before and after in the SAME environment (LightGBM output can shift
across environments, not within one).
"""
import os
import sys

# Same isolation as tests/conftest.py: never touch the developer's real DB.
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///:memory:"

import asyncio
import json
import math
import subprocess
import uuid

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

TOL = 1e-9


# ── Inputs ────────────────────────────────────────────────────────────────────

def load_demo_patients():
    """The frontend's own A/B/C/D (diabetes) and P-001..3 (heart) demo cases."""
    src = os.path.join(ROOT, "frontend", "src", "mockPatients.js")
    js = (
        "import(process.argv[1]).then(m => "
        "console.log(JSON.stringify(m.default)))"
    )
    tmp = os.path.join(os.environ.get("TMPDIR", "/tmp"), f"mock_{uuid.uuid4().hex}.mjs")
    with open(src) as f, open(tmp, "w") as g:
        g.write(f.read())
    try:
        out = subprocess.run(["node", "-e", js, tmp], capture_output=True, text=True, check=True)
    finally:
        os.remove(tmp)
    mock = json.loads(out.stdout)
    return {
        disease: [(p["id"], p["data"]) for p in patients]
        for disease, patients in mock.items()
    }


HEART_EDGE = [
    ("edge_missing_high_impact", {
        "Age": 58, "Sex": "M", "ChestPainType": "ASY", "RestingECG": "Normal",
        "ExerciseAngina": "Y", "RestingBP": None, "Cholesterol": None,
        "FastingBS": None, "MaxHR": None, "Oldpeak": None, "ST_Slope": None,
    }),
    ("edge_min_bounds", {
        "Age": 20, "Sex": "F", "ChestPainType": "TA", "RestingECG": "ST",
        "ExerciseAngina": "N", "RestingBP": 80, "Cholesterol": 100,
        "FastingBS": 0, "MaxHR": 60, "Oldpeak": -3.0, "ST_Slope": "Down",
    }),
    ("edge_max_bounds", {
        "Age": 100, "Sex": "M", "ChestPainType": "ASY", "RestingECG": "LVH",
        "ExerciseAngina": "Y", "RestingBP": 220, "Cholesterol": 600,
        "FastingBS": 1, "MaxHR": 220, "Oldpeak": 10.0, "ST_Slope": "Flat",
    }),
]

_DIAB_KEYS = [
    "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth",
    "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income",
]
_DIAB_MIN = {k: 0 for k in _DIAB_KEYS}
_DIAB_MIN.update({"BMI": 10.0, "GenHlth": 1, "Age": 1, "Education": 1, "Income": 1})
_DIAB_MAX = {k: 1 for k in _DIAB_KEYS}
_DIAB_MAX.update({"BMI": 100.0, "MentHlth": 30, "PhysHlth": 30, "GenHlth": 5,
                  "Age": 13, "Education": 6, "Income": 8})
_DIAB_HISTORY_ONLY = dict(_DIAB_MIN)
_DIAB_HISTORY_ONLY.update({"Stroke": 1, "HeartDiseaseorAttack": 1, "BMI": 24.5,
                           "GenHlth": 3, "Age": 9, "Education": 4, "Income": 5,
                           "CholCheck": 1, "AnyHealthcare": 1})
DIABETES_EDGE = [
    ("edge_all_min", _DIAB_MIN),
    ("edge_all_max", _DIAB_MAX),
    ("edge_history_only", _DIAB_HISTORY_ONLY),
]


def batch_rows(disease, cases):
    """12 rows: every case, then deterministic variations, one invalid row last."""
    rows = [dict(p) for _, p in cases]
    i = 0
    while len(rows) < 11:
        base = dict(cases[i % len(cases)][1])
        if disease == "heart_disease":
            base["Age"] = 30 + (i * 7) % 60
            if base.get("MaxHR") is not None:
                base["MaxHR"] = 90 + (i * 11) % 100
        else:
            base["BMI"] = 18.0 + (i * 3.7) % 30
            base["Age"] = 1 + (i * 5) % 13
        rows.append(base)
        i += 1
    bad = dict(cases[0][1])
    bad["Age"] = 5 if disease == "heart_disease" else 99   # out of schema bounds
    rows.append(bad)
    keys = list(cases[0][1].keys())
    lines = [",".join(keys)]
    for r in rows:
        lines.append(",".join("" if r.get(k) is None else str(r.get(k)) for k in keys))
    return "\n".join(lines).encode("utf-8")


# ── Capture ───────────────────────────────────────────────────────────────────

async def capture(out_path):
    from unittest.mock import patch  # noqa: F401
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

    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False}, poolclass=StaticPool,
    )
    Session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async def _get_db():
        async with Session() as s:
            yield s

    app = main_module.app
    app.dependency_overrides[get_db] = _get_db
    app.state.limiter.enabled = False
    import backend.cache as cache_module
    mem = InMemoryBackend()
    cache_module._backend = mem
    FastAPICache.init(mem, prefix="golden")

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    async with Session() as s:
        role = Role(name="doctor", description="Clinical access")
        s.add(role)
        await s.flush()
        user = User(id=str(uuid.uuid4()), email="golden@test.com",
                    hashed_password=hash_password("Golden1234"),
                    full_name="Golden", is_active=True)
        s.add(user)
        await s.flush()
        await s.execute(user_roles.insert().values(user_id=user.id, role_id=role.id))
        await s.commit()

    demos = load_demo_patients()
    cases = {
        "heart_disease": demos["heart_disease"] + HEART_EDGE,
        "diabetes": demos["diabetes"] + DIABETES_EDGE,
    }

    result = {"cases": {d: [c for c, _ in v] for d, v in cases.items()}}
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        tok = (await c.post("/auth/login", json={"email": "golden@test.com",
                                                  "password": "Golden1234"})).json()["access_token"]
        auth = {"Authorization": f"Bearer {tok}"}

        def rec(resp):
            return {"status": resp.status_code, "body": resp.json()}

        result["diseases"] = rec(await c.get("/api/v4/diseases"))
        for disease, dcases in cases.items():
            d = result[disease] = {"predict": {}, "explain": {}, "counterfactuals": {}}
            d["schema"] = rec(await c.get(f"/api/v4/{disease}/schema"))
            for name, payload in dcases:
                # patient_id bypasses the predict cache; anonymous => nothing persisted.
                d["predict"][name] = rec(await c.post(
                    f"/api/v4/{disease}/predict", json=payload,
                    params={"patient_id": "00000000-0000-0000-0000-000000000000"}))
                d["explain"][name] = rec(await c.post(f"/api/v4/{disease}/explain", json=payload))
                cf = rec(await c.post(f"/api/v4/{disease}/counterfactuals", json=payload))
                body = cf["body"]
                scen = body.get("counterfactuals") if isinstance(body, dict) else None
                cf["shape"] = {
                    "status_code": cf["status"],
                    "keys": sorted(body.keys()) if isinstance(body, dict) else None,
                    "status": body.get("status") if isinstance(body, dict) else None,
                    "scenario_count": len(scen) if isinstance(scen, list) else None,
                    "scenario_keys": sorted({k for s in scen for k in s}) if scen else [],
                }
                d["counterfactuals"][name] = cf
            d["batch"] = rec(await c.post(
                f"/api/v4/{disease}/batch", headers=auth,
                files={"file": ("golden.csv", batch_rows(disease, dcases), "text/csv")}))
        result["legacy_v3_predict"] = rec(await c.post(
            "/api/v3/predict", json=cases["heart_disease"][0][1] | {}))

    with open(out_path, "w") as f:
        json.dump(result, f, indent=1, sort_keys=True)
    print(f"wrote {out_path}")
    for disease in cases:
        d = result[disease]
        print(disease, "predict", {k: v["status"] for k, v in d["predict"].items()})
        print(disease, "explain", {k: v["status"] for k, v in d["explain"].items()})
        print(disease, "cf", {k: (v["shape"]["status"], v["shape"]["scenario_count"])
                              for k, v in d["counterfactuals"].items()})
        print(disease, "batch", d["batch"]["status"],
              {k: d["batch"]["body"].get(k) for k in ("total", "succeeded", "failed")})
        print(disease, "schema", d["schema"]["status"])


# ── Diff ──────────────────────────────────────────────────────────────────────

def diff(a, b, path="$", out=None):
    out = [] if out is None else out
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a:
                out.append(f"{path}.{k}: added in after")
            elif k not in b:
                out.append(f"{path}.{k}: missing in after")
            else:
                diff(a[k], b[k], f"{path}.{k}", out)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{path}: length {len(a)} != {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            diff(x, y, f"{path}[{i}]", out)
    elif isinstance(a, bool) or isinstance(b, bool):
        if a is not b:
            out.append(f"{path}: {a!r} != {b!r}")
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if type(a) is not type(b):
            out.append(f"{path}: type {type(a).__name__} != {type(b).__name__}")
        elif not (a == b or (math.isnan(a) and math.isnan(b)) or abs(a - b) <= TOL):
            out.append(f"{path}: {a!r} != {b!r} (|d|={abs(a - b):.3e})")
    elif a != b:
        out.append(f"{path}: {a!r} != {b!r}")
    return out


def normalise(doc, strict=False):
    """
    What the comparison is allowed to see.

    - /counterfactuals: shape + scenario count only. The diabetes scenario
      CONTENT differs between two runs of the same unchanged code in the same
      process environment (verified 2026-09-21: 119 differences between two
      back-to-back captures on deploy/v2-platform), so it cannot be a
      refactor oracle. Status code, response keys, status and scenario count
      were stable across both runs.
    - request_id: a fresh uuid4 per error response.
    """
    doc = json.loads(json.dumps(doc))
    # --strict keeps the full counterfactual content: only meaningful when
    # both captures ran with the same PYTHONHASHSEED (set iteration order in
    # the generator depends on it).
    for disease in ([] if strict else doc.get("cases", {})):
        for cf in doc[disease]["counterfactuals"].values():
            cf.pop("body", None)

    def strip(x):
        if isinstance(x, dict):
            x.pop("request_id", None)
            for v in x.values():
                strip(v)
        elif isinstance(x, list):
            for v in x:
                strip(v)
    strip(doc)
    return doc


def count_leaves(x):
    if isinstance(x, dict):
        return sum(count_leaves(v) for v in x.values())
    if isinstance(x, list):
        return sum(count_leaves(v) for v in x)
    return 1


if __name__ == "__main__":
    if sys.argv[1] == "capture":
        asyncio.run(capture(sys.argv[2]))
    elif sys.argv[1] == "diff":
        with open(sys.argv[2]) as f:
            a = json.load(f)
        with open(sys.argv[3]) as f:
            b = json.load(f)
        strict = "--strict" in sys.argv
        a, b = normalise(a, strict), normalise(b, strict)
        problems = diff(a, b)
        print(f"compared {count_leaves(a)} leaf values (float tol {TOL})")
        if problems:
            print(f"DIFFERENT: {len(problems)} difference(s)")
            for p in problems[:200]:
                print("  " + p)
            sys.exit(1)
        print("IDENTICAL")
    else:
        sys.exit("usage: capture <out.json> | diff <before.json> <after.json>")
