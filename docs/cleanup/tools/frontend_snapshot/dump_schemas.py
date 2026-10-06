"""Input for harness.mjs: the live schemas and /counterfactuals responses of the demo
patients, from the backend in the current directory. Usage (repo root):
    DATABASE_URL=sqlite+aiosqlite:///<tmp>/s.db backend/.venv/bin/python \
        docs/cleanup/tools/frontend_snapshot/dump_schemas.py <out.json>
"""
import json, logging, os, sys, warnings
warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
sys.path.insert(0, os.getcwd())
import backend.main as m
from fastapi.testclient import TestClient

out = {}
with TestClient(m.app) as c:
    demo = json.load(open("backend/demo_patients.json"))
    for d in ("heart_disease", "diabetes_nhanes"):
        out[d] = c.get(f"/api/v4/{d}/schema").json()
    out["_cf"] = {}
    for d in ("heart_disease", "diabetes_nhanes"):
        for p in demo[d]:
            out["_cf"][f"{d}/{p['id']}"] = {"patient": p["data"], "cf": c.post(f"/api/v4/{d}/counterfactuals", json=p["data"]).json()}
json.dump(out, open(sys.argv[1], "w"))
print("wrote", sys.argv[1])
