"""Replay what each frontend screen requests, against a running backend (BASE).
Writes the raw responses as JSON for the frontend-side post-processing."""
import json, sys, urllib.request, urllib.error

BASE, OUT = sys.argv[1], sys.argv[2]
DEMO = json.load(open(sys.argv[3]))           # the NEW mockPatients (heart + nhanes)


def call(method, path, body=None, token=None):
    req = urllib.request.Request(BASE + path, method=method,
                                 data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          **({"Authorization": f"Bearer {token}"} if token else {})})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return {"status": r.status, "body": json.loads(r.read() or b"null")}
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            body = json.loads(raw)
        except Exception:
            body = raw.decode(errors="replace")
        return {"status": e.code, "statusText": e.reason, "body": body}


out = {}
tok = call("POST", "/auth/login", {"email": "probe@test.com", "password": "Probe12345"})["body"]["access_token"]
out["diseases"] = call("GET", "/api/v4/diseases")
for d, patients in DEMO.items():
    out[f"schema/{d}"] = {"status": call("GET", f"/api/v4/{d}/schema")["status"]}
    for p in patients:
        k = f"{d}/{p['id']}"
        pr = call("POST", f"/api/v4/{d}/predict", p["data"])
        cf = call("POST", f"/api/v4/{d}/counterfactuals", p["data"])
        ex = call("POST", f"/api/v4/{d}/explain", p["data"])
        out[f"predict/{k}"] = {"status": pr["status"], "decision": (pr["body"] or {}).get("decision")}
        out[f"explain/{k}"] = {"status": ex["status"]}
        out[f"cf/{k}"] = {**cf, "patient": p["data"], "prediction": (pr["body"] or {}).get("prediction")}
q = call("GET", "/api/v4/review/queue?page=1&limit=100", token=tok)
out["queue"] = q
out["stats"] = call("GET", "/api/v4/review/stats", token=tok)
items = q["body"]["items"]
diab = next((i for i in items if i.get("disease") == "diabetes"), None)
heart = next((i for i in items if i.get("disease") == "heart_disease"), None)
if diab:
    out["annotate/diabetes"] = call("POST", f"/api/v4/review/{diab['id']}/annotate", {"label": 1, "notes": None}, tok)
if heart:
    out["skip/heart"] = call("POST", f"/api/v4/review/{heart['id']}/skip", None, tok)
pts = sys.argv[4].split(",")
for pt in pts:
    out[f"history/{pt}"] = call("GET", f"/api/v4/patients/{pt}/predictions?page=1&limit=10", token=tok)
out["retrain/heart"] = {"status": call("POST", "/admin/retrain", {"disease": "heart_disease", "min_samples": 10000}, tok)["status"]}
json.dump(out, open(OUT, "w"), indent=1, default=str)
print("probe written", OUT)
