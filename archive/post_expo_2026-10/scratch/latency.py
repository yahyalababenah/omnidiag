"""AUDIT 6 — endpoint latency, cold vs warm. Read-only HTTP client."""
import json, time, urllib.request, statistics, sys
BASE = "http://127.0.0.1:8901"

HEART = {"Age":63,"Sex":"M","ChestPainType":"ASY","RestingBP":150,"Cholesterol":290,
         "FastingBS":1,"RestingECG":"ST","MaxHR":108,"ExerciseAngina":"Y","Oldpeak":3.2,
         "ST_Slope":"Flat"}
DIAB = {"HighBP":1,"HighChol":1,"CholCheck":1,"BMI":36.0,"Smoker":1,"Stroke":0,
        "HeartDiseaseorAttack":0,"PhysActivity":0,"Fruits":0,"Veggies":0,
        "HvyAlcoholConsump":0,"AnyHealthcare":1,"NoDocbcCost":0,"GenHlth":4,
        "MentHlth":15,"PhysHlth":20,"DiffWalk":1,"Sex":1,"Age":10,"Education":3,"Income":2}
NOTE = {"note":"63 year old male, BP 150/90, cholesterol 290, max HR 108 on stress test, "
                "exercise induced angina present, ST depression 3.2, fasting glucose elevated.",
        "disease":"heart_disease"}

def call(path, payload, tag):
    body = json.dumps(payload).encode()
    req = urllib.request.Request(BASE+path, data=body,
                                 headers={"Content-Type":"application/json"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            data = json.loads(r.read())
            hit = r.headers.get("Cache-Hit")
    except Exception as e:
        el = time.perf_counter()-t0
        return el, {"__error__": f"{type(e).__name__}: {e}"}, None
    return time.perf_counter()-t0, data, hit

CASES = [
    ("/api/v4/heart_disease/predict",        HEART, "heart /predict"),
    ("/api/v4/heart_disease/explain",        HEART, "heart /explain (SHAP)"),
    ("/api/v4/heart_disease/counterfactuals",HEART, "heart /counterfactuals"),
    ("/api/v4/diabetes/predict",             DIAB,  "diabetes /predict"),
    ("/api/v4/diabetes/explain",             DIAB,  "diabetes /explain (SHAP)"),
    ("/api/v4/diabetes/counterfactuals",     DIAB,  "diabetes /counterfactuals"),
    ("/api/v4/parse-notes",                  NOTE,  "/parse-notes"),
]

print(f"{'endpoint':<34}{'cold s':>9}{'warm1 s':>9}{'warm2 s':>9}{'warm3 s':>9}{'cache':>8}")
print("-"*80)
results = {}
for path, payload, tag in CASES:
    cold, d0, h0 = call(path, payload, tag)
    warms = []
    for _ in range(3):
        w, d, h = call(path, payload, tag)
        warms.append(w)
    results[tag] = dict(cold=cold, warm=warms, sample=d0, cache=h0)
    print(f"{tag:<34}{cold:>9.3f}{warms[0]:>9.3f}{warms[1]:>9.3f}{warms[2]:>9.3f}{str(h0):>8}")

print()
print("== response keys (first/cold call) ==")
for tag, r in results.items():
    s = r["sample"]
    if isinstance(s, dict):
        print(f"  {tag:<34} {sorted(s.keys())}")
json.dump({k: {"cold": v["cold"], "warm": v["warm"], "cache_header": v["cache"]}
           for k, v in results.items()}, open("scratch/latency_results.json","w"), indent=2)
