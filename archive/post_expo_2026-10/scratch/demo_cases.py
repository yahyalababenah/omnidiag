"""AUDIT 7 — run every demo patient (A/B/C/D + heart) through the live API."""
import json, re, subprocess, urllib.request, time, os, sys
BASE="http://127.0.0.1:8901"
src=open("frontend/src/mockPatients.js").read()

# Extract the JS object literal via node (present in repo) to avoid hand-parsing.
node_script = """
const m = require('/home/yahia/Desktop/Projects/Heart_Disease_Project/scratch/_mock.cjs');
console.log(JSON.stringify(m));
"""
js = None
pass
out = subprocess.run(["node","-e",node_script],capture_output=True,text=True)
if out.returncode!=0:
    print("node failed:", out.stderr[-500:]); sys.exit(1)
mock=json.loads(out.stdout)

def call(path,p):
    req=urllib.request.Request(BASE+path,data=json.dumps(p).encode(),headers={"Content-Type":"application/json"})
    t=time.perf_counter()
    with urllib.request.urlopen(req,timeout=600) as r: return json.loads(r.read()), time.perf_counter()-t

IMMUTABLE = None
import importlib.util
spec=importlib.util.spec_from_file_location("cfg","backend/counterfactual_generator.py")
mod=importlib.util.module_from_spec(spec); sys.path.insert(0,os.getcwd()); spec.loader.exec_module(mod)
IMMUTABLE=set(getattr(mod,"IMMUTABLE_FEATURES",[]))
MUTABLE_RANGES=getattr(mod,"MUTABLE_RANGES",None) or getattr(mod,"PERTURBATION_RANGES",{})
print(f"IMMUTABLE_FEATURES (backend/counterfactual_generator.py) = {sorted(IMMUTABLE)}\n")

for disease, patients in mock.items():
    print("="*78); print(f"DISEASE: {disease}")
    for p in patients:
        data=p["data"]
        pred,tp=call(f"/api/v4/{disease}/predict", data)
        cf,tc=call(f"/api/v4/{disease}/counterfactuals", data)
        scen=cf.get("counterfactuals") or []
        imm_present=sorted([k for k in data if k in IMMUTABLE])
        print(f"\n  {p['id']} {p['name']}")
        print(f"    risk (confidence)      : {pred['confidence']*100:.1f}%   prediction={pred['prediction']} ({pred['diagnosis']})")
        if "inference_threshold" in pred: print(f"    inference_threshold    : {pred['inference_threshold']}")
        print(f"    counterfactual status  : {cf.get('status')}   scenarios={len(scen)}  ({tc:.1f}s)")
        if len(scen)<3: print(f"    *** FLAG: fewer than 3 scenarios ***")
        print(f"    immutable feats in case: {imm_present}")
        if scen:
            s=scen[0]
            print(f"    scenario[0] keys       : {sorted(s.keys())}")
            print(f"    scenario[0] changes is : {type(s.get('changes')).__name__}")
