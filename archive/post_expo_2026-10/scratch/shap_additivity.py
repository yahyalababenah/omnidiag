"""AUDIT 7 — prove the heart label-inversion is applied exactly once (SHAP additivity)."""
import json,urllib.request,subprocess,math
BASE="http://127.0.0.1:8901"
mock=json.loads(subprocess.run(["node","-e",'console.log(JSON.stringify(require("/home/yahia/Desktop/Projects/Heart_Disease_Project/scratch/_mock.cjs")))'],capture_output=True,text=True).stdout)
def call(p,d):
    r=urllib.request.Request(BASE+p,data=json.dumps(d).encode(),headers={"Content-Type":"application/json"})
    return json.loads(urllib.request.urlopen(r,timeout=600).read())
sig=lambda z: 1/(1+math.exp(-z))
for pt in mock["heart_disease"]:
    d=pt["data"]; e=call("/api/v4/heart_disease/explain",d); pr=call("/api/v4/heart_disease/predict",d)
    z=e["base_value"]+sum(c["shap_value"] for c in e["chart_data"])
    print(f"{pt['id']}: sigmoid(base+Σshap)={sig(z):.6f}  explain.confidence={e['confidence']:.6f}  "
          f"predict.confidence={pr['confidence']:.6f}  additive={abs(sig(z)-e['confidence'])<1e-4}  "
          f"explain==predict={abs(e['confidence']-pr['confidence'])<1e-9}")
print()
print("If the flip were applied twice, sigmoid(base+Σshap) would equal 1-confidence instead.")
for pt in mock["diabetes"][:2]:
    d=pt["data"]; e=call("/api/v4/diabetes/explain",d); pr=call("/api/v4/diabetes/predict",d)
    z=e["base_value"]+sum(c["shap_value"] for c in e["chart_data"])
    print(f"{pt['id']} (diabetes, no inversion expected): sigmoid(base+Σshap)={sig(z):.4f} "
          f"vs ensemble confidence={e['confidence']:.4f} (weighted-avg SHAP across 3 models is NOT "
          f"expected to be additive w/ the meta-learner output)  explain==predict={abs(e['confidence']-pr['confidence'])<1e-9}")
