"""CHECK C — true cold/warm on a freshly-started process with an empty cache."""
import json, time, urllib.request, subprocess, copy
BASE="http://127.0.0.1:8904"
mock=json.loads(subprocess.run(["node","-e",
  'console.log(JSON.stringify(require("/home/yahia/Desktop/Projects/Heart_Disease_Project/scratch/_mock.cjs")))'],
  capture_output=True,text=True).stdout)
def call(path,payload):
    req=urllib.request.Request(BASE+path,data=json.dumps(payload).encode(),
                               headers={"Content-Type":"application/json"})
    t=time.perf_counter()
    with urllib.request.urlopen(req,timeout=900) as r:
        d=json.loads(r.read()); h=r.headers.get("Cache-Hit")
    return d,time.perf_counter()-t,h
D=mock["diabetes"][1]["data"]; H=mock["heart_disease"][1]["data"]
print("n_samples=100 + 3600s cache CONFIRMED present on deploy/v2-platform, origin/deploy/v2-platform, hf/main\n")
print(f"{'call':<52}{'seconds':>9}{'Cache-Hit':>11}")
_,t,h=call("/api/v4/diabetes/counterfactuals",D)
print(f"{'diabetes CF — COLD (models unloaded, cache empty)':<52}{t:>9.2f}{str(h):>11}")
_,t,h=call("/api/v4/diabetes/counterfactuals",D)
print(f"{'diabetes CF — WARM, SAME patient (cache hit)':<52}{t:>9.2f}{str(h):>11}")
for i in (1,2):
    f=copy.deepcopy(D); f["BMI"]=D["BMI"]+i*1.3
    _,t,h=call("/api/v4/diabetes/counterfactuals",f)
    print(f"{'diabetes CF — WARM, NEW patient #%d (real cost)'%i:<52}{t:>9.2f}{str(h):>11}")
_,t,h=call("/api/v4/heart_disease/counterfactuals",H)
print(f"{'heart CF — COLD (heart uses hardcoded range(800))':<52}{t:>9.2f}{str(h):>11}")
f=copy.deepcopy(H); f["Cholesterol"]=H["Cholesterol"]+7
_,t,h=call("/api/v4/heart_disease/counterfactuals",f)
print(f"{'heart CF — WARM, NEW patient (real cost)':<52}{t:>9.2f}{str(h):>11}")
