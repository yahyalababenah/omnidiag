"""CHECK B (shape diff) + CHECK C (cold/warm timing). Read-only HTTP."""
import json, time, urllib.request, subprocess, copy
BASE="http://127.0.0.1:8902"
mock=json.loads(subprocess.run(["node","-e",
  'console.log(JSON.stringify(require("/home/yahia/Desktop/Projects/Heart_Disease_Project/scratch/_mock.cjs")))'],
  capture_output=True,text=True).stdout)
def call(path,payload):
    req=urllib.request.Request(BASE+path,data=json.dumps(payload).encode(),
                               headers={"Content-Type":"application/json"})
    t=time.perf_counter()
    with urllib.request.urlopen(req,timeout=900) as r:
        d=json.loads(r.read()); h=r.headers.get("Cache-Hit")
    return d, time.perf_counter()-t, h

DIAB = mock["diabetes"][1]["data"]      # D-002 Karim Yaghi — known to produce 3 scenarios
HEART = mock["heart_disease"][1]["data"] # P-002 Fatima Hassan — known to produce 3

print("="*78); print("CHECK C — cold / warm timing on the CURRENT (deployed) branch")
print("="*78)
d_cold, t_dc, h_dc = call("/api/v4/diabetes/counterfactuals", DIAB)
d_warm, t_dw, h_dw = call("/api/v4/diabetes/counterfactuals", DIAB)
fresh = copy.deepcopy(DIAB); fresh["BMI"] = DIAB["BMI"] + 0.7
d_new,  t_dn, h_dn = call("/api/v4/diabetes/counterfactuals", fresh)
print(f"  diabetes CF  COLD (first ever)     : {t_dc:6.2f}s   Cache-Hit={h_dc}")
print(f"  diabetes CF  WARM (same patient)   : {t_dw:6.2f}s   Cache-Hit={h_dw}  <- cache")
print(f"  diabetes CF  WARM (NEW patient)    : {t_dn:6.2f}s   Cache-Hit={h_dn}  <- real cost")
h_cold, t_hc, h_hc = call("/api/v4/heart_disease/counterfactuals", HEART)
h_warm, t_hw, h_hw = call("/api/v4/heart_disease/counterfactuals", HEART)
print(f"  heart    CF  COLD                  : {t_hc:6.2f}s   Cache-Hit={h_hc}")
print(f"  heart    CF  WARM (same patient)   : {t_hw:6.2f}s   Cache-Hit={h_hw}")

print()
print("="*78); print("CHECK B — RAW diabetes /counterfactuals JSON")
print("="*78)
print(json.dumps(d_cold, indent=2)[:1800])
print()
print("--- RAW heart /counterfactuals JSON (first scenario only) ---")
print(json.dumps({"status":h_cold.get("status"),
                  "baseline_probability":h_cold.get("baseline_probability"),
                  "counterfactuals":h_cold.get("counterfactuals",[])[:1]}, indent=2))

print()
print("="*78); print("CHECK B — FIELD-LEVEL SHAPE DIFF")
print("="*78)
ds = (d_cold.get("counterfactuals") or [{}])[0]
hs = (h_cold.get("counterfactuals") or [{}])[0]
print(f"  envelope keys  heart   : {sorted(h_cold.keys())}")
print(f"  envelope keys  diabetes: {sorted(d_cold.keys())}")
print(f"  scenario keys  heart   : {sorted(hs.keys())}")
print(f"  scenario keys  diabetes: {sorted(ds.keys())}")
print(f"  'changes' type heart   : {type(hs.get('changes')).__name__}")
print(f"  'changes' type diabetes: {type(ds.get('changes')).__name__}")
print()
# exactly what the component reads
READS = [("scenario_id","branch discriminator, WhatIfScenarioCard.jsx:166"),
         ("probability","getReductionPct, :131"),
         ("changes","mapped as ARRAY, :231-232 and PDFReport.jsx:194"),
         ("feature","mock branch, :185/188/198"),
         ("current","mock branch, :201"),
         ("proposed","mock branch, :205"),
         ("riskReduction","mock branch, :209 + :130"),
         ("description","mock branch, :213")]
print(f"  {'field frontend reads':<18}{'heart':>10}{'diabetes':>12}   where")
for f,where in READS:
    hv = "present" if f in hs else "MISSING"
    dv = "present" if f in ds else "MISSING"
    print(f"  {f:<18}{hv:>10}{dv:>12}   {where}")
print()
print("  fields diabetes emits that the frontend never reads:",
      sorted(set(ds.keys()) - {f for f,_ in READS}))
json.dump({"diabetes":d_cold,"heart":h_cold},open("scratch/cf_raw.json","w"),indent=2)
