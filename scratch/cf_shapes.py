import json,urllib.request,subprocess
BASE="http://127.0.0.1:8901"
mock=json.loads(subprocess.run(["node","-e",'console.log(JSON.stringify(require("/home/yahia/Desktop/Projects/Heart_Disease_Project/scratch/_mock.cjs")))'],capture_output=True,text=True).stdout)
def call(p,d):
    r=urllib.request.Request(BASE+p,data=json.dumps(d).encode(),headers={"Content-Type":"application/json"})
    return json.loads(urllib.request.urlopen(r,timeout=600).read())
h=call("/api/v4/heart_disease/counterfactuals", mock["heart_disease"][1]["data"])
d=call("/api/v4/diabetes/counterfactuals", mock["diabetes"][1]["data"])
print("### HEART scenario[0] (P-002) ###"); print(json.dumps(h["counterfactuals"][0],indent=2)[:900])
print("\n### DIABETES scenario[0] (D-002) ###"); print(json.dumps(d["counterfactuals"][0],indent=2)[:900])
print("\n### /explain response keys, diabetes (declared vs emitted) ###")
e=call("/api/v4/diabetes/explain", mock["diabetes"][1]["data"])
print("emitted:", sorted(e.keys()))
print("chart_data[0]:", e["chart_data"][0], " base_value:", e["base_value"])
eh=call("/api/v4/heart_disease/explain", mock["heart_disease"][1]["data"])
print("heart explain keys:", sorted(eh.keys()))
print("heart chart_data[0]:", eh["chart_data"][0], " base_value:", eh["base_value"], " confidence:", eh["confidence"])
ph=call("/api/v4/heart_disease/predict", mock["heart_disease"][1]["data"])
print("heart predict confidence:", ph["confidence"], "-> explain/predict agree:", abs(ph["confidence"]-eh["confidence"])<1e-9)
