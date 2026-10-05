"""AUDIT 6b — cache-busted latency (true warm cost) + CF model-invocation count."""
import json, time, urllib.request, copy
BASE="http://127.0.0.1:8901"
HEART={"Age":63,"Sex":"M","ChestPainType":"ASY","RestingBP":150,"Cholesterol":290,"FastingBS":1,
       "RestingECG":"ST","MaxHR":108,"ExerciseAngina":"Y","Oldpeak":3.2,"ST_Slope":"Flat"}
DIAB={"HighBP":1,"HighChol":1,"CholCheck":1,"BMI":36.0,"Smoker":1,"Stroke":0,"HeartDiseaseorAttack":0,
      "PhysActivity":0,"Fruits":0,"Veggies":0,"HvyAlcoholConsump":0,"AnyHealthcare":1,"NoDocbcCost":0,
      "GenHlth":4,"MentHlth":15,"PhysHlth":20,"DiffWalk":1,"Sex":1,"Age":10,"Education":3,"Income":2}
def call(path,p):
    req=urllib.request.Request(BASE+path,data=json.dumps(p).encode(),headers={"Content-Type":"application/json"})
    t=time.perf_counter()
    with urllib.request.urlopen(req,timeout=600) as r:
        d=json.loads(r.read()); h=r.headers.get("Cache-Hit")
    return time.perf_counter()-t,d,h

print("CACHE-BUSTED (each call a distinct patient -> real compute cost, process already warm)")
print(f"{'endpoint':<36}{'run1':>8}{'run2':>8}{'run3':>8}{'cacheHit':>10}")
for path,base,name,knob in [("/api/v4/heart_disease/counterfactuals",HEART,"heart /counterfactuals","Cholesterol"),
                            ("/api/v4/diabetes/counterfactuals",DIAB,"diabetes /counterfactuals","BMI"),
                            ("/api/v4/heart_disease/explain",HEART,"heart /explain","Cholesterol"),
                            ("/api/v4/diabetes/explain",DIAB,"diabetes /explain","BMI"),
                            ("/api/v4/heart_disease/predict",HEART,"heart /predict","Cholesterol"),
                            ("/api/v4/diabetes/predict",DIAB,"diabetes /predict","BMI")]:
    ts=[];hh=None
    for i in range(3):
        p=copy.deepcopy(base); p[knob]=base[knob]+i+1
        t,d,h=call(path,p); ts.append(t); hh=h
    print(f"{name:<36}{ts[0]:>8.3f}{ts[1]:>8.3f}{ts[2]:>8.3f}{str(hh):>10}")
