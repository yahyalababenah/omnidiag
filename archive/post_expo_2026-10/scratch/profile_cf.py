"""AUDIT 6c — where does counterfactual time actually go? In-process, read-only."""
import os,sys,time,warnings,json
warnings.filterwarnings("ignore")
ROOT="/home/yahia/Desktop/Projects/Heart_Disease_Project"; sys.path.insert(0,ROOT); os.chdir(ROOT)
import yaml, pandas as pd, numpy as np
from backend.ensemble_loader import EnsembleModelLoader
from backend.model_loader import ModelLoader

HEART={"Age":63,"Sex":"M","ChestPainType":"ASY","RestingBP":150,"Cholesterol":290,"FastingBS":1,
       "RestingECG":"ST","MaxHR":108,"ExerciseAngina":"Y","Oldpeak":3.2,"ST_Slope":"Flat"}
DIAB={"HighBP":1,"HighChol":1,"CholCheck":1,"BMI":36.0,"Smoker":1,"Stroke":0,"HeartDiseaseorAttack":0,
      "PhysActivity":0,"Fruits":0,"Veggies":0,"HvyAlcoholConsump":0,"AnyHealthcare":1,"NoDocbcCost":0,
      "GenHlth":4,"MentHlth":15,"PhysHlth":20,"DiffWalk":1,"Sex":1,"Age":10,"Education":3,"Income":2}

# ---------- DIABETES ----------
cfg=yaml.safe_load(open("configs/diabetes.yaml"))
L=EnsembleModelLoader(cfg)
t=time.perf_counter(); L.predict(DIAB); print(f"[diabetes] first predict (loads 3 models+meta): {time.perf_counter()-t:.3f}s")
t=time.perf_counter(); [L.predict(DIAB) for _ in range(20)]; per=(time.perf_counter()-t)/20
print(f"[diabetes] warm predict            : {per*1000:.1f} ms/call  ({per*1000:.1f}ms x 4 model invocations)")

df=L._engineer_features(L._apply_preprocessors(pd.DataFrame([DIAB])))
for name,m in L.base_models.items():
    al=df[[str(c) for c in m.feature_names_in_]]
    t=time.perf_counter(); [m.predict_proba(al) for _ in range(20)]
    print(f"[diabetes]   base '{name}' predict_proba: {(time.perf_counter()-t)/20*1000:.2f} ms/row")
t=time.perf_counter(); [L._apply_preprocessors(pd.DataFrame([DIAB])) for _ in range(20)]
print(f"[diabetes]   preprocess+df build       : {(time.perf_counter()-t)/20*1000:.2f} ms/row")

t=time.perf_counter(); r=L.generate_counterfactuals(DIAB); el=time.perf_counter()-t
print(f"[diabetes] generate_counterfactuals : {el:.3f}s -> {len(r['counterfactuals'])} scenarios, status={r['status']}")
print(f"[diabetes]   n_samples=100 candidates x (3 base + 1 meta) = ~400 model invocations + 1 baseline predict")
t=time.perf_counter(); L.explain(DIAB); print(f"[diabetes] explain (reloads 3 pkls from disk): {time.perf_counter()-t:.3f}s")
sz=sum(os.path.getsize(L._resolve_path(b['weights_path'])) for b in cfg['model']['ensemble']['base_models'])
print(f"[diabetes]   bytes re-read from disk per /explain call: {sz/1048576:.1f} MB")

# ---------- HEART ----------
hcfg=yaml.safe_load(open("configs/heart_disease.yaml"))
H=ModelLoader(hcfg)
t=time.perf_counter(); H.predict(HEART); print(f"\n[heart] first predict (loads model)   : {time.perf_counter()-t:.3f}s")
t=time.perf_counter(); [H.predict(HEART) for _ in range(50)]; per=(time.perf_counter()-t)/50
print(f"[heart] warm predict                 : {per*1000:.2f} ms/call (1 predict + 1 predict_proba)")
t=time.perf_counter(); r=H.generate_counterfactuals(HEART); el=time.perf_counter()-t
print(f"[heart] generate_counterfactuals     : {el:.3f}s -> {len(r['counterfactuals'])} scenarios, status={r['status']}")
print(f"[heart]   loop is range(800) (model_loader.py:457); each iteration calls self.predict()")
print(f"[heart]   => up to 800 x 2 = 1600 model invocations; 800 x (preprocess+engineer) pipeline runs")
print(f"[heart]   measured per-predict {per*1000:.2f} ms x 800 = {per*800:.2f}s  (vs {el:.2f}s measured total)")
t=time.perf_counter(); H.explain(HEART); print(f"[heart] explain (TreeExplainer cached): {time.perf_counter()-t:.3f}s")
import shap
print(f"\n[shap] heart explainer type={type(H.explainer).__name__}; background data passed = NONE "
      f"(shap.TreeExplainer(model) at model_loader.py:161) -> tree_path_dependent, background sample size = 0")
