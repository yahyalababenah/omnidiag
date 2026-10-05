import joblib, json, os, sys, warnings
warnings.filterwarnings("ignore")
ROOT="/home/yahia/Desktop/Projects/Heart_Disease_Project"
paths=[
 "models/heart_disease/omni_diag_xgb_optimized.pkl",
 "models/diabetes/omni_diag_xgb_optimized.pkl",
 "models/diabetes/omni_diag_lgb_optimized.pkl",
 "models/diabetes/omni_diag_rf.pkl",
 "models/diabetes/meta_learner.pkl",
 "models/diabetes/xgb_model.pkl",
 "models/diabetes/lgb_model.pkl",
 "models/diabetes/rf_model.pkl",
 "models/diabetes/voting_ensemble.pkl",
]
for p in paths:
    fp=os.path.join(ROOT,p)
    print("="*70); print(p)
    if not os.path.exists(fp):
        print("  MISSING"); continue
    print("  size:", os.path.getsize(fp), "mtime:", __import__("datetime").datetime.fromtimestamp(os.path.getmtime(fp)).isoformat())
    try:
        m=joblib.load(fp)
    except Exception as e:
        print("  LOAD FAILED:", type(e).__name__, e); continue
    print("  type:", type(m).__name__)
    fni=getattr(m,"feature_names_in_",None)
    print("  feature_names_in_:", list(fni) if fni is not None else None)
    print("  n_features_in_:", getattr(m,"n_features_in_",None))
    try:
        b=m.get_booster()
        print("  booster.feature_names:", b.feature_names)
    except Exception: pass
    try:
        params={k:v for k,v in m.get_params().items() if v is not None and k in
          ("n_estimators","max_depth","learning_rate","subsample","colsample_bytree",
           "min_child_weight","gamma","reg_alpha","reg_lambda","random_state",
           "num_leaves","min_samples_leaf","max_features","criterion","C","penalty","solver")}
        print("  params:", json.dumps(params, default=str))
    except Exception as e: print("  params err", e)
    if hasattr(m,"coef_"):
        print("  coef_:", m.coef_.tolist(), "intercept_:", m.intercept_.tolist())
