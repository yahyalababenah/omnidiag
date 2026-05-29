"""
OmniDiag v5.1 — A/B Diagnostic: Isolate accuracy drop cause
============================================================
Test A: Train with v5.0 PROVEN hyperparameters on v5.1 18-feature dataset.
  → If ≈81%: problem is NEW PARAMS (from 300-trial Optuna)
  → If ≈78%: problem is NEW FEATURES (Age_Bins, Global_Risk_Score)

v5.0 params (81.82% test accuracy on 15 features):
  n_estimators=1124, lr=0.0063, max_depth=5, colsample=0.479,
  reg_alpha=0.190, reg_lambda=0.142, gamma=1.446, subsample=0.833

Usage:
    python experiment_files/models/diagnose_v5_drop.py
"""

import pandas as pd
import xgboost as xgb
import sys
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from configs.config_loader import load_config, resolve_path
from models.advanced_feature_engineering import (
    engineer_heuristic_features,
    engineer_medical_features,
    engineer_age_bins,
    engineer_global_risk,
)

print("=" * 70)
print("🔬 OmniDiag v5.1 — A/B Diagnostic: v5.0 PARAMS × v5.1 FEATURES")
print("=" * 70)

# Load config & data
cfg = load_config("heart_disease")
processed_path = resolve_path(cfg, "data", "processed_path")
target_col = cfg["disease"]["target_column"]

data_path = os.path.join(processed_path, cfg["data"]["final_clean_file"])
df = pd.read_csv(data_path)
print(f"\n📂 Data: {df.shape[0]} rows × {df.shape[1]} cols (12 base)")

# Apply ALL v5.1 feature engineering (18 features)
df_fe = engineer_heuristic_features(df)
df_fe = engineer_medical_features(df_fe)
df_fe = engineer_age_bins(df_fe)
df_fe = engineer_global_risk(df_fe)
print(f"   After engineering: {df_fe.shape[0]} rows × {df_fe.shape[1]} cols (18 features)")

X = df_fe.drop(columns=[target_col])
y = df_fe[target_col]
print(f"   Features: {list(X.columns)}")
print(f"   Target distribution: {y.value_counts().to_dict()}")

# v5.0 PROVEN hyperparameters (achieved 81.82% test accuracy)
v5_params = {
    'n_estimators': 1124,
    'max_depth': 5,
    'learning_rate': 0.006308564514079806,
    'subsample': 0.8331048481710659,
    'colsample_bytree': 0.4792434452981927,
    'min_child_weight': 3,
    'gamma': 1.4459953680077136,
    'reg_alpha': 0.1902031263838727,
    'reg_lambda': 0.14167100417623021,
    'random_state': 42,
    'eval_metric': 'logloss',
}

print(f"\n📋 Using v5.0 PROVEN params:")
for k, v in v5_params.items():
    print(f"   {k}: {v}")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n✂️ Split: Train={X_train.shape[0]}, Test={X_test.shape[0]}")

# Train
model = xgb.XGBClassifier(**v5_params)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

# Evaluate
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

print(f"\n{'='*70}")
print(f"📊 DIAGNOSTIC RESULT")
print(f"{'='*70}")
print(f"   v5.0 PARAMS + v5.1 FEATURES (18 features):")
print(f"   ✅ Accuracy: {accuracy:.4f}")
print(f"   ✅ ROC-AUC:  {roc_auc:.4f}")
print()

# Interpretation
print(f"{'─'*70}")
if accuracy >= 0.80:
    print(f"🔍 INTERPRETATION: Accuracy ≈ {accuracy:.4f} (≥80%)")
    print(f"   → The v5.0 PARAMS work well with 18 features.")
    print(f"   → The accuracy DROP is caused by the NEW HYPERPARAMETERS from 300-trial Optuna.")
    print(f"   → SOLUTION: Keep v5.1 features, use v5.0 params (or re-run Optuna with v5.0's param space).")
else:
    print(f"🔍 INTERPRETATION: Accuracy ≈ {accuracy:.4f} (<80%)")
    print(f"   → The v5.0 PARAMS also drop with 18 features.")
    print(f"   → The accuracy DROP is caused by the NEW FEATURES (Age_Bins, Global_Risk_Score).")
    print(f"   → SOLUTION: Drop Age_Bins and Global_Risk_Score, keep only 16 features (12 base + 3 heuristic + 2 medical).")
print(f"{'─'*70}")

# Save diagnostic result
import json
from datetime import datetime
result = {
    "diagnostic": "v5.0_params_on_v5.1_features",
    "timestamp": datetime.now().isoformat(),
    "n_features": X.shape[1],
    "feature_names": list(X.columns),
    "accuracy": round(float(accuracy), 6),
    "roc_auc": round(float(roc_auc), 6),
    "params_used": {k: v for k, v in v5_params.items() if k != 'random_state'},
}
result_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "models/heart_disease/diagnostic_v5_ab.json"
)
with open(result_path, "w") as f:
    json.dump(result, f, indent=2)
print(f"\n💾 Diagnostic saved: {result_path}")
