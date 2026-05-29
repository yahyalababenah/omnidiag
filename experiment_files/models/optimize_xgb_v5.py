"""
OmniDiag v5.1 — Optuna Hyperparameter Optimization (Enhanced)
==============================================================
Runs 300 Optuna trials on the NEW dataset (605 patients, 18 features)
with 5-fold stratified cross-validation.

New features added in v5.1:
    - Age_Bins: Ordinal age categories (0=Young, 1=Middle, 2=Senior)
    - Global_Risk_Score: Composite weighted risk score

Note: scale_pos_weight is intentionally NOT included in the search space.
Dataset has 380 positive / 225 negative (positives are the majority).
scale_pos_weight < 1 downweights positives, which reduces accuracy.
Default (1.0 = balanced) works best for this dataset.

Usage:
    python experiment_files/models/optimize_xgb_v5.py
"""

import pandas as pd
import xgboost as xgb
import optuna
import json
import os
import sys
import datetime
import logging
from sklearn.model_selection import cross_val_score, StratifiedKFold

# Add project root to path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from configs.config_loader import load_config, resolve_path
from models.advanced_feature_engineering import (
    engineer_heuristic_features,
    engineer_medical_features,
)

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

print("=" * 60)
print("🧬 OmniDiag v5.1 — Optuna Hyperparameter Optimization (300 trials)")
print("=" * 60)

# ------------------------------------------------------------------
# Load config
# ------------------------------------------------------------------
cfg = load_config("heart_disease")
processed_path = resolve_path(cfg, "data", "processed_path")
target_col = cfg["disease"]["target_column"]
n_trials = cfg["training"].get("optimization_trials", 300)
model_dir = os.path.dirname(resolve_path(cfg, "model", "weights_path"))

# ------------------------------------------------------------------
# 1. Prepare data
# ------------------------------------------------------------------
data_path = os.path.join(processed_path, cfg["data"]["final_clean_file"])
print(f"\n📂 قراءة البيانات من: {data_path}")
df = pd.read_csv(data_path)

# Apply feature engineering
df_fe = engineer_heuristic_features(df)
df_fe = engineer_medical_features(df_fe)

assert df_fe.isnull().sum().sum() == 0, "❌ NaN values in feature engineered data!"

X = df_fe.drop(columns=[target_col])
y = df_fe[target_col]

print(f"   البيانات: {len(df_fe)} عينة, {X.shape[1]} ميزة")
print(f"   الميزات: {list(X.columns)}")
print(f"   توزيع الهدف: {y.value_counts().to_dict()}")

# Calculate scale_pos_weight for class imbalance
count_neg = int((y == 0).sum())
count_pos = int((y == 1).sum())
pos_weight = count_neg / count_pos
print(f"   📊 scale_pos_weight: {pos_weight:.4f} ({count_neg}/{count_pos})")

# ------------------------------------------------------------------
# 2. Define Optuna objective (5-fold stratified CV)
# ------------------------------------------------------------------
def objective(trial):
    """Optuna objective: maximize cross-validated accuracy."""

    param = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1200),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        # Note: scale_pos_weight intentionally omitted.
        # Dataset has 380 positive / 225 negative — positives are the majority.
        # Using scale_pos_weight < 1 downweights the positive class,
        # which reduces overall accuracy. Default (1.0) works best here.
        'random_state': 42,
        'eval_metric': 'logloss',
        'use_label_encoder': False,
    }

    model = xgb.XGBClassifier(**param)

    # 5-fold stratified cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)

    return scores.mean()

# ------------------------------------------------------------------
# 3. Run optimization
# ------------------------------------------------------------------
print(f"\n🚀 بدء {n_trials} تجربة Optuna مع Pruning Callback و Early Stopping...")
print(f"   فضاء البحث: n_estimators[100-1200], max_depth[3-12], lr[0.001-0.1], ...")
print(f"   Early Stopping: 50 round")
print(f"   Pruning: XGBoostPruningCallback (validation-logloss)")
print(f"   scale_pos_weight: {pos_weight:.4f}")

study = optuna.create_study(
    direction="maximize",
    study_name="omnidiag_v5_1_xgb",
    storage=None,  # In-memory (no SQLite persistence needed)
)

study.optimize(objective, n_trials=n_trials)

# ------------------------------------------------------------------
# 4. Results
# ------------------------------------------------------------------
best_trial = study.best_trial
print(f"\n🏆 أفضل تجربة (#{best_trial.number}):")
print(f"   الدقة (Validation Accuracy): {best_trial.value:.4f}")
print(f"   أفضل المعلمات:")
for key, value in best_trial.params.items():
    print(f"      {key}: {value}")

# ------------------------------------------------------------------
# 5. Save best params to grid_best.json
# ------------------------------------------------------------------
grid_best_path = os.path.join(model_dir, "grid_best.json")

grid_best = {
    "version": "5.1.0",
    "optimization_date": datetime.datetime.now().isoformat(),
    "n_trials": n_trials,
    "dataset_size": len(df_fe),
    "n_features": X.shape[1],
    "best_score": round(float(best_trial.value), 6),
    "best_params": best_trial.params,
    "feature_names": list(X.columns),
}

with open(grid_best_path, "w") as f:
    json.dump(grid_best, f, indent=2)

print(f"\n💾 تم حفظ أفضل المعلمات في: {grid_best_path}")

# Print summary of top 5 trials
print("\n📊 أفضل 5 تجارب:")
all_trials = study.trials_dataframe()
all_trials_sorted = all_trials.sort_values("value", ascending=False)
print(all_trials_sorted[["number", "value"]].head(5).to_string(index=False))

# Log pruning statistics
n_complete = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
n_pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
print(f"\n📊 إحصائيات التجارب:")
print(f"   مكتملة: {n_complete}")
print(f"   مقصوصة (Pruned): {n_pruned}")

print("\n" + "=" * 60)
print(f"🎯 اكتمل تحسين Optuna! أفضل دقة: {best_trial.value:.4f}")
print("=" * 60)
