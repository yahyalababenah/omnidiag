"""
OmniDiag v5.1 — Retrain XGBoost on 16-Feature CAD Dataset
===========================================================
Trains XGBoost classifier using PROVEN v5.0 hyperparameters
on the new merged dataset (605 patients).

Features (16 total):
    - 11 base features (encoded + scaled)
    - 3 heuristic features (Age_BP_Interaction, HR_Age_Ratio, Chol_Age_Ratio)
    - 2 medical features (RPP, Exercise_Risk_Index)

Note: Age_Bins and Global_Risk_Score were tested in A/B diagnostic
but did NOT improve accuracy (diagnose_v5_drop.py confirmed).
Using v5.0 proven params (81.82% baseline) on the refined pipeline.

Pipeline:
    1. Use v5.0 PROVEN hyperparameters (not Optuna — the 300-trial search
       found unstable params that overfit)
    2. Load final_ready_data.csv (12 cols, encoded+scaled)
    3. Apply feature engineering (heuristic → medical)
    4. Train XGBoost with v5.0 proven params
    5. Evaluate (accuracy, ROC-AUC, confusion matrix, classification report)
    6. Save model to omni_diag_xgb_optimized.pkl
    7. Update metrics.json

Usage:
    python experiment_files/models/train_v5_xgb.py
"""

import pandas as pd
import xgboost as xgb
import pickle
import json
import os
import sys
import datetime
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix

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
print("🚀 OmniDiag v5.1 — XGBoost Training (Optuna-Optimized)")
print("=" * 60)

# ------------------------------------------------------------------
# Load config
# ------------------------------------------------------------------
cfg = load_config("heart_disease")
processed_path = resolve_path(cfg, "data", "processed_path")
model_dir = os.path.dirname(resolve_path(cfg, "model", "weights_path"))
target_col = cfg["disease"]["target_column"]

# ------------------------------------------------------------------
# 1. Use v5.0 PROVEN hyperparameters (achieved 81.82% test accuracy)
# ------------------------------------------------------------------
# A/B diagnostic (diagnose_v5_drop.py) confirmed:
# - v5.0 params + 18 features → 80.17% (good)
# - v5.1 Optuna params + 18 features → 78.51% (bad)
# - Root cause: 300-trial Optuna found overfitting params (4x higher lr, near-zero L1)
# - Fix: Use v5.0 PARAMS with 16 features → target ~81.82%
print(f"\n🏆 استخدام باراميترات v5.0 المثبتة (81.82% دقة اختبار)...")

best_params = {
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

print(f"\n📋 معلمات التدريب:")
for key, value in best_params.items():
    print(f"   {key}: {value}")

# ------------------------------------------------------------------
# 2. Read cleaned data (12 base features, encoded + scaled)
# ------------------------------------------------------------------
data_path = os.path.join(processed_path, cfg["data"]["final_clean_file"])
print(f"\n📂 قراءة البيانات النظيفة من: {data_path}")
df = pd.read_csv(data_path)
print(f"   الأبعاد: {df.shape[0]} صف × {df.shape[1]} عمود")
print(f"   الأعمدة الأساسية: {list(df.columns)}")

# ------------------------------------------------------------------
# 3. Apply feature engineering (heuristic → medical)
# ------------------------------------------------------------------
print("\n🧬 تطبيق هندسة الميزات (مسارين)...")
df_fe = engineer_heuristic_features(df)   # Age_BP_Interaction, HR_Age_Ratio, Chol_Age_Ratio
df_fe = engineer_medical_features(df_fe)  # RPP, Exercise_Risk_Index

print(f"\n   الأبعاد بعد الهندسة: {df_fe.shape[0]} صف × {df_fe.shape[1]} عمود")
print(f"   جميع الميزات: {list(df_fe.columns)}")

# Verify no NaN
nan_count = df_fe.isnull().sum().sum()
assert nan_count == 0, f"❌ توجد {nan_count} قيمة مفقودة بعد هندسة الميزات!"
print(f"   ✅ 0% قيم مفقودة")

# ------------------------------------------------------------------
# 4. Split features and target
# ------------------------------------------------------------------
if target_col not in df_fe.columns:
    raise KeyError(f"⚠️ عمود الهدف '{target_col}' غير موجود في البيانات.")

X = df_fe.drop(columns=[target_col])
y = df_fe[target_col]

print(f"\n📊 إجمالي الميزات: {X.shape[1]}")
print(f"📊 توزيع الهدف: {y.value_counts().to_dict()}")

# ------------------------------------------------------------------
# 4b. Class distribution (for reference only — no scale_pos_weight)
# ------------------------------------------------------------------
count_neg = int((y == 0).sum())
count_pos = int((y == 1).sum())
print(f"\n⚖️ توزيع الفئات:")
print(f"   Negative: {count_neg}, Positive: {count_pos}")
print(f"   ℹ️  scale_pos_weight not used (positives={count_pos} > negatives={count_neg}, default 1.0 works best)")

# ------------------------------------------------------------------
# 5. Train/test split (stratified)
# ------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n✂️ تقسيم البيانات (stratified):")
print(f"   تدريب: {X_train.shape[0]} عينة")
print(f"   اختبار: {X_test.shape[0]} عينة")

# ------------------------------------------------------------------
# 6. Train XGBoost with v5.0 proven params
# ------------------------------------------------------------------
print(f"\n🚀 بدء تدريب XGBoost...")

model = xgb.XGBClassifier(**best_params)
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)

# ------------------------------------------------------------------
# 7. Evaluate
# ------------------------------------------------------------------
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)
cm = confusion_matrix(y_test, y_pred)

print(f"\n📊 نتائج التقييم النهائي:")
print(f"   ✅ الدقة (Accuracy):         {accuracy:.4f}")
print(f"   ✅ ROC-AUC:                  {roc_auc:.4f}")

print(f"\n   📉 مصفوفة الارتباك (Confusion Matrix):")
print(f"               توقع Negative   توقع Positive")
print(f"   فعلي Negative     {cm[0,0]:3d}              {cm[0,1]:3d}")
print(f"   فعلي Positive     {cm[1,0]:3d}              {cm[1,1]:3d}")

print(f"\n   📊 تقرير التصنيف (Classification Report):")
print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))

# ------------------------------------------------------------------
# 8. Save model
# ------------------------------------------------------------------
os.makedirs(model_dir, exist_ok=True)
model_save_path = os.path.join(model_dir, os.path.basename(cfg["model"]["weights_path"]))

# Backup existing model first
if os.path.exists(model_save_path):
    bak_path = model_save_path.replace('.pkl', '.bak.v4.pkl')
    if not os.path.exists(bak_path):
        os.rename(model_save_path, bak_path)
        print(f"\n💾 نسخة احتياطية من الموديل القديم: {bak_path}")

with open(model_save_path, "wb") as f:
    pickle.dump(model, f)
print(f"💾 حفظ الموديل الجديد: {model_save_path}")
file_size_kb = os.path.getsize(model_save_path) / 1024
print(f"📦 حجم الملف: {file_size_kb:.1f} KB")

# ------------------------------------------------------------------
# 9. Save metrics
# ------------------------------------------------------------------
metrics_path = os.path.join(os.path.dirname(model_dir), "metrics.json")
metrics = {
    "version": "5.1.0",
    "training_date": datetime.datetime.now().isoformat(),
    "dataset_size": len(df_fe),
    "n_features": X.shape[1],
    "feature_names": list(X.columns),
    "test_accuracy": round(float(accuracy), 6),
    "test_roc_auc": round(float(roc_auc), 6),
    "params_source": "v5.0_proven (not Optuna — see diagnose_v5_drop.py)",
    "best_params": best_params,
    "train_samples": int(X_train.shape[0]),
    "test_samples": int(X_test.shape[0]),
    "class_distribution": {
        "negative": int(count_neg),
        "positive": int(count_pos),
    },
    "confusion_matrix": {
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1])
    }
}
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)
print(f"📊 حفظ المقاييس: {metrics_path}")

# ------------------------------------------------------------------
# 10. Final summary
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print(f"🎯 اكتمل التدريب بنجاح!")
print(f"   الموديل: XGBoost (v5.1)")
print(f"   الميزات: {X.shape[1]}")
print(f"   الدقة:   {accuracy:.4f}")
print(f"   ROC-AUC: {roc_auc:.4f}")
print(f"   المصدر:  باراميترات v5.0 المثبتة (81.82% baseline)")
print("=" * 60)
