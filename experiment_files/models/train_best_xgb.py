import pandas as pd
import xgboost as xgb
import pickle
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Add project root to path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config_loader import load_config, resolve_path

print("🚀 جاري بدء تدريب وحفظ XGBoost بالإعدادات الذهبية المكتشفة...")

# Load config for config-driven paths
cfg = load_config("heart_disease")
processed_path = resolve_path(cfg, "data", "processed_path")
model_weights_dir = os.path.dirname(resolve_path(cfg, "model", "weights_path"))
target_col = cfg["disease"]["target_column"]

# 1. قراءة البيانات الفائزة في تجربة الـ A/B Testing
data_path = os.path.join(processed_path, cfg["data"]["heuristic_file"])
if not os.path.exists(data_path):
    raise FileNotFoundError(f"⚠️ لم يتم العثور على ملف البيانات: {data_path}. تأكد من تشغيل سكريبت هندسة الميزات أولاً.")

print(f"📂 قراءة البيانات من: {data_path}")
df = pd.read_csv(data_path)

# 2. فصل الميزات عن عمود الهدف (Target) بشكل صريح ومباشر
if target_col not in df.columns:
    raise KeyError(f"⚠️ لم يتم العثور على عمود الهدف '{target_col}' في البيانات. تأكد من اسم العمود.")

X = df.drop(columns=[target_col])
y = df[target_col]

# 3. تقسيم البيانات إلى مجموعات تدريب واختبار
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. تثبيت الإعدادات الذهبية المستخرجة من Trial 83 في Optuna
best_params = {
    'n_estimators': 898,
    'max_depth': 5,
    'learning_rate': 0.013594126498405943,
    'subsample': 0.963733407970185,
    'colsample_bytree': 0.5454718650965412,
    'random_state': 42,
    'eval_metric': 'logloss'
}

# 5. بناء وتدريب النموذج النهائي
best_model = xgb.XGBClassifier(**best_params)
best_model.fit(X_train, y_train)

# 6. التحقق النهائي من الدقة على بيانات الاختبار
y_pred = best_model.predict(X_test)
final_accuracy = accuracy_score(y_test, y_pred)
print(f"🎯 الدقة المؤكدة للموديل الجديد في بيئة الاختبار: {final_accuracy:.4f}")

# 7. حفظ الموديل المحسن بصيغة pickle لاستخدامه في نظام التحالف (Ensemble)
os.makedirs(model_weights_dir, exist_ok=True)
model_save_path = os.path.join(model_weights_dir, os.path.basename(cfg["model"]["weights_path"]))

with open(model_save_path, "wb") as f:
    pickle.dump(best_model, f)

print(f"💾 تم حفظ الموديل المحسن بنجاح في المسار: {model_save_path}")