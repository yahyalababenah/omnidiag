import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import os
import sys

# Add project root to path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config_loader import load_config, resolve_path

print("🚀 بدء تشغيل نظام OmniDiag للفحص السريري المباشر...")

# Load config for config-driven paths
cfg = load_config("heart_disease")
processed_path = resolve_path(cfg, "data", "processed_path")
model_path = resolve_path(cfg, "model", "weights_path")
target_col = cfg["disease"]["target_column"]

# 1. تحميل البيانات والموديل الذهبي
data_path = os.path.join(processed_path, cfg["data"]["heuristic_file"])
print(f"📂 قراءة البيانات من: {data_path}")
print(f"📂 تحميل الموديل من: {model_path}")
df = pd.read_csv(data_path)

# نختار مريضاً محدداً من قاعدة البيانات لاختباره (يمكنك تغيير الرقم لتجربة مرضى آخرين)
patient_index = 10
patient_data = df.drop(columns=[target_col]).iloc[[patient_index]]
actual_status = df[target_col].iloc[patient_index]

model = joblib.load(model_path)

# 2. إجراء التشخيص
prediction = model.predict(patient_data)[0]
probability = model.predict_proba(patient_data)[0][1]

# 3. عرض التقرير الطبي في الـ Terminal
print("\n" + "=" * 50)
print(f"👨‍⚕️ التقرير الطبي للمريض رقم [{patient_index}]:")
print("=" * 50)
# طباعة أبرز 5 ميزات للمريض
print(patient_data.iloc[:, :5].to_string(index=False)) 
print("-" * 50)
print(f"🩺 الحالة الطبية الحقيقية المسجلة: {'مصاب بأمراض القلب 💔' if actual_status == 1 else 'سليم ومعافى 💚'}")
print(f"🤖 تشخيص OmniDiag النهائي: {'مصاب 💔' if prediction == 1 else 'سليم 💚'}")
print(f"📊 نسبة ثقة الموديل في القرار: {probability * 100:.1f}%")
print("=" * 50)

# 4. التفسير الطبي الذكي (XAI - SHAP Waterfall)
print("\n🔍 جاري توليد التفسير الطبي (SHAP Waterfall Plot)...")
print("يرجى التحقق من النافذة المنبثقة لرؤية كيف تم اتخاذ القرار.")

explainer = shap.TreeExplainer(model)
shap_values = explainer(patient_data)

# إعداد الرسم البياني ليظهر بشكل احترافي
plt.figure(figsize=(10, 6))
plt.title(f"OmniDiag XAI Explanation - Patient {patient_index}", fontsize=14, pad=20)
shap.waterfall_plot(shap_values[0])