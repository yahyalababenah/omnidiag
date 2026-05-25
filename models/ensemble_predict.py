import pandas as pd
import joblib
import os
import sys
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore') # لإخفاء تحذيرات الـ CPU من TabNet

# Add project root to path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config_loader import load_config, resolve_path

print("🤝 جاري بدء جلسة التشاور الطبي النهائية بين TabNet و XGBoost...")

# Load config for config-driven paths
cfg = load_config("heart_disease")
processed_path = resolve_path(cfg, "data", "processed_path")
model_path = resolve_path(cfg, "model", "weights_path")
target_col = cfg["disease"]["target_column"]
tabnet_weights_dir = os.path.join(
    os.path.dirname(os.path.dirname(resolve_path(cfg, "model", "weights_path"))),
    "tabnet_weights"
)

# =========================================
# 1. تجهيز بيانات TabNet (الأساسية)
# =========================================
base_data_path = os.path.join(processed_path, cfg["data"]["final_clean_file"])
print(f"📂 قراءة بيانات TabNet من: {base_data_path}")
df_base = pd.read_csv(base_data_path)
X_base = df_base.drop(columns=[target_col]).values # TabNet يحتاج Numpy Array
y_base = df_base[target_col].values
_, X_test_base, _, y_test = train_test_split(X_base, y_base, test_size=0.2, random_state=42)

# =========================================
# 2. تجهيز بيانات XGBoost (الميزات المحسنة)
# =========================================
heuristic_data_path = os.path.join(processed_path, cfg["data"]["heuristic_file"])
print(f"📂 قراءة بيانات XGBoost من: {heuristic_data_path}")
df_heuristic = pd.read_csv(heuristic_data_path)
# XGBoost يفضل DataFrame ليطابق أسماء الأعمدة بدقة
X_heuristic = df_heuristic.drop(columns=[target_col])
_, X_test_heuristic, _, _ = train_test_split(X_heuristic, y_base, test_size=0.2, random_state=42)

# =========================================
# 3. تحميل الأوزان والموديلات
# =========================================
tabnet_zip_path = os.path.join(tabnet_weights_dir, "omni_diag_model.zip")
print(f"📂 تحميل TabNet من: {tabnet_zip_path}")
tabnet_model = TabNetClassifier()
tabnet_model.load_model(tabnet_zip_path)

print(f"📂 تحميل XGBoost من: {model_path}")
xgb_model = joblib.load(model_path)

# =========================================
# 4. دمج القرارات (Soft Voting)
# =========================================
prob_tabnet = tabnet_model.predict_proba(X_test_base)[:, 1]
prob_xgb = xgb_model.predict_proba(X_test_heuristic)[:, 1]

# إعطاء XGBoost وزناً أكبر (60%) لأنه الأقوى حالياً
final_prob = (prob_tabnet * 0.40) + (prob_xgb * 0.60)
final_preds = (final_prob > 0.5).astype(int)

# 5. طباعة النتيجة الحاسمة
acc = accuracy_score(y_test, final_preds)
print(f"🎯 دقة نظام التحالف النهائية (Ensemble Accuracy): {acc:.4f}")