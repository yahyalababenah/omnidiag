import pandas as pd
import joblib
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore') # لإخفاء تحذيرات الـ CPU من TabNet

print("🤝 جاري بدء جلسة التشاور الطبي النهائية بين TabNet و XGBoost...")

# تحديد الهدف بشكل صريح وآمن
target_col = 'HeartDisease'

# =========================================
# 1. تجهيز بيانات TabNet (الأساسية)
# =========================================
df_base = pd.read_csv("data/processed/final_ready_data.csv")
X_base = df_base.drop(columns=[target_col]).values # TabNet يحتاج Numpy Array
y_base = df_base[target_col].values
_, X_test_base, _, y_test = train_test_split(X_base, y_base, test_size=0.2, random_state=42)

# =========================================
# 2. تجهيز بيانات XGBoost (الميزات المحسنة)
# =========================================
df_heuristic = pd.read_csv("data/processed/data_heuristic.csv")
# XGBoost يفضل DataFrame ليطابق أسماء الأعمدة بدقة
X_heuristic = df_heuristic.drop(columns=[target_col]) 
_, X_test_heuristic, _, _ = train_test_split(X_heuristic, y_base, test_size=0.2, random_state=42)

# =========================================
# 3. تحميل الأوزان والموديلات
# =========================================
tabnet_model = TabNetClassifier()
tabnet_model.load_model("models/tabnet_weights/omni_diag_model.zip")

xgb_model = joblib.load("models/omni_diag_xgb_optimized.pkl") 

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