import pandas as pd
import joblib
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

print("🤝 جاري بدء جلسة التشاور الطبي بين TabNet و XGBoost...")

# 1. تحميل البيانات
df = pd.read_csv("data/processed/final_ready_data.csv")
X = df.drop(columns=['HeartDisease']).values
y = df['HeartDisease'].values
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. تحميل الموديلات
tabnet_model = TabNetClassifier()
tabnet_model.load_model("models/tabnet_weights/omni_diag_model.zip")

xgb_model = joblib.load("models/xgboost_weights/omni_diag_xgb_optimized.pkl") # نستخدم الموديل المحسن هنا

# 3. الحصول على احتمالات الخطر (Soft Voting)
# أخذ احتمالية أن المريض مصاب (العمود 1)
prob_tabnet = tabnet_model.predict_proba(X_test)[:, 1]
prob_xgb = xgb_model.predict_proba(X_test)[:, 1]

# 4. دمج القرارات (إعطاء وزن أعلى قليلاً لـ XGBoost لأنه أثبت دقة أعلى)
final_prob = (prob_tabnet * 0.45) + (prob_xgb * 0.55)

# تحويل الاحتمالية إلى قرار نهائي (1 مصاب، 0 سليم)
final_preds = (final_prob > 0.5).astype(int)

acc = accuracy_score(y_test, final_preds)
print(f"🎯 دقة نظام التحالف (Ensemble Accuracy): {acc:.4f}")