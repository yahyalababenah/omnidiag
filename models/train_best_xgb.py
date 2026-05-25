import pandas as pd
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

print("🚀 جاري تدريب وحفظ XGBoost بالإعدادات الذهبية...")

# 1. جلب البيانات
df = pd.read_csv("data/processed/final_ready_data.csv")
X = df.drop(columns=['HeartDisease']).values
y = df['HeartDisease'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. الإعدادات الذهبية من Optuna
best_params = {
    'n_estimators': 536, 
    'max_depth': 4, 
    'learning_rate': 0.0167535696123816, 
    'subsample': 0.5039562178974109, 
    'colsample_bytree': 0.6507129420569848
}

# 3. التدريب والحفظ
clf = xgb.XGBClassifier(**best_params, random_state=42, eval_metric='logloss')
clf.fit(X_train, y_train)

preds = clf.predict(X_test)
print(f"✅ الدقة المؤكدة: {accuracy_score(y_test, preds):.4f}")

joblib.dump(clf, "models/xgboost_weights/omni_diag_xgb_optimized.pkl")
print("💾 تم حفظ الموديل المحسن بنجاح!")