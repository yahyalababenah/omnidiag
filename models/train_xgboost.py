import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import mlflow
import joblib
import os

def run_xgboost_training():
    print("🚀 بدء تشغيل محرك XGBoost...")
    
    # 1. قراءة البيانات
    df = pd.read_csv("data/processed/final_ready_data.csv")
    X = df.drop(columns=['HeartDisease']).values
    y = df['HeartDisease'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 2. إعداد MLflow 
    # سنضعه في نفس الـ Experiment لكي نقارنه مع TabNet وجهاً لوجه!
    mlflow.set_tracking_uri("sqlite:///mlruns.db")
    mlflow.set_experiment("OmniDiag_Models_Comparison")
    
    with mlflow.start_run(run_name="XGBoost_Medical_Model"):
        print("🧠 جاري تدريب خوارزمية XGBoost...")
        
        # 3. بناء معمارية XGBoost
        clf = xgb.XGBClassifier(
            n_estimators=200,          # عدد الأشجار
            max_depth=5,               # عمق الشجرة (قليل لمنع الـ Overfitting)
            learning_rate=0.05,        # سرعة التعلم
            random_state=42,
            eval_metric='logloss'
        )
        
        # التدريب
        clf.fit(X_train, y_train)
        
        # 4. التقييم واستخراج المقاييس
        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)
        
        print(f"\n✅ اكتمل التدريب! دقة النظام (Accuracy): {acc:.4f}")
        
        # 5. التوثيق في MLflow
        mlflow.log_param("architecture", "XGBoost")
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        
        # 6. حفظ الموديل
        os.makedirs("models/xgboost_weights", exist_ok=True)
        save_path = "models/xgboost_weights/omni_diag_xgb.pkl"
        joblib.dump(clf, save_path)
        
        print(f"💾 تم حفظ أوزان الموديل بنجاح في: {save_path}")

if __name__ == "__main__":
    run_xgboost_training()