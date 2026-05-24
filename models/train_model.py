import pandas as pd
import numpy as np
import torch
import os
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import mlflow

def run_tabnet_training():
    print("🚀 بدء تشغيل محرك TabNet...")
    
    # 1. قراءة البيانات الجاهزة
    df = pd.read_csv("data/processed/final_ready_data.csv")
    
    # فصل المميزات (X) عن الهدف (y)
    # عمود HeartDisease هو الذي نتوقع منه خطر الإصابة
    X = df.drop(columns=['HeartDisease']).values
    y = df['HeartDisease'].values
    
    # تقسيم البيانات (80% تدريب - 20% اختبار)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 2. إعداد مختبر MLflow للتتبع
    mlflow.set_tracking_uri("sqlite:///mlruns.db") # حفظ النتائج في قاعدة بيانات محلية
    mlflow.set_experiment("OmniDiag_TabNet_Experiment")
    
    with mlflow.start_run():
        print("🧠 جاري تدريب الشبكة العصبية...")
        
        # 3. بناء معمارية TabNet
        clf = TabNetClassifier(
            n_d=8, n_a=8, n_steps=3,
            gamma=1.3, n_independent=2, n_shared=2,
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=2e-2),
            scheduler_params={"step_size":10, "gamma":0.9},
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            mask_type='entmax' # نوع الانتباه (Attention) للشفافية الطبية
        )
        
        # التدريب
        clf.fit(
            X_train=X_train, y_train=y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            eval_name=['train', 'valid'],
            eval_metric=['accuracy'],
            max_epochs=100,
            patience=20, # الإيقاف المبكر إذا لم يتحسن الموديل
            batch_size=256, virtual_batch_size=128
        )
        
        # 4. التقييم واستخراج المقاييس
        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)
        
        print(f"\n✅ اكتمل التدريب! دقة النظام (Accuracy): {acc:.4f}")
        
        # 5. التوثيق التلقائي في MLflow
        mlflow.log_param("max_epochs", 100)
        mlflow.log_param("architecture", "PyTorch TabNet")
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        
        # 6. حفظ أوزان الموديل للمستقبل
        os.makedirs("models/tabnet_weights", exist_ok=True)
        save_path = "models/tabnet_weights/omni_diag_model"
        clf.save_model(save_path)
        
        print(f"💾 تم حفظ أوزان الموديل بنجاح في: {save_path}.zip")

if __name__ == "__main__":
    run_tabnet_training()