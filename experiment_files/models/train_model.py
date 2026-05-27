import pandas as pd
import numpy as np
import torch
import os
import sys
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import mlflow

# Add project root to path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config_loader import load_config, resolve_path

def run_tabnet_training():
    print("🚀 بدء تشغيل محرك TabNet...")
    
    # Load config for config-driven paths
    cfg = load_config("heart_disease")
    processed_path = resolve_path(cfg, "data", "processed_path")
    target_col = cfg["disease"]["target_column"]
    
    # 1. قراءة البيانات الجاهزة
    data_path = os.path.join(processed_path, cfg["data"]["final_clean_file"])
    print(f"📂 قراءة البيانات من: {data_path}")
    df = pd.read_csv(data_path)
    
    # فصل المميزات (X) عن الهدف (y)
    X = df.drop(columns=[target_col]).values
    y = df[target_col].values
    
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
        tabnet_weights_dir = os.path.join(
            os.path.dirname(os.path.dirname(resolve_path(cfg, "model", "weights_path"))),
            "tabnet_weights"
        )
        os.makedirs(tabnet_weights_dir, exist_ok=True)
        save_path = os.path.join(tabnet_weights_dir, "omni_diag_model")
        clf.save_model(save_path)
        
        print(f"💾 تم حفظ أوزان الموديل بنجاح في: {save_path}.zip")

if __name__ == "__main__":
    run_tabnet_training()