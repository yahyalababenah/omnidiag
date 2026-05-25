import pandas as pd
import numpy as np
import os
import torch  # أضفنا استدعاء مكتبة PyTorch المباشر
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

print("🧠 جاري تدريب شبكة TabNet العصبية على الميزات الإحصائية (Heuristic)...")

# 1. قراءة البيانات الفائزة
df = pd.read_csv("data/processed/data_heuristic.csv")
target_col = 'HeartDisease'

# TabNet يتطلب أن تكون البيانات على شكل Numpy Arrays
X = df.drop(columns=[target_col]).values
y = df[target_col].values

# 2. تقسيم البيانات بنفس الطريقة تماماً
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)

# 3. بناء الموديل (تم تنظيف الكود ليكون مباشراً)
tabnet_model = TabNetClassifier(
    n_d=16, n_a=16, n_steps=4,
    gamma=1.3,
    lambda_sparse=1e-3,
    optimizer_params=dict(lr=2e-2),
    scheduler_params={"step_size": 10, "gamma": 0.9},
    scheduler_fn=torch.optim.lr_scheduler.StepLR,  # استخدام آمن ومباشر
    verbose=0,
    seed=42
)

print("⏳ جاري تدريب الطبقات العصبية (قد يستغرق بضع ثوانٍ)...")
tabnet_model.fit(
    X_train=X_train, y_train=y_train,
    eval_set=[(X_valid, y_valid)],
    eval_name=['valid'],
    eval_metric=['accuracy'],
    max_epochs=100,
    patience=15, 
    batch_size=256,
    virtual_batch_size=128
)

# 4. قياس الدقة
preds = tabnet_model.predict(X_test)
acc = accuracy_score(y_test, preds)
print(f"🎯 دقة TabNet الفردية على البيانات المحسنة: {acc:.4f}")

# 5. حفظ الموديل الجديد
save_dir = "models/tabnet_weights"
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "omni_diag_tabnet_heuristic")
tabnet_model.save_model(save_path)
print(f"💾 تم حفظ أوزان TabNet المحسنة في: {save_path}.zip")