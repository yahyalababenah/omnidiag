import pandas as pd
import numpy as np
import joblib
import os
import sys
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Add project root to path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from configs.config_loader import load_config, resolve_path

def run_full_preprocessing_pipeline(disease_name: str = "heart_disease"):
    print("🚀 بدء خط الإنتاج الشامل لتجهيز البيانات (Data Preprocessing Pipeline)...")
    
    # Load config for config-driven paths
    cfg = load_config(disease_name)
    processed_path = resolve_path(cfg, "data", "processed_path")
    models_path = resolve_path(cfg, "model", "preprocessors_path")
    
    # 1. قراءة البيانات المدمجة
    merged_path = os.path.join(processed_path, cfg["data"]["merged_file"])
    print(f"📂 قراءة البيانات من: {merged_path}")
    df = pd.read_csv(merged_path)
    
    # ==========================================
    # المرحلة الأولى: الاستيفاء الشفاف (MissForest)
    # ==========================================
    print("⏳ المرحلة 1: معالجة البيانات المفقودة...")
    df['Cholesterol'] = df['Cholesterol'].replace(0, np.nan)
    df['RestingBP'] = df['RestingBP'].replace(0, np.nan)
    
    df_temp = df.copy()
    df_temp['Sex_Num'] = df_temp['Sex'].map({'M': 1, 'F': 0})
    
    features_for_imputation = ['Age', 'Sex_Num', 'RestingBP', 'MaxHR', 'Cholesterol']
    impute_data = df_temp[features_for_imputation]
    
    rf_estimator = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=5)
    imputer = IterativeImputer(estimator=rf_estimator, random_state=42, max_iter=10)
    
    imputed_values = imputer.fit_transform(impute_data)
    df_temp[features_for_imputation] = imputed_values
    
    df['Cholesterol'] = df_temp['Cholesterol'].round().astype(int)
    df['RestingBP'] = df_temp['RestingBP'].round().astype(int)
    
    # ==========================================
    # المرحلة الثانية: الترميز (Categorical Encoding)
    # ==========================================
    print("⏳ المرحلة 2: ترميز النصوص إلى أرقام (Label Encoding)...")
    categorical_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
    
    # إنشاء مجلد لحفظ المعالجات إذا لم يكن موجوداً
    os.makedirs(models_path, exist_ok=True)
    
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    
    # حفظ الـ Encoders لاستخدامها لاحقاً في FastAPI
    encoders_path = os.path.join(models_path, "label_encoders.pkl")
    joblib.dump(label_encoders, encoders_path)
    print(f"🛠️ تم حفظ الترميزات في: {encoders_path}")
    
    # ==========================================
    # المرحلة الثالثة: التحجيم (Feature Scaling)
    # ==========================================
    print("⏳ المرحلة 3: تحجيم البيانات الرقمية (Standard Scaling)...")
    numerical_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
    
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    # حفظ الـ Scaler لاستخدامه لاحقاً في واجهة النظام
    scaler_path = os.path.join(models_path, "standard_scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"🛠️ تم حفظ المحجم في: {scaler_path}")
    
    # ==========================================
    # الحفظ النهائي للبيانات الجاهزة للتدريب
    # ==========================================
    final_path = os.path.join(processed_path, cfg["data"]["final_clean_file"])
    df.to_csv(final_path, index=False)
    
    print(f"\n✅ اكتمل خط الإنتاج بنجاح!")
    print(f"📊 تم حفظ البيانات النهائية الجاهزة للتدريب في: {final_path}")
    print(f"🛠️ تم حفظ أدوات التحجيم والترميز في مجلد: {models_path}")

if __name__ == "__main__":
    run_full_preprocessing_pipeline()