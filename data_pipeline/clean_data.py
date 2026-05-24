import pandas as pd
import numpy as np
import joblib
import os
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler

def run_full_preprocessing_pipeline():
    print("🚀 بدء خط الإنتاج الشامل لتجهيز البيانات (Data Preprocessing Pipeline)...")
    
    # 1. قراءة البيانات المدمجة
    df = pd.read_csv("data/processed/merged_heart_data.csv")
    
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
    os.makedirs("models/preprocessors", exist_ok=True)
    
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    
    # حفظ الـ Encoders لاستخدامها لاحقاً في FastAPI
    joblib.dump(label_encoders, "models/preprocessors/label_encoders.pkl")
    
    # ==========================================
    # المرحلة الثالثة: التحجيم (Feature Scaling)
    # ==========================================
    print("⏳ المرحلة 3: تحجيم البيانات الرقمية (Standard Scaling)...")
    numerical_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
    
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    # حفظ الـ Scaler لاستخدامه لاحقاً في واجهة النظام
    joblib.dump(scaler, "models/preprocessors/standard_scaler.pkl")
    
    # ==========================================
    # الحفظ النهائي للبيانات الجاهزة للتدريب
    # ==========================================
    final_path = "data/processed/final_ready_data.csv"
    df.to_csv(final_path, index=False)
    
    print(f"\n✅ اكتمل خط الإنتاج بنجاح!")
    print(f"📊 تم حفظ البيانات النهائية الجاهزة للـ TabNet في: {final_path}")
    print(f"🛠️ تم حفظ أدوات التحجيم والترميز في مجلد: models/preprocessors/")

if __name__ == "__main__":
    run_full_preprocessing_pipeline()