import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import logging

logger = logging.getLogger(__name__)

def get_preprocessor() -> ColumnTransformer:
    """
    يبني Pipeline موحد لمعالجة البيانات الرقمية والنصية.
    هذا هو "السورس الوحيد" للمعالجة في المشروع بأكمله.
    """
    # 1. تحديد الأعمدة بناءً على نوعها
    numeric_features = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']
    categorical_features = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']

    # 2. معالجة الأرقام: سد الفراغات بالمتوسط (إن وجدت) ثم التقييس
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # 3. معالجة النصوص: سد الفراغات ثم تحويلها لأرقام (One-Hot Encoding)
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # 4. دمج المعالجين في محول واحد
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor

if __name__ == "__main__":
    # تجربة سريعة للتأكد من عمل الـ Pipeline
    import sys
    from pathlib import Path
    
    # إضافة مسار المشروع للتمكن من استدعاء dataset
    root_dir = Path(__file__).resolve().parents[2]
    sys.path.append(str(root_dir))
    
    from src.heart_disease.dataset import load_raw_data
    
    try:
        # تحميل البيانات
        df = load_raw_data()
        X = df.drop(columns=['HeartDisease'])
        
        # استدعاء وتطبيق المعالجة
        preprocessor = get_preprocessor()
        X_processed = preprocessor.fit_transform(X)
        
        print("\n--- نجاح المعالجة! ---")
        print(f"شكل البيانات قبل المعالجة: {X.shape}")
        print(f"شكل البيانات بعد المعالجة (بسبب One-Hot Encoding): {X_processed.shape}")
        
    except Exception as e:
        print(f"حدث خطأ: {e}")