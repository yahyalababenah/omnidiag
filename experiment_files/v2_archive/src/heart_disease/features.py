import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import KNNImputer

def get_preprocessor() -> ColumnTransformer:
    """
    يجهز معالج البيانات (Pipeline) الذي يقوم بـ:
    1. ملء القيم المفقودة (KNN Imputer)
    2. معالجة النصوص (OneHotEncoder)
    3. توحيد المقاييس (StandardScaler)
    """
    
    # تحديد الأعمدة بناءً على طبيعتها
    numeric_features = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']
    categorical_features = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']

    # معالجة البيانات الرقمية
    numeric_transformer = Pipeline(steps=[
        ('imputer', KNNImputer(n_neighbors=5)),
        ('scaler', StandardScaler())
    ])

    # معالجة البيانات النصية
    categorical_transformer = Pipeline(steps=[
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    # تجميعهم
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor

def process_features(X: pd.DataFrame, preprocessor: ColumnTransformer) -> pd.DataFrame:
    """
    تطبيق الـ preprocessor على البيانات (X).
    """
    X_processed = preprocessor.fit_transform(X)
    return X_processed