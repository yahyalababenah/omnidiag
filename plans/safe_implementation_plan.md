# خطة التعديل الآمنة على المشروع
## Safe Implementation Plan — No Breaking Changes

> **المبدأ:** كل تعديل يحافظ على backward compatibility. الملفات القديمة تبقى كما هي إلا إذا استبدلناها بشكل صريح. النسخ الاحتياطية تُنشأ قبل أي تغيير.

---

## ⚠️ إجراءات الأمان قبل البدء

```bash
# 1. إنشاء نسخة احتياطية من الـ Raw Data
cp data/heart_disease/raw/heart.csv data/heart_disease/raw/heart.csv.bak

# 2. إنشاء نسخة احتياطية من الملفات التي سنعدلها
cp experiment_files/data_pipeline/merge_data.py experiment_files/data_pipeline/merge_data.py.bak
cp experiment_files/data_pipeline/clean_data.py experiment_files/data_pipeline/clean_data.py.bak
cp experiment_files/models/advanced_feature_engineering.py experiment_files/models/advanced_feature_engineering.py.bak
cp features/heart_disease_features.py features/heart_disease_features.py.bak
cp configs/heart_disease.yaml configs/heart_disease.yaml.bak
```

---

## قائمة الملفات التي ستنشأ أو تُعدّل

| # | الملف | الإجراء | نوع التغيير |
|---|------|---------|------------|
| 1 | `experiment_files/data_pipeline/transform_heart11.py` | **إنشاء** | جديد كلياً |
| 2 | `experiment_files/data_pipeline/merge_data.py` | **تعديل** | محدود (separator + Friedewald) |
| 3 | `experiment_files/models/advanced_feature_engineering.py` | **تعديل** | إضافة ميزتين جديدتين |
| 4 | `features/heart_disease_features.py` | **تعديل** | إضافة استدعاء الميزتين الجديدتين |
| 5 | `experiment_files/data_pipeline/clean_data.py` | **تعديل** | تحديث قائمة الميزات للتحجيم |
| 6 | `configs/heart_disease.yaml` | **تعديل** | تحديث الوصف + الميزات |
| 7 | `backend/schemas.py` | **تعديل** | تحديث docstring فقط |
| 8 | `backend/model_loader.py` | **تعديل** | إضافة حساب الميزات الرياضية تلقائياً |

---

## الملف 1: [`experiment_files/data_pipeline/transform_heart11.py`](experiment_files/data_pipeline/)
### جديد — إنشاء سكربت تحويل heart11.csv

**الوظيفة:** تحويل heart11.csv من 14 عموداً إلى 12 عموداً قياسياً (بإسقاط `ca` و `thal`)

```python
import pandas as pd
import os
import shutil

# ==========================================
# 1. نسخ احتياطي للـ heart.csv الحالي
# ==========================================
raw_dir = "data/heart_disease/raw"
backup_path = os.path.join(raw_dir, "heart.csv.bak")
original_path = os.path.join(raw_dir, "heart.csv")
if os.path.exists(original_path) and not os.path.exists(backup_path):
    shutil.copy2(original_path, backup_path)
    print(f"✅ Backup created: {backup_path}")

# ==========================================
# 2. قراءة heart11.csv
# ==========================================
df = pd.read_csv("heart11.csv")
print(f"📂 Loaded heart11.csv: {df.shape[0]} rows, {df.shape[1]} cols")

# ==========================================
# 3. إسقاط الأعمدة غير المرغوب فيها
# ==========================================
# ca → Data Leakage (Angiography result)
# thal → Thallium Stress Test (expensive, non-invasive triage focus)
df = df.drop(columns=["ca", "thal"])
print(f"✅ Dropped 'ca' and 'thal'. Now: {df.shape[1]} cols")

# ==========================================
# 4. إعادة تسمية الأعمدة (14 → 12)
# ==========================================
column_mapping = {
    "age": "Age",
    "sex": "Sex",
    "cp": "ChestPainType",
    "trestbps": "RestingBP",
    "chol": "Cholesterol",
    "fbs": "FastingBS",
    "restecg": "RestingECG",
    "thalach": "MaxHR",
    "exang": "ExerciseAngina",
    "oldpeak": "Oldpeak",
    "slope": "ST_Slope",
    "target": "HeartDisease",
}
df = df.rename(columns=column_mapping)

# ==========================================
# 5. تحويل الترميزات الرقمية إلى نصوص
# ==========================================
# Sex: 0→F, 1→M
df["Sex"] = df["Sex"].map({0: "F", 1: "M"})

# ChestPainType: 0→TA, 1→ATA, 2→NAP, 3→ASY
cp_map = {0: "TA", 1: "ATA", 2: "NAP", 3: "ASY"}
df["ChestPainType"] = df["ChestPainType"].map(cp_map)

# RestingECG: 0→Normal, 1→ST, 2→LVH
ecg_map = {0: "Normal", 1: "ST", 2: "LVH"}
df["RestingECG"] = df["RestingECG"].map(ecg_map)

# ExerciseAngina: 0→N, 1→Y
df["ExerciseAngina"] = df["ExerciseAngina"].map({0: "N", 1: "Y"})

# ST_Slope: 0→Up, 1→Flat, 2→Down
slope_map = {0: "Up", 1: "Flat", 2: "Down"}
df["ST_Slope"] = df["ST_Slope"].map(slope_map)

# ==========================================
# 6. حذف الصفوف المكررة
# ==========================================
before = len(df)
df = df.drop_duplicates()
print(f"✅ Removed {before - len(df)} duplicate rows")

# ==========================================
# 7. ترتيب الأعمدة حسب القالب الذهبي
# ==========================================
expected_columns = [
    "Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol",
    "FastingBS", "RestingECG", "MaxHR", "ExerciseAngina",
    "Oldpeak", "ST_Slope", "HeartDisease",
]
df = df[expected_columns]

# ==========================================
# 8. الحفظ
# ==========================================
df.to_csv(original_path, index=False)
print(f"✅ Saved transformed data to {original_path}")
print(f"📊 Final shape: {df.shape[0]} rows × {df.shape[1]} cols")
print(f"📊 Value counts:\n{df['HeartDisease'].value_counts()}")
```

**التحقق:** `python experiment_files/data_pipeline/transform_heart11.py`

---

## الملف 2: [`experiment_files/data_pipeline/merge_data.py`](experiment_files/data_pipeline/merge_data.py)
### تعديل — 3 تغييرات محدودة

**التغيير 1:** تغيير فاصل القراءة من tab إلى comma (لأن heart.csv الآن CSV)

| السطر الحالي | السطر الجديد |
|-------------|-------------|
| `old_data = pd.read_csv(old_data_path, sep='\t')` | `old_data = pd.read_csv(old_data_path, sep=',')` |

**التغيير 2:** تعديل حساب Cholesterol في Z-Alizadeh باستخدام Friedewald

| السطر الحالي | السطر الجديد |
|-------------|-------------|
| `df_new['Cholesterol'] = np.round(new_data['LDL'] + (new_data['TG'] / 5.0))` | `df_new['Cholesterol'] = np.round(new_data['LDL'] + new_data['HDL'] + (new_data['TG'] / 5.0))` |

**التغيير 3:** إضافة سطر تحقق بعد الدمج

أضف بعد سطر `final_df = pd.concat([old_data, df_new], axis=0, ignore_index=True)`:
```python
# التأكد من عدم وجود أعمدة NaN بعد الدمج (جميع الأعمدة الـ 12 موجودة في المصدرين)
assert final_df.isnull().sum().sum() == 0, "⚠️ NaN values detected after merge!"
```

---

## الملف 3: [`experiment_files/models/advanced_feature_engineering.py`](experiment_files/models/advanced_feature_engineering.py)
### تعديل — إضافة ميزتين جديدتين

**أضف بعد الدالة `engineer_clinical_features`:**

```python
# ==========================================
# المسار الثالث: الميزات الطبية المتقدمة (Medical)
# ==========================================
def engineer_medical_features(df):
    """
    Generate advanced medical features for CAD prediction.
    
    Features created:
        1. RPP (Rate-Pressure Product): RestingBP × MaxHR
           يقيس استهلاك عضلة القلب للأكسجين — معيار عالمي.
        2. Exercise_Risk_Index: Oldpeak × ExerciseAngina (encoded)
           إذا ألم + انخفاض ST → تأكيد CAD.
    
    Args:
        df: DataFrame with base features.
    
    Returns:
        DataFrame with medical features appended.
    """
    df = df.copy()
    
    # RPP — Rate-Pressure Product (Double Product)
    if 'RestingBP' in df.columns and 'MaxHR' in df.columns:
        df['RPP'] = df['RestingBP'] * df['MaxHR']
    
    # Exercise Risk Index
    if 'Oldpeak' in df.columns and 'ExerciseAngina' in df.columns:
        # ExerciseAngina is 'Y'/'N' → encode to 1/0
        exang_num = df['ExerciseAngina'].map({'Y': 1, 'N': 0})
        df['Exercise_Risk_Index'] = df['Oldpeak'] * exang_num
    
    return df
```

**ثم عدّل `__main__` لإضافة المسار الطبي الجديد:**

أضف في نهاية `__main__`:
```python
# المسار الثالث: الميزات الطبية المتقدمة (Medical)
df_medical = engineer_medical_features(df)
medical_path = os.path.join(processed_path, "data_medical.csv")
df_medical.to_csv(medical_path, index=False)
print(f"💾 تم حفظ الميزات الطبية في: {medical_path}")
```

---

## الملف 4: [`features/heart_disease_features.py`](features/heart_disease_features.py)
### تعديل — إضافة `engineer_medical` method

**أضف import جديد:**
```python
from models.advanced_feature_engineering import (
    engineer_heuristic_features,
    engineer_clinical_features,
    engineer_medical_features,  # NEW
)
```

**أضف method جديد في `HeartDiseaseFeatureEngineer`:**

```python
def engineer_medical(self, df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate advanced medical features for CAD risk assessment.
    
    Delegates to models.advanced_feature_engineering.engineer_medical_features().
    
    Features created:
        - RPP (Rate-Pressure Product): RestingBP × MaxHR.
        - Exercise_Risk_Index: Oldpeak × ExerciseAngina (encoded).
    
    Args:
        df: DataFrame with at least 'RestingBP', 'MaxHR', 'Oldpeak', 'ExerciseAngina'.
    
    Returns:
        DataFrame with medical features appended.
    """
    return engineer_medical_features(df)
```

---

## الملف 5: [`experiment_files/data_pipeline/clean_data.py`](experiment_files/data_pipeline/clean_data.py)
### تعديل — تحديث قوائم الميزات لتشمل الميزات الجديدة

**التغيير 1:** أضف `'RPP'` و `'Exercise_Risk_Index'` إلى قائمة `numerical_cols`:

```python
numerical_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak', 'RPP', 'Exercise_Risk_Index']
```

**التغيير 2:** أضف `'RPP'` و `'Exercise_Risk_Index'` إلى `features_for_imputation`:

```python
features_for_imputation = ['Age', 'Sex_Num', 'RestingBP', 'MaxHR', 'Cholesterol', 'RPP', 'Exercise_Risk_Index']
```

---

## الملف 6: [`configs/heart_disease.yaml`](configs/heart_disease.yaml)
### تعديل — تحديث الوصف والميزات

**التغيير 1:** تحديث display_name والوصف:

```yaml
disease:
  name: "heart_disease"
  display_name: "Coronary Artery Disease Risk"
  description: "Non-invasive early triage for Coronary Artery Disease (CAD) using XGBoost with SHAP explainability and advanced medical feature engineering"
  target_column: "HeartDisease"
  version: "5.0.0"
```

**التغيير 2:** تحديث `version` في الـ api:
```yaml
api:
  endpoint_prefix: "/api/v5"
  version: "5.0.0"
```

**التغيير 3:** إضافة الميزات الجديدة إلى `heuristic_features`:

```yaml
  heuristic_features:
    - name: "Age_BP_Interaction"
      formula: "Age * RestingBP"
      description: "Multiplicative interaction between age and resting blood pressure"
    - name: "HR_Age_Ratio"
      formula: "MaxHR / Age"
      description: "Ratio of maximum heart rate to age (fitness indicator)"
    - name: "Chol_Age_Ratio"
      formula: "Cholesterol / Age"
      description: "Ratio of cholesterol to age (risk normalization)"
    - name: "RPP"                              # NEW
      formula: "RestingBP * MaxHR"
      description: "Rate-Pressure Product — myocardial oxygen consumption indicator"
    - name: "Exercise_Risk_Index"              # NEW
      formula: "Oldpeak * ExerciseAngina"
      description: "Composite risk index combining ST depression with exercise angina"
```

---

## الملف 7: [`backend/schemas.py`](backend/schemas.py)
### تعديل — تحديث Docstring فقط

```python
class HeartDiseaseInput(BaseModel):
    """
    Patient input schema for Coronary Artery Disease (CAD) risk assessment.
    
    All 11 fields correspond to the clinical features used in the
    UCI Heart Disease (Cleveland) and Z-Alizadeh Sani harmonized datasets.
    The backend automatically computes 4 engineered medical features
    (RPP, Age_BP_Interaction, Chol_Age_Ratio, Exercise_Risk_Index)
    before model inference.
    ...
    """
```

---

## الملف 8: [`backend/model_loader.py`](backend/model_loader.py)
### تعديل — إضافة حساب الميزات الطبية تلقائياً

**أضف قبل سطر `df = self._engineer_features(df)` في دالة `predict` و `explain`:**

```python
# حساب الميزات الطبية المتقدمة
# RPP = RestingBP * MaxHR
if 'RestingBP' in df.columns and 'MaxHR' in df.columns:
    df['RPP'] = df['RestingBP'] * df['MaxHR']

# Exercise_Risk_Index = Oldpeak * ExerciseAngina (encoded)
if 'Oldpeak' in df.columns and 'ExerciseAngina' in df.columns:
    exang_num = df['ExerciseAngina'].map({'Y': 1, 'N': 0}).fillna(0)
    df['Exercise_Risk_Index'] = df['Oldpeak'] * exang_num
```

**ملاحظة:** هذا سيحتاج إضافته في مكانين: داخل `predict()` وداخل `explain()`.

---

## ترتيب التنفيذ

| الخطوة | الأمر | ماذا يحدث |
|--------|-------|-----------|
| 1 | إنشاء النسخ الاحتياطية | حماية كل الملفات الأصلية |
| 2 | إنشاء `transform_heart11.py` | سكربت تحويل جديد |
| 3 | تشغيل `transform_heart11.py` | heart.csv ← بيانات محولة بدون ca/thal |
| 4 | تعديل `merge_data.py` | 3 تغييرات محدودة |
| 5 | تشغيل `merge_data.py` | دمج Cleveland + Z-Alizadeh → merged_heart_data.csv |
| 6 | تعديل `advanced_feature_engineering.py` | إضافة engineer_medical_features |
| 7 | تعديل `heart_disease_features.py` | إضافة engineer_medical method |
| 8 | تعديل `clean_data.py` | تحديث قوائم الميزات |
| 9 | تشغيل `clean_data.py` | → final_ready_data.csv مع imputation, encoding, scaling |
| 10 | تعديل `configs/heart_disease.yaml` | تحديث الاسم والوصف والميزات |
| 11 | تعديل `backend/schemas.py` | تحديث docstring |
| 12 | تعديل `backend/model_loader.py` | إضافة الحساب التلقائي في predict/explain |
| 13 | التحقق النهائي | التأكد من عمل كل شيء |

---

## 🛡️ الضمانات (Safeguards)

1. **Backup لكل ملف** قبل التعديل (`.bak`)
2. **اختبار كل سكربت** بعد إنشائه/تعديله قبل الانتقال للخطوة التالية
3. **assert على الـ merge** لاكتشاف NaN مبكراً
4. **الملفات غير المذكورة أعلاه لا تلمس** — backend/router.py, backend/main.py, frontend/, models/.gitkeep, models/heart_disease/grid_best.json وغيرها تبقى كما هي
5. **لا تغيير في API endpoints** — فقط إضافة feature engineering داخل الـ Backend
