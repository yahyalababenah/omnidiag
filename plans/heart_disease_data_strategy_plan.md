# الخطة النهائية: التنبؤ بمرض الشريان التاجي (CAD)
## الإستراتيجية الذهبية — بناءً على توجيهات الخبير الطبي

---

## 1. الهدف النهائي (Project Goal)
**Predicting Coronary Artery Disease / Heart Risk**  
منصة تشخيص مبكر غير جراحي (Non-invasive Early Triage) لمرض الشريان التاجي.

---

## 2. الاستراتيجية المعتمدة ✅

```
┌────────────────────────────────────────────────────────────────────┐
│                     THE GOLDEN STRATEGY                           │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  heart11.csv (UCI Cleveland)    Z-Alizadeh sani (Tehran)          │
│  ┌──────────────────────┐       ┌──────────────────────────┐      │
│  │ 14 features          │       │ 54+ features             │      │
│  │ ~1026 rows           │       │ ~303 rows                │      │
│  └────────┬─────────────┘       └──────────┬───────────────┘      │
│           │                                │                       │
│           ▼                                ▼                       │
│  ┌─────────────────────────────────────────────┐                  │
│  │           STEP 1: Data Transformation        │                 │
│  │                                              │                 │
│  │  قلب11:                                     │                 │
│  │  • Drop ca ❌ (Data Leakage — Angiography)   │                 │
│  │  • Drop thal ❌ (Thallium Stress Test —      │                 │
│  │    expensive nuclear test, not appropriate   │                 │
│  │    for non-invasive triage)                  │                 │
│  │  • Map columns to 12 standard names          │                 │
│  │  • Remove duplicate rows                     │                 │
│  │                                              │                 │
│  │  Z-Alizadeh:                                 │                 │
│  │  • Keep 12 core features + map to standard   │                 │
│  │  • Calculate Chol via Friedewald:            │                 │
│  │    LDL + HDL + (TG / 5)                      │                 │
│  │  • Drop ca, thal (not in this dataset)       │                 │
│  └─────────────────────┬───────────────────────┘                  │
│                        │                                          │
│                        ▼                                          │
│  ┌─────────────────────────────────────────────┐                  │
│  │         STEP 2: Merge (Concatenate)         │                 │
│  │                                              │                 │
│  │  12 common features ✓   No NaN columns ✓     │                 │
│  │  Total: ~1329 unique patients                │                 │
│  │  After deduplication: ~605 unique patients   │                 │
│  └─────────────────────┬───────────────────────┘                  │
│                        │                                          │
│                        ▼                                          │
│  ┌─────────────────────────────────────────────┐                  │
│  │  STEP 3: Feature Engineering (Medical Math)  │                 │
│  │                                              │                 │
│  │  1. RPP (Rate-Pressure Product)              │                 │
│  │     trestbps × thalach                       │                 │
│  │     📊 يقيس استهلاك عضلة القلب للأكسجين      │                 │
│  │                                              │                 │
│  │  2. Age-BP Interaction                       │                 │
│  │     age × trestbps                            │                 │
│  │     📊 العبء التراكمي على الشرايين مع العمر  │                 │
│  │                                              │                 │
│  │  3. Cholesterol/Age Ratio                    │                 │
│  │     chol / age                                │                 │
│  │     📊 تراكم الدهون مقارنة بالسن الطبيعي     │                 │
│  │                                              │                 │
│  │  4. Exercise Risk Index                      │                 │
│  │     oldpeak × exang (encoded 0/1)            │                 │
│  │     📊 إذا ألم + انخفاض ST → تأكيد CAD       │                 │
│  │                                              │                 │
│  │  ✅ Total: 12 core + 4 engineered = 16 cols  │                 │
│  │  ✅ No NaN values — all derived from core     │                 │
│  └─────────────────────┬───────────────────────┘                  │
│                        │                                          │
│                        ▼                                          │
│  ┌─────────────────────────────────────────────┐                  │
│  │     STEP 4: Preprocessing (Clean)           │                 │
│  │                                              │                 │
│  │  • MissForest Imputation (Cholesterol, BP)   │                 │
│  │  • Label Encoding (Sex, ChestPainType, etc.) │                 │
│  │  • StandardScaler (ALL 16 features!)         │                 │
│  │  • Save encoders + scaler to preprocessors/  │                 │
│  └─────────────────────────────────────────────┘                  │
│                        │                                          │
│                        ▼                                          │
│  ┌─────────────────────────────────────────────┐                  │
│  │  STEP 5: Train XGBoost Model                │                 │
│  │                                              │                 │
│  │  • 16 features → XGBoost optimized           │                 │
│  │  • Optuna hyperparameter tuning              │                 │
│  │  • Save model weights                        │                 │
│  └─────────────────────────────────────────────┘                  │
└────────────────────────────────────────────────────────────────────┘
```

---

## 3. معالجة مخاطر الإنتاج (Production Risks) ✅

| المخاطرة | الحل البرمجي |
|----------|-------------|
| **User Burden** — الطبيب يدخل 12 ميزة فقط | الـ Backend يحسب الـ 4 ميزات الرياضية تلقائياً في الذاكرة قبل تمريرها للمودل |
| **Scaling Mismatch** — الميزات الجديدة أرقامها كبيرة | `scaler.pkl` يُدرّب على كل الـ 16 ميزة ويُستخدم في الـ Backend |
| **Missing Values** — الطبيب يترك حقل فارغ | التحقق من صحة المدخلات (Validation) + Mean Imputation قبل حساب المعادلات |

---

## 4. خريطة تحويل البيانات

### 4.1 heart11.csv → 12 عمود قياسي

| Old Column | New Column | Mapping Logic |
|------------|------------|---------------|
| `age` | `Age` | Direct |
| `sex` | `Sex` | 0→'F', 1→'M' |
| `cp` | `ChestPainType` | 0→'TA', 1→'ATA', 2→'NAP', 3→'ASY' |
| `trestbps` | `RestingBP` | Direct |
| `chol` | `Cholesterol` | Direct |
| `fbs` | `FastingBS` | Direct |
| `restecg` | `RestingECG` | 0→'Normal', 1→'ST', 2→'LVH' |
| `thalach` | `MaxHR` | Direct |
| `exang` | `ExerciseAngina` | 0→'N', 1→'Y' |
| `oldpeak` | `Oldpeak` | Direct |
| `slope` | `ST_Slope` | 0→'Up', 1→'Flat', 2→'Down' |
| `target` | `HeartDisease` | Direct |
| `ca` | ❌ **Dropped** | Data Leakage |
| `thal` | ❌ **Dropped** | Thallium Test (expensive, non-invasive focus) |

### 4.2 Z-Alizadeh → 12 عمود قياسي

| Old Column | New Column | Mapping Logic |
|------------|------------|---------------|
| `Age` | `Age` | Direct |
| `Sex` | `Sex` | Male→'M', Female→'F' |
| `Typical Chest Pain`, `Atypical`, `Nonanginal` | `ChestPainType` | Composition logic |
| `BP` | `RestingBP` | Direct |
| `LDL`, `HDL`, `TG` | `Cholesterol` | **Friedewald:** LDL + HDL + (TG/5) |
| `FBS` | `FastingBS` | Direct |
| `Q Wave`, `St Elevation`, `St Depression`, `Tinversion`, `LVH` | `RestingECG` | Composition logic |
| `PR` | `MaxHR` | Direct (Pulse Rate) |
| `Exertional CP` | `ExerciseAngina` | Direct |
| `St dep` | `Oldpeak` | Direct |
| ECG features | `ST_Slope` | Composition logic |
| `Cath` | `HeartDisease` | Normal→0, Cad→1 |

---

## 5. الميزات الجديدة (Feature Engineering)

| الميزة | المعادلة | المبرر الطبي |
|--------|----------|-------------|
| `RPP` | `RestingBP * MaxHR` | مؤشر استهلاك الأكسجين — معيار عالمي في طب القلب |
| `Age_BP_Interaction` | `Age * RestingBP` | العبء التراكمي للضغط مع تقدم العمر |
| `Chol_Age_Ratio` | `Cholesterol / Age` | معدل تراكم الدهون مقارنة بالعمر |
| `Exercise_Risk_Index` | `Oldpeak * ExerciseAngina` | تأكيد CAD عند وجود ألم + انخفاض ST |

---

## 6. خطوات التنفيذ (Implementation Plan)

### المرحلة 1: تحويل البيانات
- [ ] إنشاء [`experiment_files/data_pipeline/transform_heart11.py`](experiment_files/data_pipeline/)
  - تحويل heart11.csv → 12 عموداً قياسياً
  - إسقاط `ca` و `thal`
  - حذف التكرارات
  - حفظ كـ `heart_cleveland.csv`
- [ ] تحديث [`experiment_files/data_pipeline/merge_data.py`](experiment_files/data_pipeline/merge_data.py)
  - قراءة Z-Alizadeh واستخراج الـ 12 عموداً
  - حساب Chol عبر Friedewald: `LDL + HDL + (TG / 5)`
  - دمج مع heart_cleveland.csv (row-wise)
  - حفظ كـ `merged_heart_data.csv`

### المرحلة 2: Feature Engineering + تنظيف
- [ ] تحديث [`models/advanced_feature_engineering.py`](models/advanced_feature_engineering.py)
  - إضافة `RPP = RestingBP * MaxHR`
  - إضافة `Exercise_Risk_Index = Oldpeak * ExerciseAngina` (encoded)
  - الحفاظ على `Age_BP_Interaction` و `Chol_Age_Ratio` الموجودين
- [ ] تحديث [`features/heart_disease_features.py`](features/heart_disease_features.py)
  - إضافة الدوال الجديدة
- [ ] تحديث [`configs/heart_disease.yaml`](configs/heart_disease.yaml)
  - إضافة الميزات الجديدة إلى `heuristic_features`
  - تحديث `display_name` و `description`
- [ ] تحديث [`experiment_files/data_pipeline/clean_data.py`](experiment_files/data_pipeline/clean_data.py)
  - توسيع imputation ليشمل كل الميزات
  - توسيع scaling ليشمل الميزات الـ 4 الجديدة

### المرحلة 3: تحديث الـ Backend (للإنتاج)
- [ ] تحديث [`backend/schemas.py`](backend/schemas.py)
  - تحديث docstring
- [ ] التأكد من [`backend/model_loader.py`](backend/model_loader.py)
  - حساب الميزات الرياضية تلقائياً قبل `_apply_preprocessors()`
  - معالجة القيم المفقودة

### المرحلة 4: التدريب
- [ ] إعادة تدريب نموذج XGBoost بالميزات الـ 16
- [ ] تحسين Hyperparameters عبر Optuna
- [ ] حفظ المودل الجديد و preprocessors

### المرحلة 5: التوثيق
- [ ] تحديث [`README.md`](README.md)
- [ ] تحديث [`configs/heart_disease.yaml`](configs/heart_disease.yaml)

---

## 7. ملخص البيانات النهائي

```
📊 الـ Dataset النهائي:
   • ~605 مريض فريد (بعد دمج Cleveland + Z-Alizadeh وإزالة التكرارات)
   • 12 ميزة أساسية (Core Features)
   • 4 ميزات رياضية (Engineered Features)
   • 0% NaN — كل الميزات محسوبة من الـ 12 ميزة الأساسية
   • هدف واحد: HeartDisease (0 = سليم, 1 = CAD)

🎯 المودل في الإنتاج (Production):
   • الطبيب يدخل: 12 ميزة أساسية
   • الـ Backend يحسب: 4 ميزات إضافية تلقائياً
   • المجموع: 16 ميزة → XGBoost → SHAP Explanation
```
