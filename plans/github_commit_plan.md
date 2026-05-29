# خطة رفع الملفات إلى GitHub

## الهدف
رفع الملفات الأساسية للإنتاج فقط إلى GitHub (`origin` remote)، وتجاهل ملفات التجارب والنسخ الاحتياطية والملفات غير الضرورية.

## الوضع الحالي
- نحن على فرع `production-final-v3`
- remote `origin` ← `https://github.com/yahyalababenah/omnidiag.git`
- remote `hf` ← `https://huggingface.co/spaces/yahyoha/omnidiag.git` (تم الرفع بالفعل)

## ⚠️ مشكلة مهمة: `.gitignore` يمنع `*.pkl`

السطر 6 في `.gitignore` يمنع رفع أي ملف `.pkl` بما في ذلك:
- `models/heart_disease/omni_diag_xgb_optimized.pkl` (ملف النموذج)
- `models/heart_disease/preprocessors/label_encoders.pkl`
- `models/heart_disease/preprocessors/standard_scaler.pkl`

**الحل**: استخدام `git add -f` (force) لإجبار رفع هذه الملفات.

## الملفات التي يجب رفعها إلى GitHub

### 📁 Backend (أساسي)
| الملف | الحالة |
|-------|--------|
| `backend/__init__.py` | موجود مسبقاً |
| `backend/main.py` | موجود مسبقاً (تم تعديل CORS) |
| `backend/model_loader.py` | موجود مسبقاً |
| `backend/router.py` | موجود مسبقاً |
| `backend/schemas.py` | موجود مسبقاً |
| `backend/shap_service.py` | موجود مسبقاً |

### 📁 Configs
| الملف | الحالة |
|-------|--------|
| `configs/config_loader.py` | موجود مسبقاً |
| `configs/heart_disease.yaml` | موجود مسبقاً |

### 📁 Features
| الملف | الحالة |
|-------|--------|
| `features/__init__.py` | موجود مسبقاً |
| `features/base_features.py` | موجود مسبقاً |
| `features/heart_disease_features.py` | **تم التعديل** (إضافة `engineer_medical`) ✅ |

### 📁 Models
| الملف | الحالة |
|-------|--------|
| `models/advanced_feature_engineering.py` | **تم التعديل** (إصلاح StringDtype) ✅ |
| `models/heart_disease/omni_diag_xgb_optimized.pkl` | **نموذج معاد تدريبه** (16 ميزة) ✅ - يحتاج `git add -f` |
| `models/heart_disease/grid_best.json` | موجود مسبقاً |
| `models/heart_disease/metadata.json` | موجود مسبقاً |
| `models/heart_disease/metrics.json` | موجود مسبقاً |
| `models/heart_disease/preprocessors/label_encoders.pkl` | **محدث** ✅ - يحتاج `git add -f` |
| `models/heart_disease/preprocessors/standard_scaler.pkl` | **محدث** ✅ - يحتاج `git add -f` |

### 📁 Frontend (كامل)
| الملف | الحالة |
|-------|--------|
| `frontend/` بكامله | **ملفات جديدة/معدلة** |
| `frontend/vercel.json` | **ملف جديد** (إصلاح SPA routing) ✅ |

### 📁 Infrastructure
| الملف | الحالة |
|-------|--------|
| `Dockerfile` | **تم التعديل** ✅ |
| `.dockerignore` | موجود مسبقاً |
| `requirements.txt` | موجود مسبقاً |
| `README.md` | موجود مسبقاً |
| `LICENSE` | موجود مسبقاً |
| `.gitignore` | موجود مسبقاً |

### 📁 Data (ملفات البيانات الأساسية)
| الملف | الحالة |
|-------|--------|
| `data/heart_disease/raw/heart11.csv` | **منقول من جذر المشروع** 🔄 — المصدر الأساسي للبيانات |
| `data/heart_disease/raw/heart.csv` | **محدث** ✅ |
| `data/heart_disease/processed/final_ready_data.csv` | **محدث** ✅ |
| `data/heart_disease/processed/merged_heart_data.csv` | **محدث** ✅ |

### 📁 GitHub CI/CD
| الملف | الحالة |
|-------|--------|
| `.github/workflows/ci.yml` | موجود مسبقاً |

## الملفات التي يجب تجاهلها (لا ترفع إلى GitHub)

| المسار | السبب |
|--------|-------|
| `experiment_files/` | ملفات تجارب وتطوير، ليست للإنتاج |
| `*.bak` | نسخ احتياطية |
| `plans/` | خطط داخلية، اختياري |
| `.vscode/` | إعدادات محرر شخصية |
| `.gemini_context` | ملف سياق Gemini |
| `diagnose_features.py` | سكربت تشخيص مؤقت |

## خطوة إضافية: نقل heart11.csv إلى data/heart_disease/raw/

قبل الـ commit، يجب نقل `heart11.csv` من جذر المشروع إلى `data/heart_disease/raw/`:

```bash
mv heart11.csv data/heart_disease/raw/heart11.csv
```

ثم تحديث المسار في `experiment_files/data_pipeline/transform_heart11.py` (line 39):
```python
# قبل:
heart11_path = os.path.join(PROJECT_ROOT, "heart11.csv")
# بعد:
heart11_path = os.path.join(raw_dir, "heart11.csv")
```

## خطة التنفيذ (Code Mode)

### الخطوة 1: نقل heart11.csv إلى مجلد raw
```bash
mv heart11.csv data/heart_disease/raw/heart11.csv
```

### الخطوة 2: تحديث المسار في transform_heart11.py

### الخطوة 3: تحديث `.gitignore`
إزالة `*.pkl` من `.gitignore` (أو إضافة استثناءات للملفات المطلوبة):
```
# استثناء ملفات النموذج والمعالجات (مطلوبة للتشغيل)
!models/heart_disease/omni_diag_xgb_optimized.pkl
!models/heart_disease/preprocessors/*.pkl
```

### الخطوة 2: إضافة الملفات و commit
```bash
# Force-add model files (مطلوب لأن .gitignore يمنع *.pkl)
git add -f models/heart_disease/omni_diag_xgb_optimized.pkl
git add -f models/heart_disease/preprocessors/label_encoders.pkl
git add -f models/heart_disease/preprocessors/standard_scaler.pkl

# إضافة باقي الملفات
git add backend/ configs/ features/ models/advanced_feature_engineering.py
git add frontend/ Dockerfile .dockerignore requirements.txt README.md LICENSE .gitignore
git add data/heart_disease/raw/heart11.csv
git add data/heart_disease/raw/heart.csv
git add .github/workflows/ci.yml

git status  # تأكد من أن الملفات المطلوبة فقط موجودة

git commit -m "OmniDiag v4.0 — إصلاح StringDtype bug + إعادة تدريب النموذج + إصلاح البناء

- إصلاح StringDtype bug في engineer_medical_features (Pandas 3.x)
- إعادة تدريب النموذج لـ 16 ميزة ثابتة
- إضافة engineer_medical إلى HeartDiseaseFeatureEngineer
- إنشاء vercel.json لتوجيه SPA
- تحديث CORS whitelist
- إصلاح بناء Docker"
```

### الخطوة 3: الدفع إلى GitHub
نحتاج لاختيار أي فرع ندفع إليه. الخيارات:
- **الخيار أ**: دفع `production-final-v3` إلى `origin/main` (أو `origin/production-final-v3`)
- **الخيار ب**: دمج `production-final-v3` في `main1` (الفرع الحالي على origin)
- **الخيار ج**: إنشاء فرع جديد `production` على origin

### الخطوة 4: إعادة نشر الفرونت إند على Vercel
```bash
cd frontend
npx vercel --prod
```

## ملاحظة مهمة
الـ `origin` remote يشير إلى `github.com/yahyalababenah/omnidiag.git` بينما `hf` remote يشير إلى `huggingface.co/spaces/yahyoha/omnidiag.git`. كل remote له محتوى مختلف:
- **HF**: يحتوي على جميع الملفات بما فيها `experiment_files/` (لأنه تم دفعه كاملاً)
- **GitHub**: سنرفع فقط ملفات الإنتاج
