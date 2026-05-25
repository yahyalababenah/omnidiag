"""
OmniDiag Multi-Disease API
===========================
Dynamic FastAPI backend that routes patient data to the correct disease model
using the OmniDiagRouter. New diseases are added by creating a YAML config
and a feature engineer class — no API code changes needed.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from backend.router import OmniDiagRouter
from backend.schemas import get_schema_for_disease

# 1. تهيئة الموجه الديناميكي (يُحمّل جميع الإعدادات من configs/ تلقائياً)
router = OmniDiagRouter(configs_dir="configs")

# 2. تهيئة التطبيق وتوثيقه
app = FastAPI(
    title="OmniDiag Multi-Disease Diagnostic API",
    description="منصة تشخيص متعددة الأمراض: توجيه ديناميكي للمرضى إلى النموذج الصحيح مع تفسير SHAP.",
    version="4.0.0"
)

# 3. حماية الـ CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================================
# المسارات العامة (System)
# =========================================================================

@app.get("/", tags=["System"])
def health_check():
    """فحص صحة النظام وعرض الأمراض المتاحة."""
    return {
        "status": "Healthy",
        "version": "4.0.0",
        "available_diseases": router.get_available_diseases()
    }

@app.get("/api/v4/diseases", tags=["System"])
def list_diseases():
    """عرض قائمة الأمراض المتاحة في المنصة."""
    return {
        "diseases": [
            {
                "name": name,
                "info": router.get_disease_info(name)
            }
            for name in router.get_available_diseases()
        ]
    }

# =========================================================================
# المسارات الديناميكية (التشخيص)
# =========================================================================

@app.post("/api/v4/{disease}/predict", tags=["Clinical Diagnosis"])
def predict_disease(disease: str, patient: dict):
    """
    تشخيص مريض لمرض معين.
    
    Args:
        disease: اسم المرض (مطابق لاسم ملف الإعدادات في configs/).
        patient: بيانات المريض (تختلف حسب المرض).
    
    Returns:
        التشخيص مع نسبة الثقة والتفسير.
    """
    # التحقق من صحة البيانات حسب schema المرض
    schema = get_schema_for_disease(disease)
    if schema:
        validated = schema(**patient)
        patient_data = validated.model_dump()
    else:
        patient_data = patient
    
    return router.predict(disease, patient_data)

@app.post("/api/v4/{disease}/explain", tags=["Clinical Diagnosis"])
def explain_disease(disease: str, patient: dict):
    """
    تفسير قرار التشخيص باستخدام SHAP.
    
    Args:
        disease: اسم المرض.
        patient: بيانات المريض.
    
    Returns:
        قيم SHAP والتفسير الطبي.
    """
    schema = get_schema_for_disease(disease)
    if schema:
        validated = schema(**patient)
        patient_data = validated.model_dump()
    else:
        patient_data = patient
    
    return router.explain(disease, patient_data)

# =========================================================================
# المسارات القديمة (متوافقة مع الإصدارات السابقة)
# =========================================================================

@app.post("/api/v3/predict", tags=["Legacy"])
def predict_legacy(patient: dict):
    """نقطة نهاية قديمة - تعيد التوجيه إلى heart_disease."""
    return router.predict("heart_disease", patient)