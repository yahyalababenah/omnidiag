"""
OmniDiag Multi-Disease API
===========================
Dynamic FastAPI backend that routes patient data to the correct disease model
using the OmniDiagRouter. New diseases are added by creating a YAML config
and a feature engineer class — no API code changes needed.
"""

import sys
import os
import logging

# Configure startup logging to stdout for HF Spaces debugging
logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("omnidiag.startup")
log.info("=" * 60)
log.info("OmniDiag starting up...")
log.info(f"Python version: {sys.version}")
log.info(f"Current working directory: {os.getcwd()}")
log.info(f"Files in cwd: {os.listdir('.')}")
log.info(f"models/heart_disease/xgboost_weights/ exists: {os.path.isdir('models/heart_disease/xgboost_weights/')}")
log.info(f"models/heart_disease/preprocessors/ exists: {os.path.isdir('models/heart_disease/preprocessors/')}")
log.info("=" * 60)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from backend.router import OmniDiagRouter
from backend.schemas import get_schema_for_disease, ExplainResponse

# 1. تهيئة الموجه الديناميكي (يُحمّل جميع الإعدادات من configs/ تلقائياً)
log.info("Initializing OmniDiagRouter...")
try:
    router = OmniDiagRouter(configs_dir="configs")
    log.info(f"Router initialized. Available diseases: {router.get_available_diseases()}")
except Exception as e:
    log.error(f"Failed to initialize router: {e}", exc_info=True)
    raise

# 2. تهيئة التطبيق وتوثيقه
app = FastAPI(
    title="OmniDiag Multi-Disease Diagnostic API",
    description="منصة تشخيص متعددة الأمراض: توجيه ديناميكي للمرضى إلى النموذج الصحيح مع تفسير SHAP.",
    version="4.0.0"
)

# 3. CORS — Strict allowlist (production Vercel URL + local dev)
#    NEVER use ["*"] in production; this prevents unauthorized origins.
#
#    💡 When you deploy the frontend to a NEW Vercel URL (e.g. a preview
#       deployment), add it here or set the CORS_ALLOWED_ORIGINS env var
#       on Hugging Face Spaces to override the defaults.
_DEFAULT_ORIGINS = (
    "https://omnidiag-delta.vercel.app,"
    "https://omnidiag-qnhrjmoaq-yahia-s-projects05.vercel.app,"
    "http://localhost:5173,"
    "http://localhost:3000"
)

_ALLOWED_ORIGINS = os.getenv("CORS_ALLOWED_ORIGINS", _DEFAULT_ORIGINS).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in _ALLOWED_ORIGINS if origin.strip()],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "Accept"],
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


@app.get("/api/v4/{disease}/schema", tags=["Clinical Diagnosis"])
def get_disease_schema(disease: str):
    """
    Get the JSON Schema for a disease's patient input fields.
    
    Returns the complete Pydantic model JSON Schema including all field names,
    types, descriptions, validation constraints (ge/le), and example values.
    Enables dynamic form generation in the React frontend without hardcoding
    field definitions per disease.
    """
    schema = get_schema_for_disease(disease)
    return schema.model_json_schema()

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

@app.post("/api/v4/{disease}/explain", response_model=ExplainResponse, tags=["Clinical Diagnosis"])
def explain_disease(disease: str, patient: dict):
    """
    تفسير قرار التشخيص باستخدام SHAP.
    
    Args:
        disease: اسم المرض.
        patient: بيانات المريض.
    
    Returns:
        قيم SHAP والتفسير الطبي.
    """
    import traceback
    log = logging.getLogger("omnidiag.explain")
    try:
        schema = get_schema_for_disease(disease)
        if schema:
            validated = schema(**patient)
            patient_data = validated.model_dump()
        else:
            patient_data = patient
        
        log.debug(f"Explain request for disease={disease}, patient={patient_data}")
        result = router.explain(disease, patient_data)
        log.debug("Explain completed successfully")
        return result
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Explain failed for disease={disease}: {type(e).__name__}: {e}")
        log.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail={
                "error": f"Explain failed: {type(e).__name__}: {str(e)}",
                "traceback": traceback.format_exc().split("\n")[-5:] if log.isEnabledFor(logging.DEBUG) else []
            }
        )

@app.post("/api/v4/{disease}/counterfactuals", tags=["Clinical Diagnosis"])
def counterfactuals_disease(disease: str, patient: dict):
    """
    توليد سيناريوهات "ماذا لو" لتقليل المخاطر.
    
    يُنشئ 3 سيناريوهات متنوعة (DiCE) تُظهر التغييرات الممكنة
    التي يمكن للمريض إجراؤها لتقليل خطر الإصابة بالمرض.
    
    Args:
        disease: اسم المرض.
        patient: بيانات المريض.
    
    Returns:
        قائمة بسيناريوهات "ماذا لو" مع التغييرات المقترحة ونسبة تقليل المخاطر.
    """
    import traceback
    log = logging.getLogger("omnidiag.counterfactuals")
    try:
        schema = get_schema_for_disease(disease)
        if schema:
            validated = schema(**patient)
            patient_data = validated.model_dump()
        else:
            patient_data = patient
        
        log.debug(f"Counterfactuals request for disease={disease}, patient={patient_data}")
        result = router.counterfactuals(disease, patient_data)
        log.debug(f"Counterfactuals completed: status={result.get('status')}")
        return result
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Counterfactuals failed for disease={disease}: {type(e).__name__}: {e}")
        log.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail={
                "error": f"Counterfactuals generation failed: {type(e).__name__}: {str(e)}",
                "traceback": traceback.format_exc().split("\n")[-5:] if log.isEnabledFor(logging.DEBUG) else []
            }
        )

# =========================================================================
# المسارات القديمة (متوافقة مع الإصدارات السابقة)
# =========================================================================

@app.post("/api/v3/predict", tags=["Legacy"])
def predict_legacy(patient: dict):
    """نقطة نهاية قديمة - تعيد التوجيه إلى heart_disease."""
    return router.predict("heart_disease", patient)
