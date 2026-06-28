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
import traceback

# Configure startup logging to stdout for HF Spaces debugging
logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("omnidiag.startup")
_explain_log = logging.getLogger("omnidiag.explain")
_cf_log = logging.getLogger("omnidiag.counterfactuals")
log.info("=" * 60)
log.info("OmniDiag starting up...")
log.info(f"Python version: {sys.version}")
log.info(f"Current working directory: {os.getcwd()}")
log.info(f"Files in cwd: {os.listdir('.')}")
log.info(f"models/heart_disease/xgboost_weights/ exists: {os.path.isdir('models/heart_disease/xgboost_weights/')}")
log.info(f"models/heart_disease/preprocessors/ exists: {os.path.isdir('models/heart_disease/preprocessors/')}")
log.info("=" * 60)

import uuid as _uuid
from contextlib import asynccontextmanager

import csv
import io
from typing import Any, Dict, List

from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, Response, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from sqlalchemy.ext.asyncio import AsyncSession

from backend.router import OmniDiagRouter
from backend.schemas import get_schema_for_disease, ExplainResponse
from backend.auth.routes import router as auth_router
from backend.admin.routes import router as admin_router
from backend.patients.routes import router as patients_router
from backend.auth.rbac import require_role, CLINICAL_ROLES
from backend.auth.dependencies import get_current_active_user
from backend.cache import init_cache, cache_get, cache_set, predict_cache_key, schema_cache_key
from backend.database import get_db
from backend.db_models.prediction import Prediction
from backend.db_models.user import User
from backend.middleware.audit import AuditMiddleware
from backend.middleware.security import SecurityHeadersMiddleware
from backend.rate_limit import limiter, LIMIT_CLINICAL, LIMIT_ADMIN
from backend.monitoring.routes import router as monitoring_router
from backend.llm.report_generator import generate_report
from backend.nlp.notes_parser import parse_clinical_note
from backend.active_learning.routes import router as review_router
from backend.active_learning.sampler import should_queue_for_review, prediction_entropy
from backend.monitoring.metrics import record_prediction, record_batch

# 1. تهيئة الموجه الديناميكي (يُحمّل جميع الإعدادات من configs/ تلقائياً)
log.info("Initializing OmniDiagRouter...")
try:
    router = OmniDiagRouter(configs_dir="configs")
    log.info(f"Router initialized. Available diseases: {router.get_available_diseases()}")
except Exception as e:
    log.error(f"Failed to initialize router: {e}", exc_info=True)
    raise


# 2. Lifespan — initialise cache on startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_cache()
    yield


# 3. تهيئة التطبيق وتوثيقه
app = FastAPI(
    title="OmniDiag Multi-Disease Diagnostic API",
    description="منصة تشخيص متعددة الأمراض: توجيه ديناميكي للمرضى إلى النموذج الصحيح مع تفسير SHAP.",
    version="4.0.0",
    lifespan=lifespan,
)

# Rate limiter state
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# ── Global exception handlers — standardized error envelope ──────────────────

def _json_safe(obj):
    """Recursively convert an object to a JSON-serializable form."""
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(i) for i in obj]
    return str(obj)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Return 422 validation errors in a consistent envelope."""
    return JSONResponse(
        status_code=422,
        content={
            "error": "Request validation failed",
            "code": "VALIDATION_ERROR",
            "request_id": str(_uuid.uuid4()),
            "detail": _json_safe(exc.errors()),
        },
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Wrap all HTTP exceptions in a consistent envelope."""
    detail = exc.detail
    # If detail is already a dict with an 'error' key, pass through
    if isinstance(detail, dict) and "error" in detail:
        body = {"request_id": str(_uuid.uuid4()), **detail}
    else:
        body = {
            "error": str(detail),
            "code": f"HTTP_{exc.status_code}",
            "request_id": str(_uuid.uuid4()),
        }
    return JSONResponse(status_code=exc.status_code, content=body, headers=getattr(exc, "headers", None) or {})

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

# Auth routes (/auth/register, /auth/login, /auth/refresh, /auth/logout, /auth/me)
app.include_router(auth_router, prefix="/auth", tags=["Auth"])

# Admin routes (/admin/audit-logs — super_admin only)
app.include_router(admin_router, prefix="/admin", tags=["Admin"])

# Patient CRUD routes (/api/v4/patients/*)
app.include_router(patients_router, prefix="/api/v4/patients", tags=["Patients"])

# Monitoring: drift detection endpoints under admin prefix
app.include_router(monitoring_router, prefix="/api/v4/admin", tags=["Monitoring"])
app.include_router(review_router, prefix="/api/v4/review", tags=["Active Learning"])

# Audit middleware — logs every /api/* and /auth/* request to audit_logs table.
# Must be added AFTER CORSMiddleware so preflight OPTIONS are excluded.
app.add_middleware(AuditMiddleware)

# Security headers middleware — OWASP-recommended headers (HIPAA alignment)
# enforce_https=False in development; set via env var in production
_enforce_https = os.getenv("ENFORCE_HTTPS", "true").lower() == "true"
app.add_middleware(SecurityHeadersMiddleware, enforce_https=_enforce_https)

# =========================================================================
# المسارات العامة (System)
# =========================================================================

@app.get("/metrics", include_in_schema=False)
async def prometheus_metrics():
    """Prometheus metrics endpoint for scraping."""
    from backend.monitoring.metrics import get_metrics_response
    from fastapi.responses import Response as _Response
    body, content_type = get_metrics_response()
    return _Response(content=body, media_type=content_type)


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
async def get_disease_schema(disease: str, response: Response):
    """
    Get the JSON Schema for a disease's patient input fields.
    Cached for 24 h — schema changes only on deployment.
    """
    key = schema_cache_key(disease)
    cached = await cache_get(key)
    if cached is not None:
        response.headers["Cache-Hit"] = "true"
        return cached

    schema = get_schema_for_disease(disease)
    result = schema.model_json_schema()
    await cache_set(key, result, ttl=86400)
    response.headers["Cache-Hit"] = "false"
    return result

# =========================================================================
# المسارات الديناميكية (التشخيص)
# =========================================================================

def _validate_patient_input(disease: str, patient: dict) -> dict:
    """Validate and coerce patient data against the disease schema if one exists."""
    schema = get_schema_for_disease(disease)
    if schema:
        return schema(**patient).model_dump()
    return patient


@app.post("/api/v4/{disease}/predict", tags=["Clinical Diagnosis"])
@limiter.limit(LIMIT_CLINICAL)
async def predict_disease(
    request: Request,
    disease: str,
    patient: dict,
    response: Response,
    patient_id: str = Query(None, description="Optional patient UUID to link this prediction"),
    current_user: User = Depends(require_role(*CLINICAL_ROLES)),
    db: AsyncSession = Depends(get_db),
):
    """
    تشخيص مريض لمرض معين.

    يتطلب دور: doctor | nurse | super_admin

    النتائج مؤقتة في الكاش لـ 5 دقائق للمدخلات المتكررة.
    يمكن ربط النتيجة بمريض موجود عبر ?patient_id=<uuid>
    """
    patient_data = _validate_patient_input(disease, patient)

    # Cache check (skip when linking to a specific patient for accurate audit)
    cache_key = predict_cache_key(disease, patient_data)
    if not patient_id:
        cached = await cache_get(cache_key)
        if cached is not None:
            response.headers["Cache-Hit"] = "true"
            return cached

    result = router.predict(disease, patient_data)

    # Store in cache
    if not patient_id:
        await cache_set(cache_key, result, ttl=300)
    response.headers["Cache-Hit"] = "false"

    # Record Prometheus metrics
    try:
        record_prediction(
            disease=disease,
            prediction=int(result.get("prediction", 0)),
            confidence=float(result.get("confidence", 0.0)),
        )
    except Exception:
        pass

    # Persist to predictions table (best-effort — never fail the response)
    try:
        confidence = float(result.get("confidence", 0.0))
        record = Prediction(
            id=str(_uuid.uuid4()),
            patient_id=patient_id,
            disease=disease,
            input_features=patient_data,
            prediction=int(result.get("prediction", 0)),
            confidence=confidence,
            diagnosis=result.get("diagnosis"),
            created_by=current_user.id,
        )
        db.add(record)
        await db.flush()

        # Auto-queue uncertain predictions for human review (Feature 1.3)
        if should_queue_for_review(confidence):
            from backend.db_models.review_queue import ReviewQueue
            rq = ReviewQueue(
                id=str(_uuid.uuid4()),
                prediction_id=record.id,
                uncertainty_score=prediction_entropy(confidence),
            )
            db.add(rq)

        await db.commit()
    except Exception as _exc:
        log.warning("predict: failed to persist prediction record — %s", _exc)

    return result


@app.post("/api/v4/{disease}/explain", response_model=ExplainResponse, tags=["Clinical Diagnosis"])
@limiter.limit(LIMIT_CLINICAL)
async def explain_disease(
    request: Request,
    disease: str,
    patient: dict,
    patient_id: str = Query(None, description="Optional patient UUID to link SHAP data"),
    current_user: User = Depends(require_role(*CLINICAL_ROLES)),
    db: AsyncSession = Depends(get_db),
):
    """
    تفسير قرار التشخيص باستخدام SHAP.

    يتطلب دور: doctor | nurse | super_admin
    """
    try:
        patient_data = _validate_patient_input(disease, patient)

        _explain_log.debug(f"Explain request for disease={disease}")
        result = router.explain(disease, patient_data)
        _explain_log.debug("Explain completed successfully")

        # Persist prediction + SHAP data (best-effort)
        try:
            record = Prediction(
                id=str(_uuid.uuid4()),
                patient_id=patient_id,
                disease=disease,
                input_features=patient_data,
                prediction=int(result.get("prediction", 0)),
                confidence=float(result.get("confidence", 0.0)),
                diagnosis=result.get("diagnosis"),
                shap_chart_data=result.get("chart_data"),
                created_by=current_user.id,
            )
            db.add(record)
            await db.commit()
        except Exception as _exc:
            log.warning("explain: failed to persist prediction record — %s", _exc)

        return result
    except HTTPException:
        raise
    except Exception as e:
        _explain_log.error(f"Explain failed for disease={disease}: {type(e).__name__}: {e}")
        _explain_log.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail={
                "error": f"Explain failed: {type(e).__name__}: {str(e)}",
                "traceback": traceback.format_exc().split("\n")[-5:] if _explain_log.isEnabledFor(logging.DEBUG) else [],
            },
        )


@app.post("/api/v4/{disease}/counterfactuals", tags=["Clinical Diagnosis"])
@limiter.limit(LIMIT_CLINICAL)
async def counterfactuals_disease(
    request: Request,
    disease: str,
    patient: dict,
    _user: User = Depends(require_role(*CLINICAL_ROLES)),
):
    """
    توليد سيناريوهات "ماذا لو" لتقليل المخاطر.

    يتطلب دور: doctor | nurse | super_admin
    """
    try:
        patient_data = _validate_patient_input(disease, patient)

        _cf_log.debug(f"Counterfactuals request for disease={disease}")
        result = router.counterfactuals(disease, patient_data)
        _cf_log.debug(f"Counterfactuals completed: status={result.get('status')}")
        return result
    except HTTPException:
        raise
    except Exception as e:
        _cf_log.error(f"Counterfactuals failed for disease={disease}: {type(e).__name__}: {e}")
        _cf_log.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail={
                "error": f"Counterfactuals generation failed: {type(e).__name__}: {str(e)}",
                "traceback": traceback.format_exc().split("\n")[-5:] if _cf_log.isEnabledFor(logging.DEBUG) else [],
            },
        )

# =========================================================================
# المسارات القديمة (متوافقة مع الإصدارات السابقة)
# =========================================================================

@app.post("/api/v3/predict", tags=["Legacy"])
def predict_legacy(patient: dict):
    """نقطة نهاية قديمة - تعيد التوجيه إلى heart_disease."""
    return router.predict("heart_disease", patient)


# ── Batch prediction ──────────────────────────────────────────────────────────

class BatchRowResult(BaseModel):
    row: int
    status: str          # "ok" | "error"
    prediction: int | None = None
    confidence: float | None = None
    diagnosis: str | None = None
    error: str | None = None


class BatchResponse(BaseModel):
    disease: str
    total: int
    succeeded: int
    failed: int
    results: List[BatchRowResult]


@app.post(
    "/api/v4/{disease}/batch",
    response_model=BatchResponse,
    tags=["Clinical Diagnosis"],
    summary="Batch predict from CSV upload",
    description=(
        "Upload a CSV file where each row is a patient record. "
        "Returns predictions for all rows. "
        "Rows that fail validation are returned with status='error' and an error message. "
        "Maximum 500 rows per request."
    ),
)
@limiter.limit(LIMIT_CLINICAL)
async def batch_predict(
    request: Request,
    disease: str,
    file: UploadFile = File(..., description="CSV file with header row matching the disease schema"),
    _user: User = Depends(require_role(*CLINICAL_ROLES)),
) -> BatchResponse:
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail={"error": "File must be a .csv", "code": "INVALID_FILE_TYPE"})

    content = await file.read()
    try:
        text = content.decode("utf-8-sig")   # strip UTF-8 BOM if present
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail={"error": "CSV must be UTF-8 encoded", "code": "ENCODING_ERROR"})

    reader = csv.DictReader(io.StringIO(text))
    rows: List[Dict[str, Any]] = list(reader)

    if len(rows) == 0:
        raise HTTPException(status_code=400, detail={"error": "CSV is empty or has no data rows", "code": "EMPTY_CSV"})
    if len(rows) > 500:
        raise HTTPException(status_code=400, detail={"error": "Maximum 500 rows per batch request", "code": "TOO_MANY_ROWS"})

    results: List[BatchRowResult] = []
    succeeded = 0
    failed = 0

    for i, raw_row in enumerate(rows, start=1):
        # Convert numeric strings to appropriate types
        coerced: Dict[str, Any] = {}
        for k, v in raw_row.items():
            if v is None or v == "":
                coerced[k] = v
                continue
            try:
                # Try int first, then float
                if "." in str(v):
                    coerced[k] = float(v)
                else:
                    coerced[k] = int(v)
            except (ValueError, TypeError):
                coerced[k] = v

        try:
            validated = _validate_patient_input(disease, coerced)
            pred = router.predict(disease, validated)
            results.append(BatchRowResult(
                row=i,
                status="ok",
                prediction=pred.get("prediction"),
                confidence=pred.get("confidence"),
                diagnosis=pred.get("diagnosis"),
            ))
            succeeded += 1
        except Exception as exc:
            results.append(BatchRowResult(row=i, status="error", error=str(exc)))
            failed += 1

    # Record Prometheus batch metrics
    try:
        record_batch(disease=disease, succeeded=succeeded, failed=failed)
    except Exception:
        pass

    return BatchResponse(
        disease=disease,
        total=len(rows),
        succeeded=succeeded,
        failed=failed,
        results=results,
    )


# ---------------------------------------------------------------------------
# Feature 1.5 — LLM Clinical Report Generation
# ---------------------------------------------------------------------------

class ReportRequest(BaseModel):
    disease: str
    probability: float
    label: str
    confidence_band: str
    shap_values: List[Dict[str, Any]]
    features: Dict[str, Any]


@app.post(
    "/api/v4/generate-report",
    tags=["Clinical Diagnosis"],
    summary="Generate AI clinical report from prediction results",
)
@limiter.limit("10/minute")
async def generate_clinical_report(
    request: Request,
    body: ReportRequest,
    _user: User = Depends(require_role(*CLINICAL_ROLES)),
) -> Dict[str, Any]:
    disease_info = router.get_disease_info(body.disease)
    disease_display = (disease_info or {}).get("display_name", body.disease)

    result = await generate_report(
        disease_display=disease_display,
        probability=body.probability,
        label=body.label,
        confidence_band=body.confidence_band,
        shap_values=body.shap_values,
        features=body.features,
    )
    return {"disease": body.disease, **result}


# ---------------------------------------------------------------------------
# Feature 1.1 — NLP Clinical Notes Parser
# ---------------------------------------------------------------------------

class NotesParseRequest(BaseModel):
    note: str
    use_bert: bool = True


@app.post(
    "/api/v4/parse-notes",
    tags=["Clinical Diagnosis"],
    summary="Extract structured features from free-text clinical notes",
)
@limiter.limit("20/minute")
async def parse_notes(
    request: Request,
    body: NotesParseRequest,
    _user: User = Depends(require_role(*CLINICAL_ROLES)),
) -> Dict[str, Any]:
    extracted = parse_clinical_note(body.note, use_bert=body.use_bert)
    return {
        "extracted_features": extracted,
        "field_count": len(extracted),
        "note": "Fields not present in the note are omitted. Supply missing values manually.",
    }
