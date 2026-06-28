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

from dotenv import load_dotenv
load_dotenv()  # loads .env file into os.environ before anything else reads it

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
from typing import Any, Dict, List, Optional

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
from backend.auth.dependencies import get_current_active_user, get_optional_user
from backend.cache import init_cache, cache_get, cache_set, predict_cache_key, schema_cache_key
from backend.database import get_db, engine, AsyncSessionLocal, Base
from backend.db_models.prediction import Prediction
from backend.db_models.user import User
from backend.db_models.role import Role
from backend.middleware.audit import AuditMiddleware
from backend.middleware.security import SecurityHeadersMiddleware
from backend.rate_limit import limiter, LIMIT_CLINICAL, LIMIT_ADMIN
from backend.monitoring.routes import router as monitoring_router
from backend.monitoring.metrics import record_prediction, record_batch

# Feature modules — imported lazily inside endpoints; safe stubs defined here
# so the rest of the file doesn't need try/except everywhere.
try:
    from backend.llm.report_generator import generate_report as _generate_report
    _HAS_LLM = True
except Exception as _e:
    log.warning("LLM module unavailable (%s) — /generate-report will return rule-based output only", _e)
    _generate_report = None
    _HAS_LLM = False

try:
    from backend.nlp.notes_parser import parse_clinical_note as _parse_clinical_note
    _HAS_NLP = True
except Exception as _e:
    log.warning("NLP module unavailable (%s) — /parse-notes will return empty results", _e)
    _parse_clinical_note = None
    _HAS_NLP = False

try:
    from backend.active_learning.routes import router as review_router
    from backend.active_learning.sampler import should_queue_for_review, prediction_entropy
    _HAS_AL = True
except Exception as _e:
    log.warning("Active learning module unavailable (%s) — review queue disabled", _e)
    review_router = None
    _HAS_AL = False
    def should_queue_for_review(_p): return False
    def prediction_entropy(_p): return 0.0

# 1. تهيئة الموجه الديناميكي (يُحمّل جميع الإعدادات من configs/ تلقائياً)
log.info("Initializing OmniDiagRouter...")
try:
    router = OmniDiagRouter(configs_dir="configs")
    log.info(f"Router initialized. Available diseases: {router.get_available_diseases()}")
except Exception as e:
    log.error(f"Failed to initialize router: {e}", exc_info=True)
    raise


# 2. Lifespan — initialise DB + cache on startup
async def init_db() -> None:
    """Create all tables and seed default roles/users if the DB is empty."""
    import uuid as _uuid
    from sqlalchemy import select as _select
    from backend.auth.hashing import hash_password as _hash

    async with engine.begin() as conn:
        # Import all models so Base.metadata is fully populated before create_all
        import backend.db_models  # noqa: F401 — side-effect: registers all ORM models
        await conn.run_sync(Base.metadata.create_all)

    async with AsyncSessionLocal() as db:
        # Seed roles
        for role_name in ("super_admin", "admin", "doctor", "nurse", "viewer"):
            existing = (await db.execute(_select(Role).where(Role.name == role_name))).scalar_one_or_none()
            if not existing:
                db.add(Role(name=role_name, description=role_name.replace("_", " ").title()))

        await db.commit()

        # Seed admin user
        admin_email = os.getenv("ADMIN_EMAIL", "admin@omnidiag.com")
        admin_pw    = os.getenv("ADMIN_PASSWORD", "Admin@123")
        if not (await db.execute(_select(User).where(User.email == admin_email))).scalar_one_or_none():
            admin_role = (await db.execute(_select(Role).where(Role.name == "super_admin"))).scalar_one_or_none()
            admin_user = User(
                id=str(_uuid.uuid4()),
                email=admin_email,
                full_name="Admin User",
                hashed_password=_hash(admin_pw),
                is_active=True,
            )
            if admin_role:
                admin_user.roles.append(admin_role)
            db.add(admin_user)

        # Seed doctor user
        doctor_email = os.getenv("DOCTOR_EMAIL", "doctor@omnidiag.com")
        doctor_pw    = os.getenv("DOCTOR_PASSWORD", "Doctor@123")
        if not (await db.execute(_select(User).where(User.email == doctor_email))).scalar_one_or_none():
            doctor_role = (await db.execute(_select(Role).where(Role.name == "doctor"))).scalar_one_or_none()
            doctor_user = User(
                id=str(_uuid.uuid4()),
                email=doctor_email,
                full_name="Dr. Sarah Al-Khalid",
                hashed_password=_hash(doctor_pw),
                is_active=True,
            )
            if doctor_role:
                doctor_user.roles.append(doctor_role)
            db.add(doctor_user)

        await db.commit()
        log.info("DB initialised — tables created + default users seeded")


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        await init_db()
    except Exception as _e:
        log.error("init_db failed: %s", _e, exc_info=True)
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
if review_router is not None:
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
    all_diseases = []
    for name in router.get_available_diseases():
        info = router.get_disease_info(name)
        all_diseases.append({"name": name, "info": info})
    # Only return diseases that have a trained model on disk
    available = [d for d in all_diseases if d["info"] and d["info"].get("available", True)]
    return {
        "diseases": available,
        "all_registered": [d["name"] for d in all_diseases],
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
    current_user: Optional[User] = Depends(get_optional_user),
    db: AsyncSession = Depends(get_db),
):
    """
    تشخيص مريض لمرض معين. يعمل بدون تسجيل دخول — النتائج تُحفظ فقط للمستخدمين المسجلين.

    يتطلب دور: doctor | nurse | super_admin

    النتائج مؤقتة في الكاش لـ 5 دقائق للمدخلات المتكررة.
    يمكن ربط النتيجة بمريض موجود عبر ?patient_id=<uuid>
    """
    _predict_log = logging.getLogger("omnidiag.predict")
    try:
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

        # Persist to predictions table only for authenticated users
        if current_user:
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

    except HTTPException:
        raise
    except Exception as e:
        _predict_log.error("Predict failed for disease=%s: %s: %s", disease, type(e).__name__, e)
        _predict_log.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail={
                "error": f"Prediction failed: {type(e).__name__}: {str(e)}",
                "hint": "Check that all model artifacts (weights + preprocessors) are present on the server.",
            },
        )


@app.post("/api/v4/{disease}/explain", response_model=ExplainResponse, tags=["Clinical Diagnosis"])
@limiter.limit(LIMIT_CLINICAL)
async def explain_disease(
    request: Request,
    disease: str,
    patient: dict,
    patient_id: str = Query(None, description="Optional patient UUID to link SHAP data"),
    current_user: Optional[User] = Depends(get_optional_user),
    db: AsyncSession = Depends(get_db),
):
    """
    تفسير قرار التشخيص باستخدام SHAP. يعمل بدون تسجيل دخول.
    """
    try:
        patient_data = _validate_patient_input(disease, patient)

        _explain_log.debug(f"Explain request for disease={disease}")
        result = router.explain(disease, patient_data)
        _explain_log.debug("Explain completed successfully")

        # Persist prediction + SHAP data only for authenticated users
        if current_user:
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
    _user: Optional[User] = Depends(get_optional_user),
):
    """
    توليد سيناريوهات "ماذا لو" لتقليل المخاطر. يعمل بدون تسجيل دخول.
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

    if _generate_report is None:
        # Fallback when anthropic package is not installed
        from backend.llm.report_generator import _rule_based_report
        report_text = _rule_based_report(
            disease_display, body.probability, body.label, body.shap_values, body.features
        )
        return {"disease": body.disease, "report": report_text, "source": "rule_based"}

    result = await _generate_report(
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
    disease: Optional[str] = None
    use_bert: bool = False


@app.post(
    "/api/v4/parse-notes",
    tags=["Clinical Diagnosis"],
    summary="Extract structured features from free-text clinical notes (no auth required)",
)
@limiter.limit("20/minute")
async def parse_notes(
    request: Request,
    body: NotesParseRequest,
) -> Dict[str, Any]:
    if _parse_clinical_note is None:
        return {"extracted_features": {}, "mapped_features": {}, "field_count": 0}
    extracted = _parse_clinical_note(body.note, use_bert=body.use_bert)
    mapped: Dict[str, Any] = {}
    if body.disease and _HAS_NLP:
        try:
            from backend.nlp.notes_parser import map_to_disease_schema
            mapped = map_to_disease_schema(extracted, body.disease)
        except Exception:
            pass
    return {
        "extracted_features": extracted,
        "mapped_features": mapped,
        "field_count": len(mapped) or len(extracted),
    }
