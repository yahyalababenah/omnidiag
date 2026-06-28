"""
OmniDiag — Audit Logging Middleware
=====================================
Starlette BaseHTTPMiddleware that writes an AuditLog record to the database
for every API request that passes through it.

What is logged:
    - user_id    — extracted from the JWT (access_token cookie or Bearer header);
                   NULL for unauthenticated requests
    - endpoint   — URL path (e.g. /api/v4/heart_disease/predict)
    - method     — HTTP verb
    - status_code — response status
    - ip_address — client IP (supports IPv6, String(45))
    - duration_ms — wall-clock time from first byte to last byte

What is NOT logged (to reduce noise):
    - GET /         (health check — polled every few seconds by HF Spaces)
    - GET /docs, /redoc, /openapi.json  (Swagger UI assets)

Design notes:
    - The middleware decodes the JWT itself (without the full FastAPI dependency
      chain) because middleware runs before route handlers and has no access to
      Depends().  A failed decode is silently ignored — the request still proceeds;
      only user_id is set to NULL in the log.
    - DB errors in the audit path are caught and logged to stderr so they never
      surface to the client.  Audit logging is best-effort; it must not break
      the request flow.
"""

import time
import logging
from typing import Optional

from jose import JWTError, jwt
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from backend.database import AsyncSessionLocal
from backend.db_models.audit_log import AuditLog

log = logging.getLogger("omnidiag.audit")

# Paths that are never logged (too noisy, no clinical value)
_SKIP_PATHS = frozenset({"/", "/docs", "/redoc", "/openapi.json", "/favicon.ico"})


def _extract_user_id(request: Request) -> Optional[str]:
    """
    Silently try to extract the user_id (sub) from the JWT.

    Returns None on any failure (missing token, expired, invalid signature).
    """
    from backend.auth.jwt import SECRET_KEY, ALGORITHM, extract_token_from_request  # lazy import avoids circular

    token = extract_token_from_request(request)
    if not token:
        return None

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        if payload.get("type") == "access":
            return payload.get("sub")
    except JWTError:
        pass

    return None


class AuditMiddleware(BaseHTTPMiddleware):
    """
    Records one AuditLog row per API request.

    Mount AFTER CORSMiddleware so preflight OPTIONS requests are not logged.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        # Skip noisy paths
        if request.url.path in _SKIP_PATHS:
            return await call_next(request)

        user_id = _extract_user_id(request)
        start = time.perf_counter()

        response: Response = await call_next(request)

        duration_ms = (time.perf_counter() - start) * 1000

        # Best-effort DB write — never raise to the client
        try:
            async with AsyncSessionLocal() as db:
                entry = AuditLog(
                    user_id=user_id,
                    endpoint=request.url.path,
                    method=request.method,
                    status_code=response.status_code,
                    ip_address=request.client.host if request.client else None,
                    duration_ms=round(duration_ms, 2),
                )
                db.add(entry)
                await db.commit()
        except Exception as exc:  # noqa: BLE001
            log.warning("AuditMiddleware: failed to write log entry — %s: %s", type(exc).__name__, exc)

        return response
