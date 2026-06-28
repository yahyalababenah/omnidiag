"""
OmniDiag — Security Headers Middleware (Feature 4.7)
======================================================
Adds OWASP-recommended security headers to every response.

Headers added:
  - Strict-Transport-Security (HSTS): enforces HTTPS for 1 year
  - X-Content-Type-Options: prevents MIME-type sniffing
  - X-Frame-Options: prevents clickjacking
  - Content-Security-Policy: restricts resource loading origins
  - Referrer-Policy: limits referrer information leakage
  - Permissions-Policy: disables unused browser APIs
  - X-Request-ID: echoes the request_id set by main.py (traceability)

HIPAA alignment:
  - HSTS ensures data is always encrypted in transit
  - CSP prevents XSS that could leak PHI
  - X-Content-Type-Options prevents content injection attacks

Usage (in main.py):
    from backend.middleware.security import SecurityHeadersMiddleware
    app.add_middleware(SecurityHeadersMiddleware)
"""

import os
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Injects OWASP-recommended HTTP security headers into every response.
    Safe for both production (HTTPS) and development (HTTP).
    """

    def __init__(self, app, enforce_https: bool = True) -> None:
        super().__init__(app)
        self._enforce_https = enforce_https

    async def dispatch(self, request: Request, call_next) -> Response:
        response: Response = await call_next(request)

        # Strict-Transport-Security — only meaningful over HTTPS
        if self._enforce_https:
            response.headers["Strict-Transport-Security"] = (
                "max-age=31536000; includeSubDomains; preload"
            )

        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = (
            "camera=(), microphone=(), geolocation=(), payment=()"
        )
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
            "font-src 'self' https://fonts.gstatic.com; "
            "img-src 'self' data: blob:; "
            "connect-src 'self'; "
            "frame-ancestors 'none';"
        )

        # Echo request ID for end-to-end tracing
        request_id = getattr(request.state, "request_id", None)
        if request_id:
            response.headers["X-Request-ID"] = str(request_id)

        return response
