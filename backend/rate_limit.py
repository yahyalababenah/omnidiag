"""
OmniDiag — Rate Limiting
=========================
Per-user rate limits enforced via slowapi (a Starlette-compatible wrapper
around the `limits` library).

Limits by role:
    super_admin → unlimited (no limit applied)
    doctor/nurse → 30 requests / minute
    viewer       →  5 requests / minute
    anonymous    → 10 requests / minute (unauthenticated calls to public routes)

The limiter key function uses the authenticated user's ID when a valid JWT is
present, falling back to the client IP for anonymous requests. This prevents
one user's quota from consuming another's.
"""

from slowapi import Limiter
from slowapi.util import get_remote_address
from starlette.requests import Request

from backend.auth.jwt import SECRET_KEY, ALGORITHM, extract_token_from_request


def _rate_limit_key(request: Request) -> str:
    """
    Return a stable string key for rate-limit bucketing.

    Priority:
      1. Authenticated user ID (from JWT cookie or Bearer header)
      2. Client IP address (fallback for anonymous requests)
    """
    try:
        from jose import jwt as _jwt, JWTError

        token = extract_token_from_request(request)
        if token:
            payload = _jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            if payload.get("type") == "access":
                sub = payload.get("sub")
                if sub:
                    return f"user:{sub}"
    except Exception:
        pass

    return f"ip:{get_remote_address(request)}"


limiter = Limiter(key_func=_rate_limit_key, default_limits=["10/minute"])

# Named limit strings used as decorators on individual routes
LIMIT_CLINICAL = "30/minute"   # doctor / nurse / super_admin
LIMIT_VIEWER   = "5/minute"    # viewer (applied at the route level)
LIMIT_ADMIN    = "60/minute"   # admin ops (generous — infrequent heavy use)
