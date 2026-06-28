"""
OmniDiag — JWT Token Utilities
================================
Creates and decodes HS256-signed JWT tokens for access and refresh flows.

Token anatomy:
    {
        "sub":  "<user_id>",   # subject — UUID string
        "type": "access" | "refresh",
        "exp":  <unix timestamp>
    }

Environment variables (all optional, sensible dev defaults applied):
    JWT_SECRET_KEY                  — signing secret (MUST be changed in production)
    JWT_ALGORITHM                   — default HS256
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES — default 15
"""

import os
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import HTTPException, status
from jose import JWTError, jwt
from starlette.requests import Request

# ── Configuration ────────────────────────────────────────────────────────────
SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "dev-secret-do-not-use-in-production")
ALGORITHM: str = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "15"))
REFRESH_TOKEN_EXPIRE_DAYS: int = 7

_CREDENTIALS_EXCEPTION = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Could not validate credentials",
    headers={"WWW-Authenticate": "Bearer"},
)


def extract_token_from_request(request: Request) -> Optional[str]:
    """
    Extract a raw JWT string from an incoming request.

    Resolution order:
        1. access_token HttpOnly cookie (browser sessions)
        2. Authorization: Bearer <token> header (API clients)

    Returns None if neither is present.
    """
    token = request.cookies.get("access_token")
    if not token:
        auth = request.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            token = auth[7:]
    return token or None


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Encode *data* as a signed access JWT.

    Args:
        data: Payload dict — must include {"sub": user_id}.
        expires_delta: Custom TTL; defaults to ACCESS_TOKEN_EXPIRE_MINUTES.

    Returns:
        Encoded JWT string.
    """
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode.update({"exp": expire, "type": "access"})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def create_refresh_token(data: dict) -> str:
    """
    Encode *data* as a signed refresh JWT (7-day TTL).

    Args:
        data: Payload dict — must include {"sub": user_id}.

    Returns:
        Encoded JWT string.
    """
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode.update({"exp": expire, "type": "refresh"})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str, expected_type: str = "access") -> dict:
    """
    Decode and validate a JWT token.

    Args:
        token: Raw JWT string.
        expected_type: "access" or "refresh" — raises 401 on mismatch to
                       prevent refresh tokens being used as access tokens.

    Returns:
        Decoded payload dict.

    Raises:
        HTTPException 401 if the token is invalid, expired, or wrong type.
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        raise _CREDENTIALS_EXCEPTION

    if payload.get("type") != expected_type:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Expected token type '{expected_type}', got '{payload.get('type')}'",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return payload
