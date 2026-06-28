"""
OmniDiag — Auth FastAPI Dependencies
======================================
Reusable dependencies injected into protected endpoints via Depends().

Token resolution order (first found wins):
    1. Authorization: Bearer <token>  header  (programmatic / API clients)
    2. access_token  HttpOnly cookie           (browser sessions)

Usage:
    @app.get("/api/v4/patients")
    async def list_patients(user: User = Depends(get_current_active_user)):
        ...
"""

from typing import Optional

from fastapi import Cookie, Depends, Header, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.auth.jwt import decode_token
from backend.database import get_db
from backend.db_models.user import User

# auto_error=False lets us fall through to the cookie / API-key check
_bearer = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
    access_token: Optional[str] = Cookie(default=None),
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    db: AsyncSession = Depends(get_db),
) -> User:
    """
    Resolve the currently authenticated user.

    Resolution order (first found wins):
        1. Authorization: Bearer <JWT>  header
        2. access_token HttpOnly cookie
        3. X-API-Key header (hashed key stored on user record)

    Raises:
        401 — no credentials, invalid/expired token, or user not found.
    """
    # ── 1 & 2: JWT (Bearer header or cookie) ─────────────────────────────
    token: Optional[str] = None
    if credentials:
        token = credentials.credentials
    elif access_token:
        token = access_token

    if token:
        payload = decode_token(token, expected_type="access")
        user_id: Optional[str] = payload.get("sub")
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={"error": "Token payload is missing 'sub' claim", "code": "BAD_TOKEN"},
            )
        result = await db.execute(select(User).where(User.id == user_id))
        user = result.scalar_one_or_none()
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={"error": "User associated with this token no longer exists", "code": "USER_NOT_FOUND"},
            )
        return user

    # ── 3: X-API-Key ──────────────────────────────────────────────────────
    if x_api_key:
        from backend.auth.api_key import get_user_by_api_key
        user = await get_user_by_api_key(x_api_key, db)
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={"error": "Invalid or expired API key", "code": "INVALID_API_KEY"},
                headers={"WWW-Authenticate": "ApiKey"},
            )
        return user

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail={"error": "Not authenticated — provide a Bearer token, cookie, or X-API-Key", "code": "NOT_AUTHENTICATED"},
        headers={"WWW-Authenticate": "Bearer"},
    )


async def get_current_active_user(
    current_user: User = Depends(get_current_user),
) -> User:
    """
    Extends get_current_user with an is_active check.

    Raises:
        403 — account exists but has been deactivated by an admin.
    """
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This account has been deactivated. Contact your administrator.",
        )
    return current_user
