"""
OmniDiag — API Key Authentication
====================================
Provides an `X-API-Key` header as an alternative auth method for
programmatic integrations (scripts, dashboards, third-party systems).

Design:
    - Keys are generated as 32-byte random hex strings (256-bit entropy)
    - Only the bcrypt hash is stored in the DB — the plain key is shown
      once at creation and never retrievable again
    - Each key has an optional expiry date
    - A valid API key grants the same permissions as the owning user's roles
    - API key lookup goes through the same `get_current_active_user` path
      so RBAC enforcement is unchanged
"""

import secrets
from datetime import datetime, timezone
from typing import Optional

from fastapi import Header, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.auth.hashing import verify_password
from backend.db_models.user import User


def generate_api_key() -> str:
    """Return a cryptographically random 64-hex-char API key."""
    return secrets.token_hex(32)


async def get_user_by_api_key(
    api_key: str,
    db: AsyncSession,
) -> Optional[User]:
    """
    Look up the user whose stored api_key_hash matches *api_key*.

    Returns None if no match or the key has expired.
    Brute-force is mitigated by bcrypt's cost factor.
    """
    # Fetch all users that have an api_key_hash set (index scan)
    result = await db.execute(
        select(User).where(User.api_key_hash.isnot(None))
    )
    candidates = result.scalars().all()

    for user in candidates:
        if verify_password(api_key, user.api_key_hash):
            # Check expiry
            if user.api_key_expires_at is not None:
                if user.api_key_expires_at < datetime.now(timezone.utc):
                    return None
            return user

    return None


async def get_current_user_via_api_key(
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    db: AsyncSession = None,  # injected by caller
) -> Optional[User]:
    """
    FastAPI dependency: resolve a User from X-API-Key header.

    Returns None if header absent — callers can chain with JWT auth.
    Raises 401 if header is present but invalid/expired.
    """
    if not x_api_key:
        return None

    user = await get_user_by_api_key(x_api_key, db)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": "Invalid or expired API key", "code": "INVALID_API_KEY"},
            headers={"WWW-Authenticate": "ApiKey"},
        )
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"error": "Account is disabled", "code": "ACCOUNT_DISABLED"},
        )
    return user
