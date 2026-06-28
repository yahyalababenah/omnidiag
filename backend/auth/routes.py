"""
OmniDiag — Auth Endpoints
===========================
Mounted at /auth/* in main.py.

Endpoints:
    POST /auth/register  — Create a new user account
    POST /auth/login     — Authenticate and issue JWT tokens (cookies + body)
    POST /auth/refresh   — Issue a new access token from a valid refresh cookie
    POST /auth/logout    — Clear both token cookies
    GET  /auth/me        — Return the current authenticated user's profile
"""

import uuid

from fastapi import APIRouter, Cookie, Depends, HTTPException, Response, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.auth.dependencies import get_current_active_user
from backend.auth.hashing import hash_password, verify_password
from backend.auth.jwt import (
    ACCESS_TOKEN_EXPIRE_MINUTES,
    REFRESH_TOKEN_EXPIRE_DAYS,
    create_access_token,
    create_refresh_token,
    decode_token,
)
from backend.auth.schemas import (
    LoginRequest,
    MessageResponse,
    RegisterRequest,
    TokenResponse,
    UserResponse,
)
from backend.database import get_db
from backend.db_models.user import User

router = APIRouter()

# ── Cookie settings ───────────────────────────────────────────────────────────
# secure=True enforces HTTPS-only in production.
# samesite="lax" protects against CSRF while allowing top-level navigation.
_COOKIE_OPTS = dict(httponly=True, secure=True, samesite="lax")


def _set_auth_cookies(response: Response, access_token: str, refresh_token: str) -> None:
    response.set_cookie(
        key="access_token",
        value=access_token,
        max_age=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        **_COOKIE_OPTS,
    )
    response.set_cookie(
        key="refresh_token",
        value=refresh_token,
        max_age=REFRESH_TOKEN_EXPIRE_DAYS * 24 * 60 * 60,
        **_COOKIE_OPTS,
    )


# ── POST /auth/register ───────────────────────────────────────────────────────

@router.post(
    "/register",
    response_model=MessageResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user account",
)
async def register(
    payload: RegisterRequest,
    db: AsyncSession = Depends(get_db),
) -> MessageResponse:
    """
    Create a new user account.

    - Email must be unique.
    - Password is bcrypt-hashed before storage.
    - New accounts have no roles by default; an admin assigns roles separately.
    """
    result = await db.execute(select(User).where(User.email == payload.email))
    if result.scalar_one_or_none() is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"An account with email '{payload.email}' already exists",
        )

    user = User(
        id=str(uuid.uuid4()),
        email=payload.email,
        hashed_password=hash_password(payload.password),
        full_name=payload.full_name,
    )
    db.add(user)
    await db.commit()
    return MessageResponse(message="Account created successfully. An administrator will assign your role.")


# ── POST /auth/login ──────────────────────────────────────────────────────────

@router.post(
    "/login",
    response_model=TokenResponse,
    summary="Authenticate and receive JWT tokens",
)
async def login(
    payload: LoginRequest,
    response: Response,
    db: AsyncSession = Depends(get_db),
) -> TokenResponse:
    """
    Verify credentials and issue an access token + refresh token.

    Tokens are returned:
      - In the response body (access_token) — for API/programmatic clients.
      - As HttpOnly cookies — for browser-based clients (Clinical EMR Mode).

    A single generic error message is returned for both wrong email and wrong
    password to prevent user enumeration attacks.
    """
    result = await db.execute(select(User).where(User.email == payload.email))
    user = result.scalar_one_or_none()

    if user is None or not verify_password(payload.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This account has been deactivated. Contact your administrator.",
        )

    access_token = create_access_token(data={"sub": user.id})
    refresh_token = create_refresh_token(data={"sub": user.id})

    _set_auth_cookies(response, access_token, refresh_token)

    return TokenResponse(
        access_token=access_token,
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )


# ── POST /auth/refresh ────────────────────────────────────────────────────────

@router.post(
    "/refresh",
    response_model=TokenResponse,
    summary="Issue a new access token from a valid refresh token",
)
async def refresh(
    response: Response,
    refresh_token: str = Cookie(default=None),
    db: AsyncSession = Depends(get_db),
) -> TokenResponse:
    """
    Exchange a valid refresh token cookie for a new access token.

    The refresh token is validated and the user's is_active status is
    re-checked so deactivated accounts cannot silently continue via refresh.
    """
    if not refresh_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="No refresh token cookie found — please log in",
        )

    payload = decode_token(refresh_token, expected_type="refresh")
    user_id: str = payload.get("sub", "")

    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()

    if user is None or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or account deactivated",
        )

    new_access_token = create_access_token(data={"sub": user.id})

    response.set_cookie(
        key="access_token",
        value=new_access_token,
        max_age=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        **_COOKIE_OPTS,
    )

    return TokenResponse(
        access_token=new_access_token,
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )


# ── POST /auth/logout ─────────────────────────────────────────────────────────

@router.post(
    "/logout",
    response_model=MessageResponse,
    summary="Clear authentication cookies",
)
async def logout(response: Response) -> MessageResponse:
    """
    Invalidate the session by deleting both auth cookies.

    Note: this does NOT invalidate the JWT itself (stateless design).
    Tokens expire naturally after ACCESS_TOKEN_EXPIRE_MINUTES. For immediate
    revocation, a token denylist (Redis) would be needed — planned for v2.1.
    """
    response.delete_cookie("access_token")
    response.delete_cookie("refresh_token")
    return MessageResponse(message="Logged out successfully")


# ── GET /auth/me ──────────────────────────────────────────────────────────────

@router.get(
    "/me",
    response_model=UserResponse,
    summary="Return the current authenticated user's profile",
)
async def get_me(
    current_user: User = Depends(get_current_active_user),
) -> UserResponse:
    """Return id, email, full_name, is_active, and assigned roles."""
    return current_user
