"""
OmniDiag — Admin Endpoints
============================
Mounted at /admin/* in main.py. All endpoints require the super_admin role.

Endpoints:
    GET  /admin/audit-logs        — Paginated audit log with date-range + user filter
    POST /admin/cache/flush       — Invalidate all cached responses after a model update
    POST /admin/api-keys          — Issue a new API key for a user
    DELETE /admin/api-keys/{user_id} — Revoke a user's API key
"""

from datetime import datetime, timezone
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import and_, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.auth.rbac import require_role, ADMIN_ROLES
from backend.database import get_db
from backend.db_models.audit_log import AuditLog
from backend.db_models.user import User

router = APIRouter()


# ── Response schema ───────────────────────────────────────────────────────────

class AuditLogOut(BaseModel):
    id: str
    user_id: Optional[str]
    endpoint: str
    method: str
    status_code: Optional[int]
    ip_address: Optional[str]
    duration_ms: Optional[float]
    created_at: datetime

    model_config = {"from_attributes": True}


class AuditLogPage(BaseModel):
    total: int
    page: int
    limit: int
    pages: int
    items: List[AuditLogOut]


# ── GET /admin/audit-logs ─────────────────────────────────────────────────────

@router.get(
    "/audit-logs",
    response_model=AuditLogPage,
    summary="Paginated audit log (super_admin only)",
    description=(
        "Returns a paginated list of audit log entries. "
        "Supports filtering by user_id, date range, HTTP method, and endpoint prefix."
    ),
)
async def list_audit_logs(
    # Pagination
    page: int = Query(1, ge=1, description="Page number (1-indexed)"),
    limit: int = Query(50, ge=1, le=500, description="Records per page"),
    # Filters
    user_id: Optional[str] = Query(None, description="Filter by user UUID"),
    method: Optional[str] = Query(None, description="HTTP method filter (GET, POST, ...)"),
    endpoint: Optional[str] = Query(None, description="Endpoint prefix filter (e.g. /api/v4)"),
    start: Optional[datetime] = Query(None, description="Earliest created_at (ISO-8601)"),
    end: Optional[datetime] = Query(None, description="Latest created_at (ISO-8601)"),
    # Auth
    _: object = Depends(require_role(*ADMIN_ROLES)),
    db: AsyncSession = Depends(get_db),
) -> AuditLogPage:
    """
    Returns audit log entries ordered by created_at descending (newest first).

    All filters are optional and combinable:
        ?user_id=abc&method=POST&start=2025-01-01T00:00:00Z&end=2025-12-31T23:59:59Z
    """
    filters = []

    if user_id:
        filters.append(AuditLog.user_id == user_id)
    if method:
        filters.append(AuditLog.method == method.upper())
    if endpoint:
        filters.append(AuditLog.endpoint.startswith(endpoint))
    if start:
        # Ensure timezone-aware for comparison
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        filters.append(AuditLog.created_at >= start)
    if end:
        if end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)
        filters.append(AuditLog.created_at <= end)

    where_clause = and_(*filters) if filters else True

    # Total count
    count_q = select(func.count()).select_from(AuditLog).where(where_clause)
    total: int = (await db.execute(count_q)).scalar_one()

    # Paginated items
    offset = (page - 1) * limit
    items_q = (
        select(AuditLog)
        .where(where_clause)
        .order_by(AuditLog.created_at.desc())
        .offset(offset)
        .limit(limit)
    )
    rows = (await db.execute(items_q)).scalars().all()

    pages = max(1, -(-total // limit))  # ceiling division

    return AuditLogPage(
        total=total,
        page=page,
        limit=limit,
        pages=pages,
        items=rows,
    )


# ── POST /admin/cache/flush ───────────────────────────────────────────────────

class FlushResponse(BaseModel):
    message: str
    keys_deleted: int


@router.post(
    "/cache/flush",
    response_model=FlushResponse,
    summary="Flush all cached responses (super_admin only)",
    description="Invalidates every OmniDiag cache key. Call after deploying a new model.",
)
async def flush_cache(
    _: object = Depends(require_role(*ADMIN_ROLES)),
) -> FlushResponse:
    from backend.cache import cache_flush
    deleted = await cache_flush()
    return FlushResponse(
        message="Cache flushed successfully",
        keys_deleted=deleted,
    )


# ── POST /admin/api-keys ──────────────────────────────────────────────────────

class ApiKeyRequest(BaseModel):
    user_id: str = Field(..., description="UUID of the user to issue a key for")
    expires_at: Optional[datetime] = Field(
        None, description="Optional expiry datetime (ISO-8601, UTC). Omit for non-expiring key."
    )


class ApiKeyResponse(BaseModel):
    message: str
    user_id: str
    api_key: str = Field(..., description="Plain-text key — shown once, store it securely")
    expires_at: Optional[datetime]


@router.post(
    "/api-keys",
    response_model=ApiKeyResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Issue an API key for a user (super_admin only)",
    description=(
        "Generates a 64-hex-character API key for the specified user. "
        "The key is shown exactly once — the stored value is hashed and cannot be recovered."
    ),
)
async def issue_api_key(
    body: ApiKeyRequest,
    _: object = Depends(require_role(*ADMIN_ROLES)),
    db: AsyncSession = Depends(get_db),
) -> ApiKeyResponse:
    result = await db.execute(select(User).where(User.id == body.user_id))
    user = result.scalar_one_or_none()
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": f"User {body.user_id} not found", "code": "USER_NOT_FOUND"},
        )

    from backend.auth.api_key import generate_api_key
    from backend.auth.hashing import hash_password

    plain_key = generate_api_key()
    user.api_key_hash = hash_password(plain_key)
    user.api_key_expires_at = body.expires_at
    await db.commit()

    return ApiKeyResponse(
        message="API key issued successfully. Store it securely — it will not be shown again.",
        user_id=user.id,
        api_key=plain_key,
        expires_at=body.expires_at,
    )


# ── DELETE /admin/api-keys/{user_id} ─────────────────────────────────────────

class RevokeResponse(BaseModel):
    message: str
    user_id: str


@router.delete(
    "/api-keys/{user_id}",
    response_model=RevokeResponse,
    summary="Revoke a user's API key (super_admin only)",
)
async def revoke_api_key(
    user_id: str,
    _: object = Depends(require_role(*ADMIN_ROLES)),
    db: AsyncSession = Depends(get_db),
) -> RevokeResponse:
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": f"User {user_id} not found", "code": "USER_NOT_FOUND"},
        )

    user.api_key_hash = None
    user.api_key_expires_at = None
    await db.commit()

    return RevokeResponse(
        message="API key revoked. The user can no longer authenticate via X-API-Key.",
        user_id=user_id,
    )
