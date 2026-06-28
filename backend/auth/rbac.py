"""
OmniDiag — Role-Based Access Control (RBAC)
=============================================
Provides require_role(), a FastAPI dependency factory that enforces role
membership before allowing access to a protected endpoint.

Roles (defined in the `roles` table via seed_db.py):
    super_admin — full system access, including admin dashboards
    doctor      — clinical endpoints (predict, explain, counterfactuals)
    nurse       — same clinical endpoints as doctor
    viewer      — read-only (schemas, disease list); no predictions

Usage:
    from backend.auth.rbac import require_role

    # Only doctors, nurses, and admins may call /predict
    @app.post("/api/v4/{disease}/predict")
    async def predict(user: User = Depends(require_role("doctor", "nurse", "super_admin"))):
        ...

    # Only super_admin may access /admin/* routes
    @app.get("/admin/audit-logs")
    async def audit_logs(user: User = Depends(require_role("super_admin"))):
        ...

CLINICAL_ROLES and ADMIN_ROLES are exported as convenience constants.
"""

from fastapi import Depends, HTTPException, status

from backend.auth.dependencies import get_current_active_user
from backend.db_models.user import User

# ── Role constants ────────────────────────────────────────────────────────────
CLINICAL_ROLES = ("doctor", "nurse", "super_admin")
ADMIN_ROLES = ("super_admin",)


def require_role(*allowed_roles: str):
    """
    FastAPI dependency factory — enforces that the authenticated user holds
    at least one of the specified roles.

    Args:
        *allowed_roles: One or more role name strings the user must have.

    Returns:
        A FastAPI dependency callable that resolves to the authenticated User.

    Raises:
        401 — no valid JWT token present (via get_current_active_user).
        403 — user is authenticated but lacks the required role.

    Example:
        Depends(require_role("doctor", "super_admin"))
    """
    async def _check_role(
        current_user: User = Depends(get_current_active_user),
    ) -> User:
        user_role_names = {role.name for role in current_user.roles}

        if not user_role_names.intersection(allowed_roles):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "error": "Insufficient permissions",
                    "required_roles": list(allowed_roles),
                    "your_roles": sorted(user_role_names) or ["(none assigned)"],
                    "hint": "Contact your system administrator to request access.",
                },
            )

        return current_user

    # Preserve a readable name in FastAPI's dependency graph
    _check_role.__name__ = f"require_role({'|'.join(allowed_roles)})"
    return _check_role
