"""
Test-suite isolation guards (WEAKNESS_REGISTER.md P-13).

Every pytest run used to append rows to the developer's real omnidiag_dev.db:
the audit middleware opened its own session instead of going through get_db,
so the test DB override never reached it. These tests pin the fix down.
"""

import pytest
from sqlalchemy import select

from backend.db_models.audit_log import AuditLog
from tests.conftest import TestSessionLocal


def test_backend_engine_is_in_memory_under_pytest():
    import backend.database as database
    assert ":memory:" in database.DATABASE_URL, database.DATABASE_URL
    assert "omnidiag_dev.db" not in str(database.engine.url)


async def test_audit_middleware_writes_to_the_test_database(client, db_tables):
    marker = "/api/v4/isolation-probe-p13"
    await client.get(marker)   # 404 is fine — the middleware logs every path

    async with TestSessionLocal() as db:
        rows = (
            await db.execute(select(AuditLog).where(AuditLog.endpoint == marker))
        ).scalars().all()
    assert len(rows) == 1, "audit row did not reach the test database"


def test_middleware_no_longer_opens_its_own_session():
    import inspect
    import backend.middleware.audit as audit
    src = inspect.getsource(audit)
    assert "AsyncSessionLocal" not in src
    assert "app_session(request.app)" in src


def test_drift_route_imports_resolve():
    """run_drift imported a non-existent `async_session_maker`; it now uses get_db."""
    import inspect
    import backend.monitoring.routes as routes
    import ast
    tree = ast.parse(inspect.getsource(routes))
    imported = {
        alias.name
        for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert "async_session_maker" not in imported
    import backend.database as database
    assert not hasattr(database, "async_session_maker")   # the name never existed
    assert "db" in inspect.signature(routes.run_drift).parameters
