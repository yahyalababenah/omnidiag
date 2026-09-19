"""
OmniDiag — Async Database Engine
==================================
Async SQLAlchemy 2.0 engine setup with PostgreSQL support and SQLite fallback
for local development without requiring a running PostgreSQL instance.

Usage:
    from backend.database import get_db

    @router.get("/items")
    async def get_items(db: AsyncSession = Depends(get_db)):
        ...

Environment Variables:
    DATABASE_URL  (optional, default: sqlite+aiosqlite:///./omnidiag_dev.db)
"""

import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import declarative_base

# ── Database URL ────────────────────────────────────────────────────────────
# Default to SQLite for local development; set DATABASE_URL env var in
# production to point to PostgreSQL (e.g. via docker-compose.yml).
DATABASE_URL: str = os.getenv(
    "DATABASE_URL",
    "sqlite+aiosqlite:///./omnidiag_dev.db",
)

# ── Async Engine ────────────────────────────────────────────────────────────
# echo=True logs all SQL statements (useful for debugging, disable in prod).
engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    # SQLite requires this pragma for concurrent async access
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {},
)

# ── Session Factory ─────────────────────────────────────────────────────────
# expire_on_commit=False allows accessing model attributes after session commit.
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

# ── Declarative Base ────────────────────────────────────────────────────────
# All ORM models should inherit from this Base class.
Base = declarative_base()


# ── FastAPI Dependency ──────────────────────────────────────────────────────
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency that provides an async database session.

    Usage:
        @app.get("/items")
        async def get_items(db: AsyncSession = Depends(get_db)):
            result = await db.execute(...)

    The session is automatically closed when the request completes,
    even if an exception occurs.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


# ── Sessions outside Depends() ──────────────────────────────────────────────
@asynccontextmanager
async def app_session(app) -> AsyncGenerator[AsyncSession, None]:
    """
    An AsyncSession for code that runs outside FastAPI's dependency system —
    middleware, startup hooks — resolved the SAME way a route's
    `Depends(get_db)` would be, including `app.dependency_overrides[get_db]`.

    Why this exists: the audit middleware used to open `AsyncSessionLocal()`
    directly. That bypassed the override the test suite installs, so every
    `pytest` run appended audit rows to whatever DATABASE_URL pointed at —
    in local development, the real `omnidiag_dev.db` (7,888 rows had built up
    there). Going through this helper, a test run writes to the test engine
    and production behaves exactly as before.
    """
    provider = app.dependency_overrides.get(get_db, get_db)
    agen = provider()
    session = await agen.__anext__()
    try:
        yield session
    finally:
        await agen.aclose()
