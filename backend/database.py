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
