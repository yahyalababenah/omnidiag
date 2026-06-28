"""
OmniDiag — Alembic Environment Configuration
==============================================
Customized to:
  - Detect all ORM models from backend.db_models for autogenerate
  - Read DATABASE_URL from environment variable
  - Strip async driver suffix (+asyncpg, +aiosqlite) for sync migration engine
  - Support both offline (--sql) and online modes
"""

import os
import re
from logging.config import fileConfig

from sqlalchemy import create_engine, pool

from alembic import context

# ── Alembic Config ──────────────────────────────────────────────────────────
config = context.config

# ── Logging ─────────────────────────────────────────────────────────────────
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# ── Import all models so Alembic's autogenerate can detect them ─────────────
# pylint: disable=unused-import
from backend.database import Base
from backend.db_models import (  # noqa: F401
    User,
    user_roles,
    Role,
    Patient,
    Prediction,
    ReviewQueue,
    AuditLog,
)

target_metadata = Base.metadata


# ── Database URL helper ────────────────────────────────────────────────────
def _get_sync_url() -> str:
    """
    Get a sync-compatible database URL for Alembic migrations.

    Alembic uses sync SQLAlchemy engine, but our DATABASE_URL may use
    async drivers (asyncpg, aiosqlite). This function strips the async
    suffix to get the equivalent sync driver URL.

    Examples:
        postgresql+asyncpg://...  →  postgresql://...
        sqlite+aiosqlite://...    →  sqlite://...
    """
    url = os.getenv("DATABASE_URL", config.get_main_option("sqlalchemy.url", "sqlite:///./omnidiag_dev.db"))

    # Strip async driver suffix: +asyncpg, +aiosqlite, etc.
    url = re.sub(r"\+(asyncpg|aiosqlite|asyncmy)", "", url)

    return url


# ── Offline Mode ───────────────────────────────────────────────────────────
def run_migrations_offline() -> None:
    """
    Run migrations in 'offline' mode.

    This configures the context with just a URL and not an Engine,
    though an Engine is acceptable here as well. By skipping the Engine
    creation we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the script output.
    """
    url = _get_sync_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


# ── Online Mode ────────────────────────────────────────────────────────────
def run_migrations_online() -> None:
    """
    Run migrations in 'online' mode.

    Uses a sync engine created from the DATABASE_URL (with async suffix
    stripped) to connect to the database and apply migrations.
    """
    sync_url = _get_sync_url()
    connectable = create_engine(sync_url, poolclass=pool.NullPool)

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
        )

        with context.begin_transaction():
            context.run_migrations()


# ── Entry Point ────────────────────────────────────────────────────────────
if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
