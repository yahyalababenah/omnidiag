"""
Tests — the async database layer's greenlet dependency is declared
==================================================================
SQLAlchemy's asyncio extension needs greenlet. From SQLAlchemy 2.1 greenlet is
only installed through the `asyncio` extra, and the image resolves 2.1.x. Until
B7 it reached the image solely as a dependency of flwr, which nothing used:
removing flwr would have broken the first database query in the image while the
local suite (SQLAlchemy 2.0.x, where greenlet is a hard dependency) stayed green
(CL-4 in docs/phase9/FINDINGS_REGISTER.md).
"""

from pathlib import Path

import greenlet  # noqa: F401 -- a missing greenlet must fail here, not be skipped
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

REQUIREMENTS = Path(__file__).resolve().parent.parent / "requirements.txt"


async def test_an_async_session_round_trips(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'g.db'}")
    try:
        async with async_sessionmaker(engine, class_=AsyncSession)() as session:
            await session.execute(text("CREATE TABLE t (v INTEGER)"))
            await session.execute(text("INSERT INTO t VALUES (42)"))
            await session.commit()
            assert (await session.execute(text("SELECT v FROM t"))).scalar_one() == 42
    finally:
        await engine.dispose()


def test_requirements_declare_the_asyncio_extra():
    """The image installs from requirements.txt, so greenlet must come from an
    explicit line there, not from whichever other package happens to pull it."""
    lines = [l.split("#")[0].strip().replace(" ", "") for l in REQUIREMENTS.read_text().splitlines()]
    assert any(l.lower().startswith("sqlalchemy[asyncio]") for l in lines), \
        "requirements.txt must install sqlalchemy[asyncio] (it brings greenlet)"
