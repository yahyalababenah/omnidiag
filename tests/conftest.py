"""
OmniDiag Test Configuration
=============================
Shared fixtures for the entire test suite.

Strategy:
  - SQLite in-memory database (no PostgreSQL required)
  - InMemoryBackend for cache (no Redis required)
  - OmniDiagRouter is mocked — no ML models loaded during tests
  - Each test function gets a clean database via function-scoped fixtures
"""

import asyncio
import uuid
from typing import AsyncGenerator
from unittest.mock import MagicMock, patch

import pytest
import pytest_asyncio
from fastapi_cache import FastAPICache
from fastapi_cache.backends.inmemory import InMemoryBackend
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from backend.auth.hashing import hash_password
from backend.database import Base, get_db
from backend.db_models.user import User, user_roles
from backend.db_models.role import Role
from backend.db_models.patient import Patient
from backend.db_models.prediction import Prediction
from backend.db_models.review_queue import ReviewQueue
from backend.db_models.audit_log import AuditLog  # noqa: F401 – ensures table is registered

# ── In-memory SQLite engine ───────────────────────────────────────────────────
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"

test_engine = create_async_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestSessionLocal = async_sessionmaker(
    test_engine, class_=AsyncSession, expire_on_commit=False
)


# ── Mock OmniDiagRouter ───────────────────────────────────────────────────────
def _make_mock_router():
    mock = MagicMock()
    mock.get_available_diseases.return_value = ["heart_disease", "diabetes"]
    mock.get_disease_info.return_value = {
        "name": "heart_disease",
        "display_name": "Coronary Artery Disease Risk",
        "description": "Test disease",
        "version": "5.0.0",
        "model_type": "xgboost",
        "explainer_type": "tree",
    }
    mock.predict.return_value = {
        "prediction": 1,
        "confidence": 0.82,
        "diagnosis": "Positive",
    }
    mock.explain.return_value = {
        "prediction": 1,
        "confidence": 0.82,
        "diagnosis": "Positive",
        "chart_data": [{"feature": "Age", "shap_value": 0.3, "direction": "risk-increasing"}],
        "text_explanation": "High risk due to Age.",
        "base_value": 0.5,
    }
    mock.counterfactuals.return_value = {
        "status": "success",
        "baseline_probability": 0.82,
        "counterfactuals": [],
    }
    return mock


# ── App fixture (module-scoped so import side-effects run once) ───────────────
@pytest.fixture(scope="module")
def mock_router():
    return _make_mock_router()


@pytest.fixture(scope="module")
def app(mock_router):
    """
    Return the FastAPI app with:
      - DB overridden to SQLite in-memory
      - OmniDiagRouter mocked (no ML model loading)
      - Cache using InMemoryBackend
    """
    # Patch the router BEFORE importing main so the module-level router init
    # never actually loads ML models
    with patch("backend.router.OmniDiagRouter", return_value=mock_router):
        import backend.main as main_module

        # Override the module-level `router` variable too
        main_module.router = mock_router

        # Override DB dependency
        async def _override_get_db() -> AsyncGenerator[AsyncSession, None]:
            async with TestSessionLocal() as session:
                yield session

        main_module.app.dependency_overrides[get_db] = _override_get_db

        # Initialise cache with InMemoryBackend and wire backend module variable
        import backend.cache as _cache_module
        _mem_backend = InMemoryBackend()
        _cache_module._backend = _mem_backend
        FastAPICache.init(_mem_backend, prefix="omnidiag-test")

        yield main_module.app

        main_module.app.dependency_overrides.clear()


@pytest_asyncio.fixture(scope="module")
async def db_tables(app):
    """Create all tables once per test module."""
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest_asyncio.fixture
async def db_session(db_tables) -> AsyncGenerator[AsyncSession, None]:
    """Function-scoped session; rolls back after each test for isolation."""
    async with TestSessionLocal() as session:
        yield session
        await session.rollback()


# ── Seed helpers ──────────────────────────────────────────────────────────────

async def _seed_roles(session: AsyncSession):
    """Insert the 4 standard roles idempotently (skip if already present)."""
    from sqlalchemy import select
    for name, desc in [
        ("super_admin", "Full system access"),
        ("doctor", "Clinical access"),
        ("nurse", "Limited clinical access"),
        ("viewer", "Read-only"),
    ]:
        existing = (await session.execute(select(Role).where(Role.name == name))).scalar_one_or_none()
        if existing is None:
            session.add(Role(name=name, description=desc))
    await session.commit()
    result = await session.execute(select(Role))
    return {r.name: r for r in result.scalars().all()}


async def _create_user(session: AsyncSession, email: str, password: str, role_name: str, roles: dict) -> User:
    user = User(
        id=str(uuid.uuid4()),
        email=email,
        hashed_password=hash_password(password),
        full_name=f"Test {role_name.title()}",
        is_active=True,
    )
    session.add(user)
    await session.flush()

    await session.execute(
        user_roles.insert().values(user_id=user.id, role_id=roles[role_name].id)
    )
    await session.commit()
    await session.refresh(user)
    return user


# ── HTTP client fixture ───────────────────────────────────────────────────────

@pytest_asyncio.fixture
async def client(app) -> AsyncGenerator[AsyncClient, None]:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


# ── Pre-seeded client fixtures ────────────────────────────────────────────────

@pytest_asyncio.fixture(scope="module")
async def seeded_db(db_tables):
    """Seed roles + users once per module. Returns a dict with user objects."""
    async with TestSessionLocal() as session:
        roles = await _seed_roles(session)
        from sqlalchemy import select
        # Create users idempotently
        users = {}
        for email, password, role_name in [
            ("admin@test.com", "Admin1234", "super_admin"),
            ("doctor@test.com", "Doctor1234", "doctor"),
            ("viewer@test.com", "Viewer1234", "viewer"),
        ]:
            existing = (await session.execute(select(User).where(User.email == email))).scalar_one_or_none()
            if existing is None:
                u = await _create_user(session, email, password, role_name, roles)
            else:
                u = existing
            users[role_name] = u
        return {"roles": roles, **users}


async def _login(client: AsyncClient, email: str, password: str) -> str:
    """Helper: log in and return the access token."""
    resp = await client.post("/auth/login", json={"email": email, "password": password})
    assert resp.status_code == 200, resp.text
    return resp.json()["access_token"]


@pytest_asyncio.fixture(scope="module")
async def _module_client(app) -> AsyncGenerator[AsyncClient, None]:
    """Module-scoped client used by token fixtures."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


@pytest_asyncio.fixture(scope="module")
async def admin_token(_module_client, seeded_db) -> str:
    return await _login(_module_client, "admin@test.com", "Admin1234")


@pytest_asyncio.fixture(scope="module")
async def doctor_token(_module_client, seeded_db) -> str:
    return await _login(_module_client, "doctor@test.com", "Doctor1234")


@pytest_asyncio.fixture(scope="module")
async def viewer_token(_module_client, seeded_db) -> str:
    return await _login(_module_client, "viewer@test.com", "Viewer1234")
