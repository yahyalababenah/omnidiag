# OmniDiag Feature 2.1 — PostgreSQL Database Layer: Implementation Plan

## 1. Overview

Add a full async PostgreSQL persistence layer to the OmniDiag backend. The system is currently completely stateless; this feature adds 7 database tables, async SQLAlchemy infrastructure, Alembic migrations, Docker Compose orchestration, and an idempotent seed script — **without modifying any existing endpoint logic**.

---

## 2. Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Async ORM** | SQLAlchemy 2.0 `AsyncSession` | Future-proof for concurrent requests; aligns with FastAPI's `async` capabilities |
| **SQLite fallback** | `sqlite+aiosqlite:///./omnidiag_dev.db` | Zero-dependency local dev; no PostgreSQL install required |
| **UUID strategy** | `String(36)` for SQLite compat; `uuid.uuid4` default factory | SQLite has no native UUID; string storage works identically on PostgreSQL |
| **Soft delete** | `deleted_at` nullable timestamp on `patients` | GDPR compliance; data recovery capability |
| **Naming** | Tables: `snake_case` | Models: `PascalCase` | Standard SQLAlchemy convention |
| **Idempotent seeds** | `ON CONFLICT DO NOTHING` for roles/users | Safe to run `seed_db.py` multiple times |
| **Additive only** | No changes to `main.py`, `router.py`, `schemas.py` | Database layer is purely infrastructure; existing stateless endpoints continue working |

---

## 3. Entity Relationship Design

### 3.1 Tables Overview

```
┌─────────────────┐     ┌───────────────────┐     ┌──────────────────────┐
│     roles       │     │    user_roles     │     │       users          │
├─────────────────┤     ├───────────────────┤     ├──────────────────────┤
│ PK id: Integer  │◄────│ FK role_id        │────►│ PK id: UUID          │
│ name: String(50)│     │ FK user_id        │     │ email: String(255)   │
│ description     │     │ assigned_at       │     │ hashed_password      │
└─────────────────┘     └───────────────────┘     │ full_name            │
                                                  │ is_active: Boolean   │
                                                  │ created_at           │
                                                  │ updated_at           │
                                                  └──────────┬───────────┘
                                                             │
                                            ┌────────────────┼─────────────────┐
                                            │                │                 │
                                            ▼                ▼                 ▼
                                  ┌──────────────────┐ ┌──────────────┐ ┌──────────────┐
                                  │    patients      │ │ predictions  │ │ audit_logs   │
                                  ├──────────────────┤ ├──────────────┤ ├──────────────┤
                                  │ PK id: UUID      │ │ PK id: UUID  │ │ PK id: BigInt│
                                  │ mrn: String(50)  │ │ FK patient_id│ │ FK user_id   │
                                  │ full_name        │ │ disease      │ │ endpoint     │
                                  │ date_of_birth    │ │ input_features│ │ method       │
                                  │ gender           │ │ prediction   │ │ status_code  │
                                  │ contact_email    │ │ confidence   │ │ ip_address   │
                                  │ FK created_by    │ │ diagnosis    │ │ duration_ms  │
                                  │ created_at       │ │ shap_chart   │ │ created_at   │
                                  │ deleted_at       │ │ FK created_by│ └──────────────┘
                                  └──────────────────┘ │ created_at   │
                                                        └──────┬───────┘
                                                               │
                                                               ▼
                                                        ┌──────────────────┐
                                                        │  review_queue    │
                                                        ├──────────────────┤
                                                        │ PK id: UUID      │
                                                        │ FK prediction_id │
                                                        │ uncertainty_score│
                                                        │ FK reviewer_id   │
                                                        │ label            │
                                                        │ reviewed_at      │
                                                        │ status           │
                                                        │ created_at       │
                                                        └──────────────────┘
```

### 3.2 Relationship Summary

| Parent | Child | Type | FK Column |
|--------|-------|------|-----------|
| `users` | `user_roles` | 1:N | `user_roles.user_id` |
| `roles` | `user_roles` | 1:N | `user_roles.role_id` |
| `users` | `patients` | 1:N | `patients.created_by` |
| `users` | `predictions` | 1:N | `predictions.created_by` |
| `users` | `audit_logs` | 1:N | `audit_logs.user_id` |
| `users` | `review_queue` | 1:N | `review_queue.reviewer_id` |
| `patients` | `predictions` | 1:N | `predictions.patient_id` |
| `predictions` | `review_queue` | 1:1 | `review_queue.prediction_id` |

---

## 4. File Structure (What Will Be Created/Modified)

```
Heart_Disease_Project/
├── .env.example                          ← NEW: env var template
├── docker-compose.yml                    ← NEW: postgres + backend services
├── requirements.txt                      ← MODIFY: add 6 packages
├── docs/
│   └── erd.md                           ← NEW: ERD documentation
├── backend/
│   ├── database.py                       ← NEW: async engine + get_db()
│   └── db_models/                        ← NEW: SQLAlchemy model package
│       ├── __init__.py
│       ├── user.py                       ← User + user_roles table
│       ├── role.py                       ← Role table
│       ├── patient.py                    ← Patient table
│       ├── prediction.py                ← Prediction table
│       ├── review_queue.py              ← ReviewQueue table
│       └── audit_log.py                 ← AuditLog table
├── alembic/                              ← NEW: Alembic migration env
│   ├── env.py                            ← Customized to use our models
│   ├── script.py.mako
│   └── versions/
│       └── 0001_initial_schema.py        ← Auto-generated migration
├── alembic.ini                           ← NEW: Alembic config
└── scripts/
    └── seed_db.py                        ← NEW: idempotent database seeder
```

---

## 5. Implementation Steps

### Step 5.1: Update `requirements.txt` (Task 2.1.1)

Append these 6 packages below the existing entries:

| Package | Version | Purpose |
|---------|---------|---------|
| `sqlalchemy` | >=2.0.0 | Async ORM |
| `asyncpg` | >=0.29.0 | PostgreSQL async driver |
| `alembic` | >=1.13.0 | Schema migrations |
| `psycopg2-binary` | >=2.9.0 | Sync driver for Alembic |
| `python-dotenv` | >=1.0.0 | `.env` file loading |
| `aiosqlite` | >=0.19.0 | SQLite async driver (local dev) |
| `passlib[bcrypt]` | >=1.7.4 | Password hashing for seed script |

**Constraint:** Preserve ALL existing entries; do not modify versions.

---

### Step 5.2: Create `docs/erd.md` (Task 2.1.2)

Document all 7 tables with:
- Table name and purpose
- Column names, types, constraints (PK, FK, unique, nullable, defaults)
- Foreign key relationships
- Indexes

---

### Step 5.3: Create `backend/database.py` (Task 2.1.3)

```python
# pseudocode
import os
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./omnidiag_dev.db")

engine = create_async_engine(DATABASE_URL, echo=False)
AsyncSessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
Base = declarative_base()

async def get_db():
    async with AsyncSessionLocal() as session:
        yield session
```

**Key details:**
- Read `DATABASE_URL` from env var; fallback to `sqlite+aiosqlite` for local dev
- Use `create_async_engine()` (not sync)
- Expose `Base` for models to inherit
- `get_db()` is an async generator — ready to be used as a FastAPI `Depends()`

---

### Step 5.4: Create ORM Models in `backend/db_models/` (Task 2.1.4)

**Package design:** One file per table, all imported in `__init__.py` so Alembic can detect them.

#### `backend/db_models/user.py`
```python
# User model
# - id: UUID, pk, default=uuid4
# - email: String(255), unique, not null, indexed
# - hashed_password: String(255), not null
# - full_name: String(255), not null
# - is_active: Boolean, default=True
# - created_at: DateTime(timezone=True), server_default=func.now()
# - updated_at: DateTime(timezone=True), onupdate=func.now()
# - Relationships: roles (M2M via user_roles), predictions, audit_logs
```

#### `backend/db_models/role.py`
```python
# Role model
# - id: Integer, pk, autoincrement
# - name: String(50), unique ("super_admin", "doctor", "nurse", "viewer")
# - description: String(255), nullable

# user_roles association table (not a model class)
# - user_id: UUID, FK -> users.id, PK composite
# - role_id: Integer, FK -> roles.id, PK composite
# - assigned_at: DateTime, server_default=func.now()
```

#### `backend/db_models/patient.py`
```python
# Patient model
# - id: UUID, pk, default=uuid4
# - mrn: String(50), unique, indexed
# - full_name: String(255), not null
# - date_of_birth: Date, nullable
# - gender: String(10), nullable
# - contact_email: String(255), nullable
# - created_by: UUID, FK -> users.id, nullable
# - created_at: DateTime(timezone=True), server_default=func.now()
# - deleted_at: DateTime, nullable  (soft delete)
# - Relationships: predictions
```

#### `backend/db_models/prediction.py`
```python
# Prediction model
# - id: UUID, pk, default=uuid4
# - patient_id: UUID, FK -> patients.id, nullable (allow anonymous)
# - disease: String(100), not null, indexed
# - input_features: JSON, not null
# - prediction: Integer, not null (0 or 1)
# - confidence: Float, not null (0.0-1.0)
# - diagnosis: String(100), nullable
# - shap_chart_data: JSON, nullable
# - created_by: UUID, FK -> users.id, nullable
# - created_at: DateTime(timezone=True), server_default=func.now()
# - Relationships: patient, created_by user, review_queue (uselist=False)
```

#### `backend/db_models/review_queue.py`
```python
# ReviewQueue model
# - id: UUID, pk, default=uuid4
# - prediction_id: UUID, FK -> predictions.id, unique
# - uncertainty_score: Float, nullable
# - reviewer_id: UUID, FK -> users.id, nullable
# - label: Integer, nullable (0 or 1)
# - reviewed_at: DateTime, nullable
# - status: String(20), default="pending"
# - created_at: DateTime(timezone=True), server_default=func.now()
```

#### `backend/db_models/audit_log.py`
```python
# AuditLog model
# - id: BigInteger, pk, autoincrement
# - user_id: UUID, FK -> users.id, nullable
# - endpoint: String(255), not null
# - method: String(10), not null
# - status_code: Integer, nullable
# - ip_address: String(45), nullable (supports IPv6)
# - duration_ms: Float, nullable
# - created_at: DateTime(timezone=True), server_default=func.now(), indexed
```

#### `backend/db_models/__init__.py`
```python
# Import all models so Alembic detects them
from backend.db_models.user import User, user_roles
from backend.db_models.role import Role
from backend.db_models.patient import Patient
from backend.db_models.prediction import Prediction
from backend.db_models.review_queue import ReviewQueue
from backend.db_models.audit_log import AuditLog
```

---

### Step 5.5: Initialize Alembic + Create Migration (Task 2.1.5)

1. **Run `alembic init alembic`** from project root — creates `alembic/` dir and `alembic.ini`
2. **Edit `alembic/env.py`:**
   - Import `Base` from `backend.database`
   - Import all models from `backend.db_models` (ensures Alembic detects them)
   - Set `target_metadata = Base.metadata`
   - Configure `sqlalchemy.url` from `DATABASE_URL` env var
3. **Run `alembic revision --autogenerate -m "initial_schema"`**
4. **Verify** the generated migration in `alembic/versions/` creates all 7 tables
5. **DO NOT run `alembic upgrade head`** — leave for the user to execute manually

**Important:** Because Alembic uses sync driver, `env.py` must use `psycopg2-binary` for the `sqlalchemy.url` when connecting to PostgreSQL, but for autogenerate purposes we can also use SQLite. The `env.py` should parse `DATABASE_URL` and potentially replace `+asyncpg` / `+aiosqlite` with the sync equivalent.

---

### Step 5.6: Create `.env.example` and `docker-compose.yml` (Task 2.1.6)

**`.env.example`** — template with:
- `DATABASE_URL` (PostgreSQL default, SQLite fallback commented out)
- `JWT_SECRET_KEY`, `JWT_ALGORITHM`, `JWT_ACCESS_TOKEN_EXPIRE_MINUTES`
- `CORS_ALLOWED_ORIGINS`

**`docker-compose.yml`** — two services:
- `postgres`: PostgreSQL 15 Alpine with health check
- `backend`: Build from Dockerfile, depends on postgres (healthy)
- Named volume `postgres_data` for persistence

---

### Step 5.7: Create `scripts/seed_db.py` (Task 2.1.7)

Standalone script (not a FastAPI endpoint). Logic:

```
1. Connect to DB using DATABASE_URL from environment
2. Create all tables (Base.metadata.create_all on sync engine)
3. Insert 4 roles: super_admin, doctor, nurse, viewer (ON CONFLICT DO NOTHING)
4. Hash passwords using passlib bcrypt
5. Insert 2 demo users: admin@omnidiag.com / doctor@omnidiag.com
6. Insert 3 CAD patients (Ahmed Al-Rashid, Fatima Hassan, Khalid Othman)
7. Insert 2 Diabetes patients (Layla Mansour, Mohammed Al-Sayed)
8. Insert 1 prediction record per patient (using mock patient data)
9. Print summary
```

**Idempotency:** Use `ON CONFLICT DO NOTHING` for roles and users (unique constraints on `roles.name` and `users.email`). Patients use `mrn` uniqueness.

---

## 6. Potential Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| **SQLite UUID incompatibility** | Model errors | Use `String(36)` for all UUID PKs; `TypeDecorator` for UUID if needed |
| **Alembic async/sync driver mismatch** | Migration fails | `env.py` strips `+asyncpg`/`+aiosqlite` from URL for sync connection |
| **Existing sync endpoints don't use DB** | No persistence for predictions | Expected behavior — DB integration into existing endpoints is Feature 2.4 |
| **passlib bcrypt on Alpine Docker** | Build failure | Ensure `libffi-dev` is available; `passlib[bcrypt]` pulls `bcrypt` package |
| **psycopg2-binary on slim images** | ImportError | `psycopg2-binary` includes C extensions; works on `python:3.10-slim` |

---

## 7. Acceptance Criteria Verification

| Criterion | How to Verify |
|-----------|---------------|
| `pip install -r requirements.txt` succeeds | Run the command; check for exit code 0 |
| `uvicorn backend.main:app --reload` starts | No import errors; `GET /` returns 200 |
| `GET /` works | Returns disease list (same as before) |
| `POST /api/v4/heart_disease/predict` works | Returns valid prediction (no DB needed) |
| `python scripts/seed_db.py` runs | Prints seeding summary |
| `alembic/versions/` has migration | At least one `.py` file in the directory |
| `docker-compose up postgres` works | Container starts, health check passes |
| `alembic upgrade head` creates tables | Run against PostgreSQL; tables appear in `\dt` |

---

## 8. Files That Must NOT Be Touched

- `backend/main.py`
- `backend/router.py`
- `backend/schemas.py`
- `backend/model_loader.py`
- `backend/ensemble_loader.py`
- `backend/shap_service.py`
- `backend/counterfactual_generator.py`
- `configs/` (any file)
- `features/` (any file)
- `models/` (any file)
- `frontend/` (any file)
