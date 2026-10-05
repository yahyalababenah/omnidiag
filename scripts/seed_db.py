#!/usr/bin/env python3
"""
OmniDiag — Database Seed Script
=================================
Idempotent script to populate the database with demo data.

Safe to run multiple times — uses ON CONFLICT DO NOTHING / get-or-create
pattern for all seed data.

Usage:
    # Default: uses DATABASE_URL from environment (or SQLite fallback)
    python scripts/seed_db.py

    # Explicit PostgreSQL connection:
    DATABASE_URL=postgresql+asyncpg://omnidiag:omnidiag_pass@localhost:5432/omnidiag_db \\
        python scripts/seed_db.py

Environment Variables:
    DATABASE_URL  (optional, default: sqlite+aiosqlite:///./omnidiag_dev.db)
"""

import asyncio
import os
import sys
import uuid
from datetime import date, datetime, timezone

# Ensure project root is on sys.path so we can import backend modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import bcrypt
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

from backend.database import Base, DATABASE_URL
from backend.db_models import (
    Role,
    User,
    Patient,
    Prediction,
)

# ── Password Hashing ────────────────────────────────────────────────────────
def hash_password(password: str) -> str:
    """Hash a password using bcrypt."""
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


# ── Helpers ─────────────────────────────────────────────────────────────────
def utcnow() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc)


# ── Seed Data ───────────────────────────────────────────────────────────────
ROLES = [
    {"name": "super_admin", "description": "Full system access — user management, audit review, model administration"},
    {"name": "doctor", "description": "Clinical access — predict, explain, counterfactuals, patient records"},
    {"name": "nurse", "description": "Limited clinical access — predict, view patient records"},
    {"name": "viewer", "description": "Read-only access — view predictions and patient data"},
]

USERS = [
    {
        "email": "admin@omnidiag.com",
        "password": "Admin@123",
        "full_name": "Admin User",
        "role": "super_admin",
    },
    {
        "email": "doctor@omnidiag.com",
        "password": "Doctor@123",
        "full_name": "Dr. Sarah Al-Khalid",
        "role": "doctor",
    },
]

# Heart disease patients (from frontend/src/mockPatients.js heart_disease entries)
CAD_PATIENTS = [
    {
        "mrn": "CAD-001",
        "full_name": "Ahmed Al-Rashid",
        "date_of_birth": date(1970, 5, 15),
        "gender": "M",
        "contact_email": "ahmed.alrashid@example.com",
        "prediction": {
            "disease": "heart_disease",
            "input_features": {
                "Age": 54,
                "Sex": "M",
                "ChestPainType": "ATA",
                "RestingBP": 140,
                "Cholesterol": 289,
                "FastingBS": 0,
                "RestingECG": "Normal",
                "MaxHR": 122,
                "ExerciseAngina": "N",
                "Oldpeak": 0.0,
                "ST_Slope": "Flat",
            },
            "prediction": 1,
            # No confidence or diagnosis literal: the shipped model supplies both
            # at seed time (_score_heart). The values that used to sit here --
            # 0.72 / 0.91 / 0.84 -- came from heart_full_tuned.pkl, archived since
            # Gate 8.1, and disagreed with the shipped model by up to 44 points.
        },
    },
    {
        "mrn": "CAD-002",
        "full_name": "Fatima Hassan",
        "date_of_birth": date(1962, 8, 22),
        "gender": "F",
        "contact_email": "fatima.hassan@example.com",
        "prediction": {
            "disease": "heart_disease",
            "input_features": {
                "Age": 62,
                "Sex": "F",
                "ChestPainType": "ASY",
                "RestingBP": 158,
                "Cholesterol": 340,
                "FastingBS": 1,
                "RestingECG": "LVH",
                "MaxHR": 98,
                "ExerciseAngina": "Y",
                "Oldpeak": 2.3,
                "ST_Slope": "Down",
            },
            "prediction": 1,
            # No confidence or diagnosis literal: the shipped model supplies both
            # at seed time (_score_heart). The values that used to sit here --
            # 0.72 / 0.91 / 0.84 -- came from heart_full_tuned.pkl, archived since
            # Gate 8.1, and disagreed with the shipped model by up to 44 points.
        },
    },
    {
        "mrn": "CAD-003",
        "full_name": "Khalid Othman",
        "date_of_birth": date(1979, 11, 3),
        "gender": "M",
        "contact_email": "khalid.othman@example.com",
        "prediction": {
            "disease": "heart_disease",
            "input_features": {
                "Age": 45,
                "Sex": "M",
                "ChestPainType": "NAP",
                "RestingBP": 120,
                "Cholesterol": 210,
                "FastingBS": 0,
                "RestingECG": "Normal",
                "MaxHR": 160,
                "ExerciseAngina": "N",
                "Oldpeak": 0.5,
                "ST_Slope": "Up",
            },
            "prediction": 0,
            # No confidence or diagnosis literal: the shipped model supplies both
            # at seed time (_score_heart). The values that used to sit here --
            # 0.72 / 0.91 / 0.84 -- came from heart_full_tuned.pkl, archived since
            # Gate 8.1, and disagreed with the shipped model by up to 44 points.
        },
    },
]


_HEART_LOADER = None


def _score_heart(input_features: dict) -> dict:
    """
    Score one seeded CAD patient with the shipped model.

    A seeded row must be the model's own output, not a literal. The three CAD
    patients used not to be: the three CAD patients carried hardcoded `confidence` literals
    (0.72, 0.91, 0.84) and a "Positive" / "Negative" label. Those belong to
    heart_full_tuned.pkl, which has not shipped since Gate 8.1. The shipped model
    scores the same three patients at 0.476, 0.465 and 0.087 and calls two of
    them UNCERTAIN, so the demo database was showing a reviewer risk figures from
    an archived model -- 91% where the current model says 46.5%.

    Also returns the conformal decision and its interval, so a seeded row is
    indistinguishable from one written by /predict.

    Deliberately has no fallback: a fabricated
    probability in a clinical demo database is worse than no row.
    """
    global _HEART_LOADER
    if _HEART_LOADER is None:
        from backend.router import OmniDiagRouter

        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        _HEART_LOADER = OmniDiagRouter(
            configs_dir=os.path.join(root, "configs")
        )._get_loader("heart_disease")

    from backend.probability_scale import scale_of_result

    result = _HEART_LOADER.predict(input_features)
    return {
        "prediction": int(result["prediction"]),
        "confidence": float(result["confidence"]),
        "diagnosis": result["diagnosis"],
        "probability_scale": scale_of_result(result).value,
        "decision": result.get("decision"),
        "probability_lower": result.get("probability_lower"),
        "probability_upper": result.get("probability_upper"),
    }


# ── Main Seeder ─────────────────────────────────────────────────────────────
async def seed_database(db_url: str) -> None:
    """
    Seed the database with initial demo data.

    This function is idempotent — safe to run multiple times.
    Uses ON CONFLICT DO NOTHING / get-or-create patterns throughout.
    """
    # Create engine and session
    engine = create_async_engine(db_url, echo=False)
    session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    # Stats tracker
    stats = {"roles": 0, "users": 0, "patients": 0, "predictions": 0}

    async with session_factory() as session:
        async with session.begin():
            # ── 1. Create tables if they don't exist ───────────────────────
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)

            # ── 2. Seed roles ─────────────────────────────────────────────
            for role_data in ROLES:
                # Check if role exists
                result = await session.execute(
                    text("SELECT id FROM roles WHERE name = :name"),
                    {"name": role_data["name"]},
                )
                existing = result.scalar_one_or_none()
                if existing is None:
                    role = Role(name=role_data["name"], description=role_data["description"])
                    session.add(role)
                    stats["roles"] += 1
                    print(f"  ➕ Created role: {role_data['name']}")
                else:
                    print(f"  ✓ Role already exists: {role_data['name']}")

            await session.flush()  # Ensure roles have IDs

            # ── 3. Seed users ─────────────────────────────────────────────
            for user_data in USERS:
                result = await session.execute(
                    text("SELECT id FROM users WHERE email = :email"),
                    {"email": user_data["email"]},
                )
                existing = result.scalar_one_or_none()
                if existing is None:
                    user = User(
                        id=str(uuid.uuid4()),
                        email=user_data["email"],
                        hashed_password=hash_password(user_data["password"]),
                        full_name=user_data["full_name"],
                        is_active=True,
                    )
                    session.add(user)
                    await session.flush()  # Get user.id

                    # Assign role
                    role_result = await session.execute(
                        text("SELECT id FROM roles WHERE name = :name"),
                        {"name": user_data["role"]},
                    )
                    role_id = role_result.scalar_one()
                    await session.execute(
                        text(
                            "INSERT INTO user_roles (user_id, role_id, assigned_at) "
                            "VALUES (:user_id, :role_id, :assigned_at)"
                        ),
                        {
                            "user_id": user.id,
                            "role_id": role_id,
                            "assigned_at": utcnow(),
                        },
                    )
                    stats["users"] += 1
                    print(f"  ➕ Created user: {user_data['email']} (role: {user_data['role']})")
                else:
                    print(f"  ✓ User already exists: {user_data['email']}")

            # Get doctor user ID for created_by fields
            doctor_result = await session.execute(
                text("SELECT id FROM users WHERE email = 'doctor@omnidiag.com'"),
            )
            doctor_id = doctor_result.scalar_one()

            # ── 4. Seed CAD patients ──────────────────────────────────────
            for pat_data in CAD_PATIENTS:
                result = await session.execute(
                    text("SELECT id FROM patients WHERE mrn = :mrn"),
                    {"mrn": pat_data["mrn"]},
                )
                existing = result.scalar_one_or_none()
                if existing is None:
                    patient_id = str(uuid.uuid4())
                    patient = Patient(
                        id=patient_id,
                        mrn=pat_data["mrn"],
                        full_name=pat_data["full_name"],
                        date_of_birth=pat_data["date_of_birth"],
                        gender=pat_data["gender"],
                        contact_email=pat_data["contact_email"],
                        created_by=doctor_id,
                    )
                    session.add(patient)
                    await session.flush()

                    # Create prediction from the real model, not a literal
                    pred = pat_data["prediction"]
                    scored = _score_heart(pred["input_features"])
                    prediction = Prediction(
                        id=str(uuid.uuid4()),
                        patient_id=patient_id,
                        disease=pred["disease"],
                        input_features=pred["input_features"],
                        prediction=scored["prediction"],
                        confidence=scored["confidence"],
                        probability_scale=scored["probability_scale"],
                        diagnosis=scored["diagnosis"],
                        decision=scored["decision"],
                        probability_lower=scored["probability_lower"],
                        probability_upper=scored["probability_upper"],
                        created_by=doctor_id,
                    )
                    session.add(prediction)
                    stats["patients"] += 1
                    stats["predictions"] += 1
                    print(
                        f"  ➕ Created CAD patient: {pat_data['full_name']} ({pat_data['mrn']}) "
                        f"— model: {scored['diagnosis']}, "
                        f"{scored['confidence']:.1%} ({scored['decision']})"
                    )
                else:
                    print(f"  ✓ CAD patient already exists: {pat_data['full_name']}")

            # (BRFSS diabetes patients were seeded here; that module is retired —
            # backend/retired_diseases.py — and gate B4 removed its seeding.)

        # ── Commit is handled by `async with session.begin()` ─────────────

    # ── Print summary ──────────────────────────────────────────────────────
    print()
    print("=" * 50)
    print("✅ Database seeding complete!")
    print("=" * 50)
    print(f"  ✅ Seeded {stats['roles']} roles")
    print(f"  ✅ Seeded {stats['users']} users")
    print(f"  ✅ Seeded {stats['patients']} patients")
    print(f"  ✅ Seeded {stats['predictions']} predictions")
    print("=" * 50)
    print()
    print("Demo credentials:")
    print("  Admin:  admin@omnidiag.com / Admin@123")
    print("  Doctor: doctor@omnidiag.com / Doctor@123")
    print()

    await engine.dispose()


def main() -> None:
    """Entry point — read DATABASE_URL from environment and run the seeder."""
    db_url = os.getenv("DATABASE_URL", DATABASE_URL)
    print(f"🌱 OmniDiag Database Seeder")
    print(f"   Database URL: {db_url}")
    print()

    asyncio.run(seed_database(db_url))


if __name__ == "__main__":
    main()
