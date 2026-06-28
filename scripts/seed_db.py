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
            "confidence": 0.72,
            "diagnosis": "Positive",
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
            "confidence": 0.91,
            "diagnosis": "Positive",
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
            "confidence": 0.84,
            "diagnosis": "Negative",
        },
    },
]

# Diabetes patients (from frontend/src/mockPatients.js diabetes entries)
DM_PATIENTS = [
    {
        "mrn": "DM-001",
        "full_name": "Layla Mansour",
        "date_of_birth": date(1966, 3, 10),
        "gender": "F",
        "contact_email": "layla.mansour@example.com",
        "prediction": {
            "disease": "diabetes",
            "input_features": {
                "HighBP": 1,
                "HighChol": 1,
                "CholCheck": 1,
                "BMI": 32.4,
                "Smoker": 0,
                "Stroke": 0,
                "HeartDiseaseorAttack": 0,
                "PhysActivity": 0,
                "Fruits": 0,
                "Veggies": 0,
                "HvyAlcoholConsump": 0,
                "AnyHealthcare": 1,
                "NoDocbcCost": 0,
                "GenHlth": 3,
                "MentHlth": 12,
                "PhysHlth": 18,
                "DiffWalk": 1,
                "Sex": 0,
                "Age": 10,
                "Education": 3,
                "Income": 4,
            },
            "prediction": 1,
            "confidence": 0.87,
            "diagnosis": "Positive",
        },
    },
    {
        "mrn": "DM-002",
        "full_name": "Mohammed Al-Sayed",
        "date_of_birth": date(1960, 7, 28),
        "gender": "M",
        "contact_email": "mohammed.alsayed@example.com",
        "prediction": {
            "disease": "diabetes",
            "input_features": {
                "HighBP": 1,
                "HighChol": 1,
                "CholCheck": 1,
                "BMI": 28.7,
                "Smoker": 1,
                "Stroke": 0,
                "HeartDiseaseorAttack": 1,
                "PhysActivity": 0,
                "Fruits": 1,
                "Veggies": 0,
                "HvyAlcoholConsump": 0,
                "AnyHealthcare": 1,
                "NoDocbcCost": 0,
                "GenHlth": 4,
                "MentHlth": 8,
                "PhysHlth": 22,
                "DiffWalk": 1,
                "Sex": 1,
                "Age": 11,
                "Education": 2,
                "Income": 3,
            },
            "prediction": 1,
            "confidence": 0.93,
            "diagnosis": "Positive",
        },
    },
]


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

                    # Create prediction
                    pred = pat_data["prediction"]
                    prediction = Prediction(
                        id=str(uuid.uuid4()),
                        patient_id=patient_id,
                        disease=pred["disease"],
                        input_features=pred["input_features"],
                        prediction=pred["prediction"],
                        confidence=pred["confidence"],
                        diagnosis=pred["diagnosis"],
                        created_by=doctor_id,
                    )
                    session.add(prediction)
                    stats["patients"] += 1
                    stats["predictions"] += 1
                    print(f"  ➕ Created CAD patient: {pat_data['full_name']} ({pat_data['mrn']})")
                else:
                    print(f"  ✓ CAD patient already exists: {pat_data['full_name']}")

            # ── 5. Seed Diabetes patients ─────────────────────────────────
            for pat_data in DM_PATIENTS:
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

                    # Create prediction
                    pred = pat_data["prediction"]
                    prediction = Prediction(
                        id=str(uuid.uuid4()),
                        patient_id=patient_id,
                        disease=pred["disease"],
                        input_features=pred["input_features"],
                        prediction=pred["prediction"],
                        confidence=pred["confidence"],
                        diagnosis=pred["diagnosis"],
                        created_by=doctor_id,
                    )
                    session.add(prediction)
                    stats["patients"] += 1
                    stats["predictions"] += 1
                    print(f"  ➕ Created DM patient: {pat_data['full_name']} ({pat_data['mrn']})")
                else:
                    print(f"  ✓ DM patient already exists: {pat_data['full_name']}")

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
