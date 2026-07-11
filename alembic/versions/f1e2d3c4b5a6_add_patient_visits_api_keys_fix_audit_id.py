"""add patient_visits, api key columns, fix audit_logs id type

Revision ID: f1e2d3c4b5a6
Revises: da87946a56a6
Create Date: 2026-06-28 00:00:00.000000

Fixes three ORM/schema mismatches found in post-audit review:

1. patient_visits table — defined in PatientVisit ORM model but absent from
   the initial migration. Required for longitudinal risk tracking.

2. users.api_key_hash / users.api_key_expires_at — defined in the User ORM
   model and used by backend/auth/api_key.py, but absent from initial migration.
   Without these columns the X-API-Key auth path raises a DB error at runtime.

3. audit_logs.id type mismatch — initial migration created it as BigInteger
   (autoincrement), but the AuditLog ORM model declares it as String(36) with
   a UUID default. This migration recreates the column as String(36) on
   databases that support ALTER COLUMN; on SQLite a table rebuild is used.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.engine.reflection import Inspector

# revision identifiers, used by Alembic.
revision: str = "f1e2d3c4b5a6"
down_revision: Union[str, Sequence[str], None] = "da87946a56a6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _is_sqlite() -> bool:
    return op.get_bind().dialect.name == "sqlite"


def upgrade() -> None:
    # ── 1. patient_visits table ───────────────────────────────────────────────
    op.create_table(
        "patient_visits",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column(
            "patient_id",
            sa.String(length=36),
            sa.ForeignKey("patients.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("disease", sa.String(), nullable=False),
        sa.Column(
            "visit_date",
            sa.DateTime(timezone=True),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
            nullable=False,
        ),
        sa.Column("features", sa.JSON(), nullable=False),
        sa.Column("risk_score", sa.Float(), nullable=False),
        sa.Column("prediction", sa.Integer(), nullable=False),
        sa.Column("notes", sa.String(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
            nullable=True,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_patient_visits_patient_id", "patient_visits", ["patient_id"])
    op.create_index("ix_patient_visits_disease", "patient_visits", ["disease"])

    # ── 2. API key columns on users ───────────────────────────────────────────
    conn = op.get_bind()
    inspector = Inspector.from_engine(conn)
    existing_cols = {c["name"] for c in inspector.get_columns("users")}

    if "api_key_hash" not in existing_cols:
        op.add_column(
            "users",
            sa.Column("api_key_hash", sa.String(length=255), nullable=True),
        )
        op.create_index("ix_users_api_key_hash", "users", ["api_key_hash"])

    if "api_key_expires_at" not in existing_cols:
        op.add_column(
            "users",
            sa.Column("api_key_expires_at", sa.DateTime(timezone=True), nullable=True),
        )

    # ── 3. Fix audit_logs.id: BigInteger → String(36) ────────────────────────
    # SQLite does not support ALTER COLUMN, so we use a table-rebuild pattern.
    # PostgreSQL supports it directly via batch_alter_table.
    if _is_sqlite():
        with op.batch_alter_table("audit_logs") as batch_op:
            batch_op.alter_column(
                "id",
                existing_type=sa.BigInteger(),
                type_=sa.String(length=36),
                existing_nullable=False,
            )
    else:
        op.alter_column(
            "audit_logs",
            "id",
            existing_type=sa.BigInteger(),
            type_=sa.String(length=36),
            existing_nullable=False,
            postgresql_using="id::text",
        )


def downgrade() -> None:
    # ── Reverse audit_logs.id fix ─────────────────────────────────────────────
    if _is_sqlite():
        with op.batch_alter_table("audit_logs") as batch_op:
            batch_op.alter_column(
                "id",
                existing_type=sa.String(length=36),
                type_=sa.BigInteger(),
                existing_nullable=False,
            )
    else:
        op.alter_column(
            "audit_logs",
            "id",
            existing_type=sa.String(length=36),
            type_=sa.BigInteger(),
            existing_nullable=False,
            postgresql_using="id::bigint",
        )

    # ── Reverse API key columns ───────────────────────────────────────────────
    op.drop_index("ix_users_api_key_hash", table_name="users")
    op.drop_column("users", "api_key_expires_at")
    op.drop_column("users", "api_key_hash")

    # ── Drop patient_visits ───────────────────────────────────────────────────
    op.drop_index("ix_patient_visits_disease", table_name="patient_visits")
    op.drop_index("ix_patient_visits_patient_id", table_name="patient_visits")
    op.drop_table("patient_visits")
