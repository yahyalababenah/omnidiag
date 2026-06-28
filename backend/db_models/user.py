"""
OmniDiag — User Model
=======================
Represents a system user (doctor, nurse, admin, viewer).
Uses bcrypt-hashed passwords and UUID primary keys stored as String(36)
for cross-compatibility with both SQLite and PostgreSQL.

Tables:
    - users:  System user accounts
    - user_roles:  Join table (many-to-many) linking users to roles
"""

import uuid

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Table,
    func,
)
from sqlalchemy.orm import relationship

from backend.database import Base


# ── Association Table ──────────────────────────────────────────────────────
user_roles = Table(
    "user_roles",
    Base.metadata,
    Column("user_id", String(36), ForeignKey("users.id", ondelete="CASCADE"), primary_key=True),
    Column("role_id", Integer, ForeignKey("roles.id", ondelete="CASCADE"), primary_key=True),
    Column("assigned_at", DateTime, server_default=func.now(), nullable=False),
)


# ── User Model ──────────────────────────────────────────────────────────────
class User(Base):
    """
    System user account.

    Each user can have multiple roles (via user_roles join table) and can be
    linked to predictions, patients, audit logs, and review actions they
    created or performed.
    """

    __tablename__ = "users"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    # API key for programmatic integrations (hashed, nullable)
    api_key_hash = Column(String(255), nullable=True, index=True)
    api_key_expires_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at = Column(
        DateTime(timezone=True),
        onupdate=func.now(),
        nullable=True,
    )

    # ── Relationships ──────────────────────────────────────────────────────
    roles = relationship("Role", secondary=user_roles, back_populates="users", lazy="selectin")
    predictions = relationship(
        "Prediction",
        back_populates="created_by_user",
        foreign_keys="Prediction.created_by",
    )
    patients_created = relationship(
        "Patient",
        back_populates="created_by_user",
        foreign_keys="Patient.created_by",
    )
    audit_logs = relationship("AuditLog", back_populates="user")
    review_actions = relationship(
        "ReviewQueue",
        back_populates="reviewer",
        foreign_keys="ReviewQueue.reviewer_id",
    )

    def __repr__(self) -> str:
        return f"<User(id={self.id}, email='{self.email}', full_name='{self.full_name}')>"
