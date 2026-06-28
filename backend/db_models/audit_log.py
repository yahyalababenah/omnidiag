"""
OmniDiag — AuditLog Model
===========================
Immutable log of every authenticated API request made to the system.

This table is the foundation for compliance auditing (HIPAA, GDPR). Every
authenticated request is logged with user ID, endpoint, method, status code,
client IP, and duration.

Design decisions:
    - id uses BigInteger (auto-increment) since audit logs are append-only
      and will grow large over time; UUIDs are unnecessary here.
    - ip_address supports IPv6 with String(45) max length.
    - created_at is indexed for efficient time-range queries.
    - user_id is nullable to allow logging requests from unauthenticated
      users (e.g., failed login attempts).
"""

import uuid as _uuid

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, func
from sqlalchemy.orm import relationship

from backend.database import Base


class AuditLog(Base):
    """
    Immutable audit record for authenticated API requests.

    Each entry captures who did what, when, from where, and how long it took.
    """

    __tablename__ = "audit_logs"

    id = Column(String(36), primary_key=True, default=lambda: str(_uuid.uuid4()))
    user_id = Column(String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    endpoint = Column(String(255), nullable=False)
    method = Column(String(10), nullable=False)
    status_code = Column(Integer, nullable=True)
    ip_address = Column(String(45), nullable=True)  # IPv6 max length
    duration_ms = Column(Float, nullable=True)
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True,
    )

    # ── Relationships ──────────────────────────────────────────────────────
    user = relationship("User", back_populates="audit_logs", foreign_keys=[user_id])

    def __repr__(self) -> str:
        return (
            f"<AuditLog(id={self.id}, user_id={self.user_id}, "
            f"endpoint='{self.endpoint}', method='{self.method}')>"
        )
