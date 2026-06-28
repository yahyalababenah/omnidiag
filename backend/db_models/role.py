"""
OmniDiag — Role Model
======================
Defines the four access control roles used in the OmniDiag RBAC system:
    - super_admin: Full system access
    - doctor: Clinical access (predict, explain, counterfactuals)
    - nurse: Limited clinical access
    - viewer: Read-only access

Table:
    - roles:  Role definitions
"""

from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import relationship

from backend.database import Base
from backend.db_models.user import user_roles


class Role(Base):
    """
    Access control role definition.

    Each role has a unique name and an optional description. Users are linked
    to roles via the user_roles association table (many-to-many).
    """

    __tablename__ = "roles"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(50), unique=True, nullable=False)
    description = Column(String(255), nullable=True)

    # ── Relationships ──────────────────────────────────────────────────────
    users = relationship("User", secondary=user_roles, back_populates="roles", lazy="selectin")

    def __repr__(self) -> str:
        return f"<Role(id={self.id}, name='{self.name}')>"
