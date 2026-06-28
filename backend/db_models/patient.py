"""
OmniDiag — Patient Model
==========================
Patient demographic records with soft-delete support for GDPR compliance.

Design decisions:
    - mrn (Medical Record Number) is the unique identifier used in clinical
      workflows; it is indexed for fast lookup.
    - deleted_at enables soft-delete: records are not physically removed but
      marked as deleted, preserving referential integrity with predictions.
    - created_by links the patient record to the user who registered them.
"""

import uuid

from sqlalchemy import Column, Date, DateTime, ForeignKey, String, func
from sqlalchemy.orm import relationship

from backend.database import Base


class Patient(Base):
    """
    Patient demographic record.

    Patients are linked to predictions and optionally to the user who created
    their record. Soft-delete is supported via the deleted_at column.
    """

    __tablename__ = "patients"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    mrn = Column(String(50), unique=True, nullable=False, index=True)
    full_name = Column(String(255), nullable=False)
    date_of_birth = Column(Date, nullable=True)
    gender = Column(String(10), nullable=True)
    contact_email = Column(String(255), nullable=True)
    created_by = Column(String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    deleted_at = Column(DateTime(timezone=True), nullable=True)

    # ── Relationships ──────────────────────────────────────────────────────
    created_by_user = relationship("User", back_populates="patients_created", foreign_keys=[created_by])
    predictions = relationship("Prediction", back_populates="patient", foreign_keys="Prediction.patient_id")
    visits = relationship("PatientVisit", back_populates="patient", order_by="PatientVisit.visit_date")

    def __repr__(self) -> str:
        return f"<Patient(id={self.id}, mrn='{self.mrn}', name='{self.full_name}')>"
