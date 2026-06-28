"""
OmniDiag — PatientVisit Model
==============================
Stores longitudinal patient visits for time-series risk tracking (Feature 1.2).
Each visit captures the disease, features, and model risk score at a point in time,
enabling the PatientRiskTimeline component to plot risk trajectory over visits.
"""

import uuid
from sqlalchemy import JSON, Column, DateTime, Float, ForeignKey, Integer, String, func
from sqlalchemy.orm import relationship
from backend.database import Base


class PatientVisit(Base):
    """
    A single visit record for a patient, storing features + risk score snapshot.

    Linked to Patient via patient_id. Multiple visits per patient enable
    longitudinal risk trend analysis and LSTM sequence modeling.
    """

    __tablename__ = "patient_visits"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    patient_id = Column(String, ForeignKey("patients.id", ondelete="CASCADE"), nullable=False, index=True)
    disease = Column(String, nullable=False, index=True)
    visit_date = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    features = Column(JSON, nullable=False)
    risk_score = Column(Float, nullable=False)
    prediction = Column(Integer, nullable=False)
    notes = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    patient = relationship("Patient", back_populates="visits")

    def to_dict(self):
        return {
            "id": self.id,
            "patient_id": self.patient_id,
            "disease": self.disease,
            "visit_date": self.visit_date.isoformat() if self.visit_date else None,
            "risk_score": self.risk_score,
            "prediction": self.prediction,
            "notes": self.notes,
            "features": self.features,
        }
