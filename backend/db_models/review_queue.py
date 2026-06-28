"""
OmniDiag — ReviewQueue Model
==============================
Stores uncertain predictions (confidence 40–60%) that require expert review.

This table is the core of the Active Learning / Human-in-the-Loop feature.
When the system produces a prediction with low confidence, it can be flagged
for a human expert (doctor) to review and provide a label.

Design decisions:
    - prediction_id has a UNIQUE constraint to enforce one review entry per
      prediction (one-to-one).
    - status tracks the lifecycle: "pending" → "reviewed" | "skipped".
    - reviewer_id is nullable until a reviewer picks up the case.
    - label stores the expert's annotation (0 or 1) separately from the
      model's original prediction.
"""

import uuid

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, func
from sqlalchemy.orm import relationship

from backend.database import Base


class ReviewQueue(Base):
    """
    A prediction flagged for expert review due to uncertainty.

    Links an uncertain prediction to a human reviewer for annotation.
    """

    __tablename__ = "review_queue"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    prediction_id = Column(
        String(36),
        ForeignKey("predictions.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
    )
    uncertainty_score = Column(Float, nullable=True)
    reviewer_id = Column(
        String(36),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )
    label = Column(Integer, nullable=True)  # 0 or 1 (expert annotation)
    reviewed_at = Column(DateTime(timezone=True), nullable=True)
    status = Column(String(20), nullable=False, default="pending")  # pending | reviewed | skipped
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    # ── Relationships ──────────────────────────────────────────────────────
    prediction = relationship("Prediction", back_populates="review_entry", foreign_keys=[prediction_id])
    reviewer = relationship("User", back_populates="review_actions", foreign_keys=[reviewer_id])

    def __repr__(self) -> str:
        return (
            f"<ReviewQueue(id={self.id}, prediction_id={self.prediction_id}, "
            f"status='{self.status}')>"
        )
