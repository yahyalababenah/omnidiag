"""
OmniDiag — ReviewQueue Model
==============================
Stores predictions that sit close to the decision boundary and therefore
warrant expert review.

This table is the core of the Active Learning / Human-in-the-Loop feature.
"Close to the boundary" is threshold-relative, not a fixed 40-60% window: the
diabetes threshold is 0.059776 on the deployment prior, so a 50% probability
there is a confident Positive, not an uncertain case. See
backend/active_learning/sampler.py.

Design decisions:
    - prediction_id has a UNIQUE constraint to enforce one review entry per
      prediction (one-to-one).
    - status tracks the lifecycle: "pending" → "reviewed" | "skipped".
    - reviewer_id is nullable until a reviewer picks up the case.
    - label stores the expert's annotation (0 or 1) separately from the
      model's original prediction; notes stores the reviewer's reasoning for
      it, which is recorded and shown back but never fed to a model.
    - uncertainty_scale / decision_threshold record how uncertainty_score was
      computed, so scores from different releases are never averaged blindly.
"""

import uuid

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, Text, func
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
    # Which probability scale uncertainty_score was computed on, and the
    # decision threshold it was measured around. Without these two a stored
    # score cannot be compared with one written by a different release:
    # rows created before the prevalence correction hold a 0.5-centred
    # entropy on the raw scale and are left NULL here.
    uncertainty_scale = Column(String(16), nullable=True)   # 'raw' | 'corrected'
    decision_threshold = Column(Float, nullable=True)
    reviewer_id = Column(
        String(36),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )
    label = Column(Integer, nullable=True)  # 0 or 1 (expert annotation)
    # Why the reviewer chose that label. The annotate endpoint has always
    # accepted a `notes` field and silently dropped it for want of a column,
    # which threw away the most informative part of an annotation: a bare 0/1
    # says what the expert decided, the note says what they saw. Never a
    # model input — retrain.get_annotated_samples reads `label` and the
    # prediction's own input_features, not this.
    notes = Column(Text, nullable=True)
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
