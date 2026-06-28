"""
OmniDiag — Database Models Package
====================================
All ORM models are imported here so that Alembic's autogenerate can detect them.
When adding a new model, import it here.

Usage:
    from backend.db_models import User, Patient, Prediction, ...
"""

from backend.db_models.user import User, user_roles
from backend.db_models.role import Role
from backend.db_models.patient import Patient
from backend.db_models.patient_visit import PatientVisit
from backend.db_models.prediction import Prediction
from backend.db_models.review_queue import ReviewQueue
from backend.db_models.audit_log import AuditLog

__all__ = [
    "User",
    "user_roles",
    "Role",
    "Patient",
    "PatientVisit",
    "Prediction",
    "ReviewQueue",
    "AuditLog",
]
