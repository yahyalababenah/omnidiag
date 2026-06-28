"""
OmniDiag — Auth Package
========================
JWT-based authentication and authorization for the OmniDiag API.

Modules:
    hashing     — bcrypt password hashing/verification via passlib
    jwt         — JWT token creation and decoding via python-jose
    schemas     — Pydantic request/response models for auth endpoints
    dependencies — FastAPI dependencies (get_current_user, get_current_active_user)
    routes      — /auth/* endpoint definitions
"""
