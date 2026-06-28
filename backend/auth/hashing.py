"""
OmniDiag — Password Hashing
=============================
Bcrypt-based password hashing and verification using the bcrypt library directly.
passlib 1.7.4 is incompatible with bcrypt>=4.0.0 on Python 3.10 (HF Spaces),
so we call bcrypt directly to avoid the version-detection crash.
"""

import bcrypt


def hash_password(plain: str) -> str:
    """Hash a plaintext password with bcrypt. Returns the hashed string."""
    return bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    """Return True if *plain* matches the stored *hashed* password."""
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
    except Exception:
        return False
