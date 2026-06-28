"""
OmniDiag — Password Hashing
=============================
Bcrypt-based password hashing and verification using passlib.

Using passlib (rather than bcrypt directly) gives us:
  - A clean, high-level API that handles salting automatically
  - Support for future algorithm migration via the 'deprecated' mechanism
  - Consistent behaviour across different bcrypt C-extension versions
"""

from passlib.context import CryptContext

_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(plain: str) -> str:
    """Hash a plaintext password with bcrypt. Returns the hashed string."""
    return _pwd_context.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    """Return True if *plain* matches the stored *hashed* password."""
    return _pwd_context.verify(plain, hashed)
