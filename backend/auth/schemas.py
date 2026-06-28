"""
OmniDiag — Auth Pydantic Schemas
==================================
Request and response models for all /auth/* endpoints.
"""

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


# ── Requests ─────────────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    email: str = Field(..., description="User email address")
    password: str = Field(..., min_length=8, description="Password — minimum 8 characters")
    full_name: str = Field(..., min_length=2, description="User's full name")

    @field_validator("email")
    @classmethod
    def email_must_contain_at(cls, v: str) -> str:
        if "@" not in v or "." not in v.split("@")[-1]:
            raise ValueError("Invalid email address format")
        return v.lower().strip()

    @field_validator("password")
    @classmethod
    def password_complexity(cls, v: str) -> str:
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one digit")
        if not any(c.isalpha() for c in v):
            raise ValueError("Password must contain at least one letter")
        return v


class LoginRequest(BaseModel):
    email: str = Field(..., description="Registered email address")
    password: str = Field(..., description="Account password")

    @field_validator("email")
    @classmethod
    def normalise_email(cls, v: str) -> str:
        return v.lower().strip()


# ── Responses ────────────────────────────────────────────────────────────────

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int = Field(..., description="Access token TTL in seconds")


class RoleOut(BaseModel):
    id: int
    name: str

    model_config = {"from_attributes": True}


class UserResponse(BaseModel):
    id: str
    email: str
    full_name: str
    is_active: bool
    roles: List[RoleOut] = []

    model_config = {"from_attributes": True}


class MessageResponse(BaseModel):
    message: str
