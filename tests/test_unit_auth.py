"""
Unit Tests — Auth: hashing.py and jwt.py
No HTTP client or database required.
"""

from datetime import timedelta
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from backend.auth.hashing import hash_password, verify_password
from backend.auth.jwt import (
    create_access_token,
    create_refresh_token,
    decode_token,
    extract_token_from_request,
)


# ── U-1 through U-4: Password Hashing ────────────────────────────────────────

class TestHashPassword:
    async def test_hash_returns_nonempty_string(self):
        h = hash_password("mysecret")
        assert isinstance(h, str) and len(h) > 0

    async def test_hash_does_not_store_plaintext(self):
        h = hash_password("mysecret")
        assert "mysecret" not in h

    async def test_two_hashes_of_same_password_differ(self):
        # bcrypt uses random salt → different hashes for same input
        h1 = hash_password("same")
        h2 = hash_password("same")
        assert h1 != h2

    async def test_verify_correct_password(self):
        h = hash_password("correct_horse_battery")
        assert verify_password("correct_horse_battery", h) is True

    async def test_verify_wrong_password(self):
        h = hash_password("correct")
        assert verify_password("wrong", h) is False

    async def test_verify_empty_against_hash(self):
        h = hash_password("nonempty")
        assert verify_password("", h) is False


# ── U-5 through U-9: JWT Operations ──────────────────────────────────────────

class TestJWT:
    async def test_create_access_token_returns_string(self):
        token = create_access_token({"sub": "user-id-123"})
        assert isinstance(token, str) and len(token) > 0

    async def test_decode_access_token_returns_correct_sub(self):
        token = create_access_token({"sub": "user-abc"})
        payload = decode_token(token, expected_type="access")
        assert payload["sub"] == "user-abc"

    async def test_decode_token_has_type_access(self):
        token = create_access_token({"sub": "x"})
        payload = decode_token(token)
        assert payload["type"] == "access"

    async def test_expired_token_raises_401(self):
        token = create_access_token({"sub": "x"}, expires_delta=timedelta(seconds=-1))
        with pytest.raises(HTTPException) as exc_info:
            decode_token(token)
        assert exc_info.value.status_code == 401

    async def test_refresh_token_type(self):
        token = create_refresh_token({"sub": "user-r"})
        payload = decode_token(token, expected_type="refresh")
        assert payload["type"] == "refresh"
        assert payload["sub"] == "user-r"

    async def test_refresh_used_as_access_raises_401(self):
        refresh = create_refresh_token({"sub": "user-r"})
        with pytest.raises(HTTPException) as exc_info:
            decode_token(refresh, expected_type="access")
        assert exc_info.value.status_code == 401

    async def test_tampered_token_raises_401(self):
        import base64, json
        token = create_access_token({"sub": "real-user"})
        header, _, sig = token.split(".")
        fake_payload = base64.urlsafe_b64encode(
            json.dumps({"sub": "attacker", "type": "access"}).encode()
        ).rstrip(b"=").decode()
        tampered = f"{header}.{fake_payload}.{sig}"
        with pytest.raises(HTTPException) as exc_info:
            decode_token(tampered)
        assert exc_info.value.status_code == 401


# ── U-10: extract_token_from_request ─────────────────────────────────────────

class TestExtractToken:
    async def test_extracts_from_cookie(self):
        req = MagicMock()
        req.cookies = {"access_token": "cookie-token"}
        req.headers = {}
        result = extract_token_from_request(req)
        assert result == "cookie-token"

    async def test_extracts_from_bearer_header(self):
        req = MagicMock()
        req.cookies = {}
        req.headers = {"Authorization": "Bearer header-token"}
        result = extract_token_from_request(req)
        assert result == "header-token"

    async def test_returns_none_when_no_token(self):
        req = MagicMock()
        req.cookies = {}
        req.headers = {}
        result = extract_token_from_request(req)
        assert result is None

    async def test_cookie_takes_priority_over_header(self):
        req = MagicMock()
        req.cookies = {"access_token": "cookie-tok"}
        req.headers = {"Authorization": "Bearer header-tok"}
        result = extract_token_from_request(req)
        assert result == "cookie-tok"
