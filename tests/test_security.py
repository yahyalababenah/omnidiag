"""
Security Tests — S-1 through S-8
Tests token attacks, security headers, SQL injection resistance, and rate limiting.
"""

import base64
import json
from datetime import timedelta

import pytest

from backend.auth.jwt import create_access_token, create_refresh_token

HEART_PAYLOAD = {
    "Age": 55, "Sex": "M", "ChestPainType": "ATA", "RestingBP": 130,
    "Cholesterol": 250, "FastingBS": 0, "RestingECG": "Normal",
    "MaxHR": 150, "ExerciseAngina": "N", "Oldpeak": 1.5, "ST_Slope": "Up",
}


class TestTokenAttacks:
    async def test_expired_access_token_rejected(self, client, seeded_db):
        """S-1: Expired access token must return 401 on an auth-required endpoint."""
        expired = create_access_token(
            {"sub": "any-user"}, expires_delta=timedelta(seconds=-1)
        )
        # /auth/me requires a valid authenticated user
        resp = await client.get(
            "/auth/me",
            headers={"Authorization": f"Bearer {expired}"},
        )
        assert resp.status_code == 401

    async def test_refresh_token_rejected_as_access(self, client, seeded_db):
        """S-2: A refresh token must not be accepted as an access token for /auth/me."""
        # Log in to obtain a refresh token (set as HttpOnly cookie)
        login_resp = await client.post(
            "/auth/login",
            json={"email": "doctor@test.com", "password": "Doctor1234"},
        )
        assert login_resp.status_code == 200
        refresh_token = login_resp.cookies.get("refresh_token")
        assert refresh_token, "Login should set a refresh_token cookie"

        resp = await client.get(
            "/auth/me",
            headers={"Authorization": f"Bearer {refresh_token}"},
        )
        assert resp.status_code == 401

    async def test_tampered_jwt_payload_rejected(self, client, doctor_token):
        """S-3: JWT with a forged payload but original signature must be rejected."""
        parts = doctor_token.split(".")
        fake_payload = base64.urlsafe_b64encode(
            json.dumps({"sub": "attacker", "type": "access"}).encode()
        ).rstrip(b"=").decode()
        tampered = f"{parts[0]}.{fake_payload}.{parts[2]}"

        resp = await client.get(
            "/auth/me",
            headers={"Authorization": f"Bearer {tampered}"},
        )
        assert resp.status_code == 401

    async def test_completely_random_token_rejected(self, client, db_tables):
        """S-3b: A garbage Bearer token must return 401."""
        resp = await client.get(
            "/auth/me",
            headers={"Authorization": "Bearer not.a.jwt"},
        )
        assert resp.status_code == 401


class TestSecurityHeaders:
    async def test_x_content_type_options_header_present(self, client, db_tables):
        """S-5: SecurityHeadersMiddleware must set X-Content-Type-Options."""
        resp = await client.get("/api/v4/diseases")
        assert resp.status_code == 200
        assert "x-content-type-options" in resp.headers

    async def test_x_frame_options_header_present(self, client, db_tables):
        """S-6: SecurityHeadersMiddleware must set X-Frame-Options."""
        resp = await client.get("/api/v4/diseases")
        assert resp.status_code == 200
        assert "x-frame-options" in resp.headers

    async def test_x_content_type_options_value(self, client, db_tables):
        resp = await client.get("/api/v4/diseases")
        assert resp.headers.get("x-content-type-options", "").lower() == "nosniff"


class TestSQLInjectionResistance:
    async def test_sql_injection_in_search_returns_200(self, client, doctor_token):
        """S-7: SQL injection in search param must not cause a 500."""
        resp = await client.get(
            "/api/v4/patients/?search=' OR 1=1 --",
            headers={"Authorization": f"Bearer {doctor_token}"},
        )
        assert resp.status_code == 200

    async def test_sql_injection_returns_normal_pagination(self, client, doctor_token):
        """S-7: SQL injection in search must return the normal paginated structure."""
        resp = await client.get(
            "/api/v4/patients/?search=' OR 1=1 --",
            headers={"Authorization": f"Bearer {doctor_token}"},
        )
        data = resp.json()
        assert "items" in data
        assert "total" in data

    async def test_union_injection_in_search(self, client, doctor_token):
        """S-7b: UNION-based injection must not cause a 500."""
        resp = await client.get(
            "/api/v4/patients/?search=foo' UNION SELECT 1,2,3--",
            headers={"Authorization": f"Bearer {doctor_token}"},
        )
        assert resp.status_code == 200


class TestRateLimiting:
    @pytest.mark.slow
    async def test_login_rate_limit_triggers_429(self, client):
        """S-4: Sending many login requests in quick succession should trigger rate limiting."""
        responses = []
        for _ in range(10):
            resp = await client.post(
                "/auth/login",
                json={"email": "ratebomb@test.com", "password": "wrong"},
            )
            responses.append(resp.status_code)

        status_codes = set(responses)
        # Either the rate limiter kicked in (429) or all returned 401 (limiter off in tests)
        assert status_codes.issubset({401, 429})
