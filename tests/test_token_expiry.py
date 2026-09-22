"""
X-6 — an access token must outlive a demo session.

Before this, JWT_ACCESS_TOKEN_EXPIRE_MINUTES defaulted to 15 and the frontend
has no refresh loop, so Admin and Batch started failing with "Could not
validate credentials" a quarter of an hour into the day and the only recovery
was signing in again.

"A token issued N minutes ago" is simulated by issuing one whose remaining
lifetime is the configured TTL minus N. That is exactly the token an
N-minutes-ago login would present now, and it needs no clock patching.
"""

from datetime import timedelta

import pytest
from fastapi import HTTPException

from backend.auth.jwt import (
    ACCESS_TOKEN_EXPIRE_MINUTES,
    REFRESH_TOKEN_EXPIRE_DAYS,
    create_access_token,
    decode_token,
)

# A booth day. The token must survive at least this long unattended.
FULL_DAY_MINUTES = 12 * 60


def token_issued_minutes_ago(user_id: str, minutes: float) -> str:
    """An access token that was minted `minutes` ago under the current TTL."""
    return create_access_token(
        {"sub": user_id},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES) - timedelta(minutes=minutes),
    )


class TestConfiguredLifetime:
    def test_default_access_token_lasts_at_least_a_working_day(self):
        assert ACCESS_TOKEN_EXPIRE_MINUTES >= FULL_DAY_MINUTES, (
            f"access tokens expire after {ACCESS_TOKEN_EXPIRE_MINUTES} min — "
            f"a booth session outlives that and there is no refresh loop (X-6)"
        )

    def test_lifetime_is_overridable_from_the_environment(self, monkeypatch):
        """A deployment with real patient data must be able to shorten it."""
        import importlib

        monkeypatch.setenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "15")
        monkeypatch.setenv("JWT_REFRESH_TOKEN_EXPIRE_DAYS", "1")
        import backend.auth.jwt as jwt_module

        reloaded = importlib.reload(jwt_module)
        try:
            assert reloaded.ACCESS_TOKEN_EXPIRE_MINUTES == 15
            assert reloaded.REFRESH_TOKEN_EXPIRE_DAYS == 1
        finally:
            # Restore the module for every other test in the session.
            monkeypatch.delenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES")
            monkeypatch.delenv("JWT_REFRESH_TOKEN_EXPIRE_DAYS")
            importlib.reload(jwt_module)


class TestExpiryWindow:
    @pytest.mark.parametrize("minutes", [1, 15, 16, 60, 240, FULL_DAY_MINUTES - 1])
    def test_token_still_decodes_across_the_whole_day(self, minutes):
        """15 and 16 are the boundary the old default failed at."""
        token = token_issued_minutes_ago("user-abc", minutes)
        assert decode_token(token)["sub"] == "user-abc"

    def test_token_is_rejected_once_it_really_has_expired(self):
        """The window is long, not absent — expiry must still be enforced."""
        expired = create_access_token({"sub": "user-abc"}, expires_delta=timedelta(seconds=-1))
        with pytest.raises(HTTPException) as exc:
            decode_token(expired)
        assert exc.value.status_code == 401

    def test_token_older_than_the_configured_ttl_is_rejected(self):
        token = token_issued_minutes_ago("user-abc", ACCESS_TOKEN_EXPIRE_MINUTES + 1)
        with pytest.raises(HTTPException) as exc:
            decode_token(token)
        assert exc.value.status_code == 401


class TestExpiryWindowOverHttp:
    """The same window, through the API a signed-in clinician actually uses."""

    @pytest.mark.asyncio
    async def test_admin_route_still_works_20_minutes_into_the_session(
        self, client, seeded_db
    ):
        token = token_issued_minutes_ago(seeded_db["super_admin"].id, minutes=20)
        resp = await client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 200, (
            "a 20-minute-old session was rejected — X-6 has regressed"
        )
        assert resp.json()["email"] == "admin@test.com"

    @pytest.mark.asyncio
    async def test_login_advertises_the_full_window_to_the_client(self, client, seeded_db):
        resp = await client.post(
            "/auth/login", json={"email": "doctor@test.com", "password": "Doctor1234"}
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["expires_in"] >= FULL_DAY_MINUTES * 60

    @pytest.mark.asyncio
    async def test_a_genuinely_expired_token_is_still_refused(self, client, seeded_db):
        expired = create_access_token(
            {"sub": seeded_db["super_admin"].id}, expires_delta=timedelta(seconds=-1)
        )
        resp = await client.get("/auth/me", headers={"Authorization": f"Bearer {expired}"})
        assert resp.status_code == 401
