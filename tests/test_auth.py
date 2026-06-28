"""
Tests — Feature 2.2: JWT Authentication
"""

import pytest


class TestRegister:
    async def test_register_success(self, client, db_tables):
        resp = await client.post("/auth/register", json={
            "email": "new@test.com",
            "password": "Newpass1",
            "full_name": "New User",
        })
        assert resp.status_code == 201
        assert "successfully" in resp.json()["message"]

    async def test_register_duplicate_email(self, client, seeded_db):
        # First registration
        await client.post("/auth/register", json={
            "email": "dup@test.com", "password": "Dup12345", "full_name": "Dup"
        })
        # Second with same email
        resp = await client.post("/auth/register", json={
            "email": "dup@test.com", "password": "Dup12345", "full_name": "Dup2"
        })
        assert resp.status_code == 409

    async def test_register_invalid_email(self, client, db_tables):
        resp = await client.post("/auth/register", json={
            "email": "not-an-email", "password": "Valid123", "full_name": "ValidName"
        })
        assert resp.status_code == 422

    async def test_register_password_too_short(self, client, db_tables):
        resp = await client.post("/auth/register", json={
            "email": "short@test.com", "password": "Ab1", "full_name": "Short User"
        })
        assert resp.status_code == 422

    async def test_register_password_no_digit(self, client, db_tables):
        resp = await client.post("/auth/register", json={
            "email": "nodigit@test.com", "password": "NoDigitPass", "full_name": "No Digit"
        })
        assert resp.status_code == 422

    async def test_register_normalises_email_to_lowercase(self, client, db_tables):
        resp = await client.post("/auth/register", json={
            "email": "UPPER@TEST.COM", "password": "Upper123", "full_name": "UpperUser"
        })
        assert resp.status_code == 201


class TestLogin:
    async def test_login_success(self, client, seeded_db):
        resp = await client.post("/auth/login", json={
            "email": "doctor@test.com", "password": "Doctor1234"
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
        assert data["expires_in"] > 0

    async def test_login_sets_httponly_cookies(self, client, seeded_db):
        resp = await client.post("/auth/login", json={
            "email": "doctor@test.com", "password": "Doctor1234"
        })
        assert resp.status_code == 200
        assert "access_token" in resp.cookies
        assert "refresh_token" in resp.cookies

    async def test_login_wrong_password(self, client, seeded_db):
        resp = await client.post("/auth/login", json={
            "email": "doctor@test.com", "password": "WrongPass1"
        })
        assert resp.status_code == 401

    async def test_login_unknown_email(self, client, seeded_db):
        resp = await client.post("/auth/login", json={
            "email": "ghost@test.com", "password": "Ghost1234"
        })
        # Same error as wrong password (prevents user enumeration)
        assert resp.status_code == 401

    async def test_login_case_insensitive_email(self, client, seeded_db):
        resp = await client.post("/auth/login", json={
            "email": "DOCTOR@TEST.COM", "password": "Doctor1234"
        })
        assert resp.status_code == 200


class TestGetMe:
    async def test_get_me_authenticated(self, client, doctor_token):
        resp = await client.get(
            "/auth/me",
            headers={"Authorization": f"Bearer {doctor_token}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["email"] == "doctor@test.com"
        assert data["is_active"] is True

    async def test_get_me_unauthenticated(self, client, db_tables):
        resp = await client.get("/auth/me")
        assert resp.status_code == 401

    async def test_get_me_invalid_token(self, client, db_tables):
        resp = await client.get(
            "/auth/me",
            headers={"Authorization": "Bearer this.is.invalid"},
        )
        assert resp.status_code == 401

    async def test_get_me_returns_roles(self, client, admin_token):
        resp = await client.get(
            "/auth/me",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert resp.status_code == 200
        roles = [r["name"] for r in resp.json()["roles"]]
        assert "super_admin" in roles


class TestRefreshAndLogout:
    async def test_refresh_issues_new_token(self, client, seeded_db):
        # Log in to get cookies
        login_resp = await client.post("/auth/login", json={
            "email": "doctor@test.com", "password": "Doctor1234"
        })
        assert login_resp.status_code == 200
        refresh_cookie = login_resp.cookies.get("refresh_token")

        resp = await client.post(
            "/auth/refresh",
            cookies={"refresh_token": refresh_cookie},
        )
        assert resp.status_code == 200
        assert "access_token" in resp.json()

    async def test_refresh_without_cookie_returns_401(self, client, db_tables):
        resp = await client.post("/auth/refresh")
        assert resp.status_code == 401

    async def test_logout_clears_cookies(self, client, seeded_db):
        await client.post("/auth/login", json={
            "email": "doctor@test.com", "password": "Doctor1234"
        })
        resp = await client.post("/auth/logout")
        assert resp.status_code == 200
        # Cookies should be cleared (set to empty / expired)
        assert resp.json()["message"] == "Logged out successfully"
