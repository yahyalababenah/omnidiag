"""
Tests — Feature 2.3: Admin Routes (Audit Logs, Cache Flush)
"""

import pytest


class TestAuditLogs:
    async def test_admin_can_list_audit_logs(self, client, admin_token):
        resp = await client.get(
            "/admin/audit-logs",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "items" in data
        assert "total" in data
        assert "page" in data

    async def test_audit_logs_default_pagination(self, client, admin_token):
        resp = await client.get(
            "/admin/audit-logs",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        data = resp.json()
        assert data["page"] == 1
        assert data["limit"] <= 100

    async def test_audit_logs_pagination_params(self, client, admin_token):
        resp = await client.get(
            "/admin/audit-logs?page=2&limit=5",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["page"] == 2
        assert data["limit"] == 5

    async def test_audit_logs_filter_by_method(self, client, admin_token):
        resp = await client.get(
            "/admin/audit-logs?method=POST",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        # All returned items should have method=POST (or list is empty)
        for item in data["items"]:
            assert item["method"] == "POST"

    async def test_audit_logs_filter_by_endpoint(self, client, admin_token):
        resp = await client.get(
            "/admin/audit-logs?endpoint=/auth",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert resp.status_code == 200

    async def test_audit_logs_filter_by_user_id(self, client, admin_token, seeded_db):
        user_id = seeded_db["super_admin"].id
        resp = await client.get(
            f"/admin/audit-logs?user_id={user_id}",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert resp.status_code == 200

    async def test_audit_logs_date_filter(self, client, admin_token):
        resp = await client.get(
            "/admin/audit-logs?start=2020-01-01T00:00:00&end=2099-12-31T23:59:59",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert resp.status_code == 200

    async def test_audit_logs_non_admin_forbidden(self, client, doctor_token):
        resp = await client.get(
            "/admin/audit-logs",
            headers={"Authorization": f"Bearer {doctor_token}"},
        )
        assert resp.status_code == 403


class TestCacheFlush:
    async def test_admin_can_flush_cache(self, client, admin_token):
        resp = await client.post(
            "/admin/cache/flush",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "message" in data
        assert "keys_deleted" in data

    async def test_cache_flush_returns_int_keys_deleted(self, client, admin_token):
        resp = await client.post(
            "/admin/cache/flush",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert isinstance(resp.json()["keys_deleted"], int)

    async def test_doctor_cannot_flush_cache(self, client, doctor_token):
        resp = await client.post(
            "/admin/cache/flush",
            headers={"Authorization": f"Bearer {doctor_token}"},
        )
        assert resp.status_code == 403

    async def test_viewer_cannot_flush_cache(self, client, viewer_token):
        resp = await client.post(
            "/admin/cache/flush",
            headers={"Authorization": f"Bearer {viewer_token}"},
        )
        assert resp.status_code == 403
