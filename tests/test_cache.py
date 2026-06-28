"""
Tests — Feature 2.5: Redis Caching Layer
Tests cache hit/miss behaviour and flush via backend cache utilities.
"""

import pytest
from fastapi_cache import FastAPICache
from fastapi_cache.backends.inmemory import InMemoryBackend

from backend.cache import cache_get, cache_set, cache_flush, predict_cache_key, schema_cache_key

HEART_PAYLOAD = {
    "Age": 55,
    "Sex": "M",
    "ChestPainType": "ATA",
    "RestingBP": 130,
    "Cholesterol": 250,
    "FastingBS": 0,
    "RestingECG": "Normal",
    "MaxHR": 150,
    "ExerciseAngina": "N",
    "Oldpeak": 1.5,
    "ST_Slope": "Up",
}


class TestCacheUtilities:
    async def test_cache_set_and_get(self, db_tables):
        key = "omnidiag:test:unit"
        value = {"hello": "world", "number": 42}
        await cache_set(key, value, ttl=60)
        result = await cache_get(key)
        assert result == value

    async def test_cache_get_missing_key_returns_none(self, db_tables):
        result = await cache_get("omnidiag:test:nonexistent-key-xyz")
        assert result is None

    async def test_cache_set_overwrites(self, db_tables):
        key = "omnidiag:test:overwrite"
        await cache_set(key, {"v": 1}, ttl=60)
        await cache_set(key, {"v": 2}, ttl=60)
        result = await cache_get(key)
        assert result == {"v": 2}

    async def test_cache_flush_removes_omnidiag_keys(self, db_tables):
        await cache_set("omnidiag:test:flush-a", {"a": 1}, ttl=60)
        await cache_set("omnidiag:test:flush-b", {"b": 2}, ttl=60)
        deleted = await cache_flush()
        assert isinstance(deleted, int)
        # Keys should be gone
        assert await cache_get("omnidiag:test:flush-a") is None
        assert await cache_get("omnidiag:test:flush-b") is None

    async def test_predict_cache_key_is_deterministic(self, db_tables):
        key1 = predict_cache_key("heart_disease", HEART_PAYLOAD)
        key2 = predict_cache_key("heart_disease", HEART_PAYLOAD)
        assert key1 == key2

    async def test_predict_cache_key_differs_by_disease(self, db_tables):
        key_heart = predict_cache_key("heart_disease", HEART_PAYLOAD)
        key_diabetes = predict_cache_key("diabetes", HEART_PAYLOAD)
        assert key_heart != key_diabetes

    async def test_predict_cache_key_differs_by_data(self, db_tables):
        payload_a = {**HEART_PAYLOAD, "age": 30}
        payload_b = {**HEART_PAYLOAD, "age": 70}
        assert predict_cache_key("heart_disease", payload_a) != predict_cache_key("heart_disease", payload_b)

    async def test_schema_cache_key_format(self, db_tables):
        key = schema_cache_key("heart_disease")
        assert "heart_disease" in key
        assert "schema" in key


class TestCacheIntegration:
    """End-to-end tests: verify Cache-Hit header changes on repeated calls."""

    async def test_second_predict_call_is_cache_hit(self, client, doctor_token):
        # First call — miss
        await client.post(
            "/api/v4/heart_disease/predict",
            json=HEART_PAYLOAD,
            headers={"Authorization": f"Bearer {doctor_token}"},
        )
        # Second call with identical payload — must be a hit
        resp2 = await client.post(
            "/api/v4/heart_disease/predict",
            json=HEART_PAYLOAD,
            headers={"Authorization": f"Bearer {doctor_token}"},
        )
        assert resp2.status_code == 200
        assert resp2.headers.get("cache-hit") == "true"

    async def test_cache_flush_resets_hit_to_miss(self, client, doctor_token, admin_token):
        # Populate cache
        await client.post(
            "/api/v4/heart_disease/predict",
            json=HEART_PAYLOAD,
            headers={"Authorization": f"Bearer {doctor_token}"},
        )
        # Flush via admin endpoint
        flush_resp = await client.post(
            "/admin/cache/flush",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert flush_resp.status_code == 200

        # Next predict should be a miss again
        resp = await client.post(
            "/api/v4/heart_disease/predict",
            json=HEART_PAYLOAD,
            headers={"Authorization": f"Bearer {doctor_token}"},
        )
        assert resp.headers.get("cache-hit") == "false"

    async def test_schema_cached_on_second_call(self, client, db_tables):
        await client.get("/api/v4/heart_disease/schema")
        resp2 = await client.get("/api/v4/heart_disease/schema")
        assert resp2.status_code == 200
        assert resp2.headers.get("cache-hit") == "true"
