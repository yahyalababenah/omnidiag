"""
OmniDiag — Cache Initialisation & Helpers
==========================================
Wraps fastapi-cache2 with a Redis backend (production) or InMemoryBackend
(local dev / tests when REDIS_URL is not set).

Usage in main.py:
    from backend.cache import init_cache, cache_get, cache_set, cache_flush

    @asynccontextmanager
    async def lifespan(app):
        await init_cache()
        yield

    # Manual cache read/write (for POST endpoints where decorator won't work)
    cached = await cache_get(key)
    if cached is None:
        result = compute()
        await cache_set(key, result, ttl=300)

TTLs:
    Schema responses  → 86400 s  (24 h) — changes only on deployment
    Predict responses →   300 s  ( 5 m) — short enough to stay fresh
"""

import hashlib
import json
import logging
import os
from typing import Any, Optional

log = logging.getLogger("omnidiag.cache")

# Will be set by init_cache()
_backend = None
_PREFIX = "omnidiag"


async def init_cache() -> None:
    """
    Initialise the cache backend.

    - If REDIS_URL is set → RedisBackend (production / docker-compose)
    - Otherwise            → InMemoryBackend (local dev & tests)
    """
    global _backend

    from fastapi_cache import FastAPICache
    from fastapi_cache.backends.inmemory import InMemoryBackend

    redis_url = os.getenv("REDIS_URL", "")
    if redis_url:
        try:
            import redis.asyncio as aioredis
            from fastapi_cache.backends.redis import RedisBackend

            client = aioredis.from_url(redis_url, encoding="utf-8", decode_responses=False)
            await client.ping()
            _backend = RedisBackend(client)
            FastAPICache.init(_backend, prefix=_PREFIX)
            log.info("Cache: RedisBackend initialised at %s", redis_url)
        except Exception as exc:
            log.warning("Cache: Redis unavailable (%s) — falling back to InMemoryBackend", exc)
            _backend = InMemoryBackend()
            FastAPICache.init(_backend, prefix=_PREFIX)
    else:
        _backend = InMemoryBackend()
        FastAPICache.init(_backend, prefix=_PREFIX)
        log.info("Cache: InMemoryBackend initialised (set REDIS_URL for Redis)")


def _make_key(*parts: str) -> str:
    return f"{_PREFIX}:" + ":".join(parts)


def predict_cache_key(disease: str, patient_data: dict) -> str:
    """Deterministic cache key for a predict request."""
    fingerprint = hashlib.sha256(
        json.dumps({"disease": disease, "data": patient_data}, sort_keys=True).encode()
    ).hexdigest()[:16]
    return _make_key("predict", disease, fingerprint)


def counterfactuals_cache_key(disease: str, patient_data: dict) -> str:
    """Deterministic cache key for a counterfactuals request."""
    fingerprint = hashlib.sha256(
        json.dumps({"disease": disease, "data": patient_data}, sort_keys=True).encode()
    ).hexdigest()[:16]
    return _make_key("counterfactuals", disease, fingerprint)


def schema_cache_key(disease: str) -> str:
    return _make_key("schema", disease)


async def cache_get(key: str) -> Optional[Any]:
    """Return the cached value for *key*, or None on miss / error."""
    if _backend is None:
        return None
    try:
        from fastapi_cache import FastAPICache
        value = await FastAPICache.get_backend().get(key)
        if value is None:
            return None
        return json.loads(value)
    except Exception as exc:
        log.debug("cache_get error for key=%s: %s", key, exc)
        return None


async def cache_set(key: str, value: Any, ttl: int = 300) -> None:
    """Store *value* under *key* with a TTL in seconds."""
    if _backend is None:
        return
    try:
        from fastapi_cache import FastAPICache
        await FastAPICache.get_backend().set(key, json.dumps(value), ttl)
    except Exception as exc:
        log.debug("cache_set error for key=%s: %s", key, exc)


async def cache_flush() -> int:
    """
    Clear all keys with the OmniDiag prefix.

    Returns the number of keys deleted (best-effort; -1 if count unavailable).
    """
    if _backend is None:
        return 0
    try:
        from fastapi_cache import FastAPICache
        backend = FastAPICache.get_backend()

        # RedisBackend exposes the raw client
        if hasattr(backend, "redis"):
            keys = await backend.redis.keys(f"{_PREFIX}:*")
            if keys:
                await backend.redis.delete(*keys)
            return len(keys)

        # InMemoryBackend: clear the internal store
        if hasattr(backend, "_store"):
            count = len(backend._store)
            backend._store.clear()
            return count

        return -1
    except Exception as exc:
        log.warning("cache_flush error: %s", exc)
        return -1
