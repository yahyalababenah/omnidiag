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

TTLs (all overridable from the environment — see the constants below):
    Schema responses         → 86400 s (24 h) — changes only on deployment
    Predict responses        →  3600 s ( 1 h)
    Counterfactual responses → 43200 s (12 h) — generation costs 9-14 s

Probability scale and the cache
-------------------------------
A cached /predict payload carries a probability, and a probability is only
meaningful together with its scale. For up to the 300 s TTL after a deploy,
entries written by the previous release are still live — and before the
prevalence correction those held raw-prior probabilities.

Two independent defences, because the failure is silent and clinical:

  1. PROBABILITY_SCALE_CONTRACT is part of every predict/counterfactual cache
     key, so bumping it makes every old entry unreachable rather than stale.
  2. cached_payload_matches_scale() re-checks the payload itself before it is
     served, so an entry that somehow survives is dropped instead of returned.
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

# Bump whenever the meaning of a cached probability changes — a new scale, a
# new threshold basis, a renamed probability field. v2 = prevalence-corrected
# probabilities (backend/prevalence_correction.py).
PROBABILITY_SCALE_CONTRACT = "v2"


def _ttl_from_env(var: str, default: int) -> int:
    """
    Read a TTL in seconds from the environment, falling back to *default*.

    A malformed or negative value falls back rather than raising: a typo in a
    Space secret must not stop the API from starting, and losing caching is a
    slowdown, not a wrong answer.
    """
    raw = os.getenv(var, "")
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        log.warning("Cache: %s=%r is not an integer — using default %ds", var, raw, default)
        return default
    if value < 0:
        log.warning("Cache: %s=%d is negative — using default %ds", var, value, default)
        return default
    return value


# A cached counterfactual is deterministic for a given input, and generating
# one costs 9-14 s for diabetes. A 1 h TTL meant a demo that started before
# lunch was cold again after it; 12 h covers a full judging day from a single
# warm-up run (scripts/warmup_demo_cache.py).
COUNTERFACTUALS_TTL_SECONDS = _ttl_from_env("CACHE_TTL_COUNTERFACTUALS", 43_200)
PREDICT_TTL_SECONDS = _ttl_from_env("CACHE_TTL_PREDICT", 3_600)
SCHEMA_TTL_SECONDS = _ttl_from_env("CACHE_TTL_SCHEMA", 86_400)


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
    """Deterministic cache key for a predict request, scoped to the scale contract."""
    fingerprint = hashlib.sha256(
        json.dumps({"disease": disease, "data": patient_data}, sort_keys=True).encode()
    ).hexdigest()[:16]
    return _make_key("predict", PROBABILITY_SCALE_CONTRACT, disease, fingerprint)


def counterfactuals_cache_key(disease: str, patient_data: dict) -> str:
    """Deterministic cache key for a counterfactuals request, scoped to the scale contract."""
    fingerprint = hashlib.sha256(
        json.dumps({"disease": disease, "data": patient_data}, sort_keys=True).encode()
    ).hexdigest()[:16]
    return _make_key("counterfactuals", PROBABILITY_SCALE_CONTRACT, disease, fingerprint)


def cached_payload_matches_scale(payload: Any, disease_config: Optional[dict]) -> bool:
    """
    True when a cached /predict payload is on the scale the caller expects.

    A disease whose config declares both prevalence priors returns corrected
    probabilities, and its payload must say so via `prevalence_correction_applied`.
    A disease that declares neither (heart_disease) has one scale only, so any
    payload is acceptable.

    Returns False rather than raising: an unreadable payload is a cache miss,
    not an error.
    """
    model_cfg = (disease_config or {}).get("model", {}) or {}
    corrected_expected = {"prevalence_train", "prevalence_deploy"} <= set(model_cfg)
    if not corrected_expected:
        return True
    if not isinstance(payload, dict):
        return False
    return payload.get("prevalence_correction_applied") is True


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
