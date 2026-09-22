"""
Booth-deployment configuration: cache lifetimes (What-If warm-up) and log
level (X-8).

The What-If cache TTL was 1 h. Generating a diabetes counterfactual costs
9-14 s, so a demo that warmed the cache in the morning was cold again by
mid-day and the first judge after that waited out the full generation. The
TTL now defaults to 12 h and is settable per deployment, so one run of
scripts/warmup_demo_cache.py covers a judging day.
"""

import importlib

import backend.cache as cache_module

# One judging day. Anything shorter goes cold between warm-up and judging.
BOOTH_DAY_SECONDS = 12 * 3600


class TestDefaultTtls:
    def test_counterfactuals_survive_a_full_judging_day(self):
        assert cache_module.COUNTERFACTUALS_TTL_SECONDS >= BOOTH_DAY_SECONDS, (
            f"What-If results expire after "
            f"{cache_module.COUNTERFACTUALS_TTL_SECONDS}s — a warm-up run will "
            f"not last the day"
        )

    def test_predict_and_schema_have_sane_defaults(self):
        assert cache_module.PREDICT_TTL_SECONDS == 3_600
        assert cache_module.SCHEMA_TTL_SECONDS == 86_400


class TestEnvironmentOverrides:
    def _reload_with(self, monkeypatch, **env):
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        return importlib.reload(cache_module)

    def test_each_ttl_is_settable_per_deployment(self, monkeypatch):
        reloaded = self._reload_with(
            monkeypatch,
            CACHE_TTL_COUNTERFACTUALS="7200",
            CACHE_TTL_PREDICT="60",
            CACHE_TTL_SCHEMA="120",
        )
        try:
            assert reloaded.COUNTERFACTUALS_TTL_SECONDS == 7_200
            assert reloaded.PREDICT_TTL_SECONDS == 60
            assert reloaded.SCHEMA_TTL_SECONDS == 120
        finally:
            for key in ("CACHE_TTL_COUNTERFACTUALS", "CACHE_TTL_PREDICT", "CACHE_TTL_SCHEMA"):
                monkeypatch.delenv(key)
            importlib.reload(cache_module)

    def test_a_typo_falls_back_instead_of_crashing_the_api(self, monkeypatch):
        """A bad Space secret must not stop the server from starting."""
        reloaded = self._reload_with(monkeypatch, CACHE_TTL_COUNTERFACTUALS="twelve hours")
        try:
            assert reloaded.COUNTERFACTUALS_TTL_SECONDS == 43_200
        finally:
            monkeypatch.delenv("CACHE_TTL_COUNTERFACTUALS")
            importlib.reload(cache_module)

    def test_a_negative_ttl_falls_back(self, monkeypatch):
        reloaded = self._reload_with(monkeypatch, CACHE_TTL_PREDICT="-1")
        try:
            assert reloaded.PREDICT_TTL_SECONDS == 3_600
        finally:
            monkeypatch.delenv("CACHE_TTL_PREDICT")
            importlib.reload(cache_module)


class TestLogLevel:
    def test_production_default_is_info_not_debug(self):
        """
        X-8: the Space logged 10,106 DEBUG lines out of 10,584, SQL among
        them. The source is read rather than the level inspected, because
        basicConfig() has already run by the time any test imports main.
        """
        source = open("backend/main.py").read()
        assert 'os.getenv("LOG_LEVEL", "INFO")' in source, (
            "log level is not env-driven with an INFO default (X-8)"
        )
        assert "level=logging.DEBUG" not in source, (
            "logging is still hard-wired to DEBUG (X-8)"
        )
