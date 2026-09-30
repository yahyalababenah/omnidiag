"""
The Grafana dashboard charts request latency and prediction-cache hits, but
REQUEST_DURATION, ACTIVE_REQUESTS, CACHE_HITS and CACHE_MISSES were registered
and never written to, so those panels were permanently empty. These tests pin
that something now writes to them.
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.monitoring.metrics import (
    MetricsMiddleware, get_metrics_response, record_cache,
)


def _value(prefix: str) -> float:
    body, _ = get_metrics_response()
    for line in body.decode().splitlines():
        if line.startswith(prefix):
            return float(line.rsplit(" ", 1)[1])
    return 0.0


def _app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(MetricsMiddleware)

    @app.get("/items/{item_id}")
    def item(item_id: int):
        return {"id": item_id}

    return app


def test_request_is_timed_under_the_route_template_not_the_raw_path():
    client = TestClient(_app())
    prefix = ('omnidiag_request_duration_seconds_count{endpoint="/items/{item_id}",'
              'method="GET",status="200"} ')
    before = _value(prefix)
    client.get("/items/1")
    client.get("/items/2")
    assert _value(prefix) == before + 2
    body, _ = get_metrics_response()
    assert 'endpoint="/items/1"' not in body.decode()


def test_unmatched_route_gets_a_bounded_label():
    client = TestClient(_app())
    client.get("/nowhere/123")
    body, _ = get_metrics_response()
    assert 'endpoint="unmatched"' in body.decode()
    assert 'endpoint="/nowhere/123"' not in body.decode()


def test_no_request_is_left_in_flight_afterwards():
    TestClient(_app()).get("/items/1")
    assert _value("omnidiag_active_requests ") == 0.0


def test_cache_lookups_are_counted_as_hits_and_misses():
    hits = 'omnidiag_cache_hits_total{disease="heart_disease"} '
    misses = 'omnidiag_cache_misses_total{disease="heart_disease"} '
    h0, m0 = _value(hits), _value(misses)
    record_cache("heart_disease", True)
    record_cache("heart_disease", True)
    record_cache("heart_disease", False)
    assert _value(hits) == h0 + 2
    assert _value(misses) == m0 + 1
