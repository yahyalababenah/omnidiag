"""
OmniDiag — Prometheus Metrics (Feature 4.6)
============================================
Exposes /metrics endpoint for Prometheus scraping.

Counters and histograms are updated:
  - In predict/explain endpoints (via middleware hook)
  - In the batch endpoint

Metrics:
  omnidiag_predictions_total{disease, prediction}  — counter
  omnidiag_prediction_confidence{}                 — histogram
  omnidiag_request_duration_seconds{endpoint}      — histogram
  omnidiag_active_requests{}                       — gauge
  omnidiag_drift_share{disease}                    — gauge (updated on drift run)
"""

from __future__ import annotations

import logging

log = logging.getLogger("omnidiag.metrics")

_prometheus_available = False
try:
    from prometheus_client import (
        Counter, Histogram, Gauge, CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST,
        REGISTRY,
    )
    _prometheus_available = True
except ImportError:
    log.warning("prometheus_client not installed — /metrics endpoint inactive")


if _prometheus_available:
    # ── Prediction counters ──────────────────────────────────────────────────
    PREDICTIONS_TOTAL = Counter(
        "omnidiag_predictions_total",
        "Total number of predictions served",
        ["disease", "prediction"],
    )

    PREDICTION_CONFIDENCE = Histogram(
        "omnidiag_prediction_confidence",
        "Model confidence score distribution",
        ["disease"],
        buckets=[0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0],
    )

    REQUEST_DURATION = Histogram(
        "omnidiag_request_duration_seconds",
        "HTTP request latency in seconds",
        ["method", "endpoint", "status"],
        buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
    )

    ACTIVE_REQUESTS = Gauge(
        "omnidiag_active_requests",
        "Number of requests currently being processed",
    )

    DRIFT_SHARE = Gauge(
        "omnidiag_drift_share",
        "Fraction of features with detected drift (from last Evidently run)",
        ["disease"],
    )

    CACHE_HITS = Counter(
        "omnidiag_cache_hits_total",
        "Number of prediction cache hits",
        ["disease"],
    )

    CACHE_MISSES = Counter(
        "omnidiag_cache_misses_total",
        "Number of prediction cache misses",
        ["disease"],
    )

    BATCH_ROWS_PROCESSED = Counter(
        "omnidiag_batch_rows_total",
        "Total rows processed in batch predictions",
        ["disease", "status"],
    )


def record_prediction(disease: str, prediction: int, confidence: float) -> None:
    if not _prometheus_available:
        return
    PREDICTIONS_TOTAL.labels(disease=disease, prediction=str(prediction)).inc()
    PREDICTION_CONFIDENCE.labels(disease=disease).observe(confidence)


def record_batch(disease: str, succeeded: int, failed: int) -> None:
    if not _prometheus_available:
        return
    BATCH_ROWS_PROCESSED.labels(disease=disease, status="ok").inc(succeeded)
    BATCH_ROWS_PROCESSED.labels(disease=disease, status="error").inc(failed)


def record_drift(disease: str, drift_share: float) -> None:
    if not _prometheus_available:
        return
    DRIFT_SHARE.labels(disease=disease).set(drift_share)


def get_metrics_response():
    """Return (body_bytes, content_type) for /metrics endpoint."""
    if not _prometheus_available:
        return b"# prometheus_client not installed\n", "text/plain"
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST
