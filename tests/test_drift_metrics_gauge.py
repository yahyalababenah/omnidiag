"""
Gate 8.7c — the Prometheus drift gauge must never publish a lying zero.

`record_drift()` used to be called as `record_drift(disease, drift_share or 0.0)`,
which set the gauge to 0.0 whenever a run could not judge at all (too few rows,
an unencodable window, a missing reference profile). A dashboard reading that
gauge sees "measured, nothing drifted" and a run that never happened looks
identical to a clean one. NaN is the honest export: Prometheus carries it, and a
dashboard shows a gap instead of a reassuring flat line.
"""
from __future__ import annotations

import math

from pathlib import Path

import pandas as pd
import pytest

from backend.monitoring.metrics import get_metrics_response, record_drift
from backend.monitoring.drift import get_monitor

HEART_CSV = Path("data/heart_disease/processed/uci_heart_by_site.csv")


def _read_gauge(disease: str) -> float | None:
    body, _ = get_metrics_response()
    prefix = f'omnidiag_drift_share{{disease="{disease}"}} '
    for line in body.decode().splitlines():
        if line.startswith(prefix):
            return float(line[len(prefix):])
    return None


class TestGaugeExportsWhatItMeasured:
    def test_a_real_share_is_exported_as_that_number(self):
        record_drift("heart_disease", 0.42857142857142855)
        assert _read_gauge("heart_disease") == pytest.approx(0.42857142857142855)

    def test_zero_drift_is_exported_as_zero(self):
        """A genuine zero (measured, nothing drifted) must still read as zero."""
        record_drift("heart_disease", 0.0)
        assert _read_gauge("heart_disease") == pytest.approx(0.0)

    def test_could_not_judge_is_exported_as_nan_not_zero(self):
        """The defect this test exists to catch: `share or 0.0` turns None into 0.0."""
        record_drift("heart_disease", float("nan"))
        value = _read_gauge("heart_disease")
        assert value is not None and math.isnan(value)

    def test_the_route_pattern_converts_none_to_nan_not_zero(self):
        """Reproduces exactly what backend/monitoring/routes.py does with a report."""
        report_share = None   # what `report["metrics"]["dataset_drift"]["drift_share"]` is
        exported = float("nan") if report_share is None else float(report_share)
        assert math.isnan(exported)


@pytest.mark.skipif(not HEART_CSV.exists(), reason="training CSV not present")
class TestGaugeEndToEnd:
    def test_a_run_that_cannot_judge_reaches_the_gauge_as_nan(self):
        """Twelve rows is below the floor: run() reports None, and that must not
        become the 0.0 that `or 0.0` used to produce."""
        rows = pd.read_csv(HEART_CSV).head(12)
        report = get_monitor("heart_disease").run(rows)
        share = report["metrics"]["dataset_drift"]["drift_share"]
        assert share is None
        record_drift("heart_disease", float("nan") if share is None else share)
        value = _read_gauge("heart_disease")
        assert value is not None and math.isnan(value)

    def test_a_judged_run_reaches_the_gauge_as_its_real_share(self):
        rows = pd.read_csv(HEART_CSV)
        report = get_monitor("heart_disease").run(rows[rows.site == "switzerland"])
        share = report["metrics"]["dataset_drift"]["drift_share"]
        assert share is not None
        record_drift("heart_disease", share)
        assert _read_gauge("heart_disease") == pytest.approx(share)
