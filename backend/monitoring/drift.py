"""
OmniDiag — Model Drift Monitor (Feature 4.1)
==============================================
Uses Evidently to compute data drift and model performance reports against
a reference (training) dataset baseline.

Usage:
    from backend.monitoring.drift import DriftMonitor

    monitor = DriftMonitor(reference_csv="data/heart_disease/processed/final_ready_data.csv")
    report  = monitor.run(current_df)         # returns dict with drift metrics
    html    = monitor.run_html(current_df)    # returns full HTML report string

The DriftMonitor is mounted at:
    GET  /api/v4/admin/drift/status   — latest drift metrics as JSON
    GET  /api/v4/admin/drift/report   — full Evidently HTML report
    POST /api/v4/admin/drift/run      — trigger a fresh drift computation
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

log = logging.getLogger("omnidiag.drift")

# Lazy-import evidently so the module loads even if evidently is not installed
_evidently_available = False
try:
    from evidently import ColumnMapping
    from evidently.metrics import (
        DatasetDriftMetric,
        DatasetMissingValuesMetric,
        ColumnDriftMetric,
    )
    from evidently.report import Report
    _evidently_available = True
except ImportError:
    log.warning("evidently not installed — drift monitoring unavailable")


class DriftMonitor:
    """
    Wrapper around Evidently's Report for OmniDiag drift detection.

    Thread-safe: uses a lock around report runs so concurrent requests
    don't corrupt state.
    """

    def __init__(
        self,
        reference_csv: str | Path,
        target_col: str = "target",
        prediction_col: Optional[str] = None,
        categorical_features: Optional[List[str]] = None,
        numerical_features: Optional[List[str]] = None,
    ) -> None:
        self._lock = threading.Lock()
        self._last_report: Optional[Dict[str, Any]] = None
        self._last_run: Optional[datetime] = None
        self._last_html: Optional[str] = None

        ref_path = Path(reference_csv)
        if not ref_path.exists():
            log.warning("Reference CSV not found: %s — drift monitor inactive", ref_path)
            self._reference: Optional[pd.DataFrame] = None
        else:
            self._reference = pd.read_csv(ref_path)
            log.info("DriftMonitor: loaded reference with %d rows from %s", len(self._reference), ref_path)

        self._target_col = target_col
        self._prediction_col = prediction_col
        self._cat_features = categorical_features
        self._num_features = numerical_features

    @property
    def is_ready(self) -> bool:
        return _evidently_available and self._reference is not None

    def run(self, current_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Run Evidently drift report against the reference dataset.
        Returns a structured dict with drift summary.
        """
        if not self.is_ready:
            return {"error": "Drift monitoring unavailable (evidently not installed or reference missing)"}

        with self._lock:
            col_mapping = ColumnMapping(
                target=self._target_col if self._target_col in self._reference.columns else None,
                prediction=self._prediction_col,
                numerical_features=self._num_features,
                categorical_features=self._cat_features,
            )

            report = Report(metrics=[
                DatasetDriftMetric(),
                DatasetMissingValuesMetric(),
            ])

            # Align columns between reference and current
            common_cols = [c for c in self._reference.columns if c in current_df.columns]
            report.run(
                reference_data=self._reference[common_cols],
                current_data=current_df[common_cols],
                column_mapping=col_mapping,
            )

            result_dict = report.as_dict()
            metrics = result_dict.get("metrics", [])

            # Extract summary values
            drift_summary: Dict[str, Any] = {
                "run_at": datetime.now(timezone.utc).isoformat(),
                "reference_rows": len(self._reference),
                "current_rows": len(current_df),
                "common_columns": len(common_cols),
                "metrics": {},
            }

            for m in metrics:
                mtype = m.get("metric", "")
                mresult = m.get("result", {})
                if "DatasetDriftMetric" in mtype:
                    drift_summary["metrics"]["dataset_drift"] = {
                        "dataset_drift_detected": mresult.get("dataset_drift"),
                        "drift_share": mresult.get("drift_share"),
                        "number_of_drifted_columns": mresult.get("number_of_drifted_columns"),
                        "number_of_columns": mresult.get("number_of_columns"),
                    }
                elif "DatasetMissingValuesMetric" in mtype:
                    drift_summary["metrics"]["missing_values"] = {
                        "current_missing_share": mresult.get("current", {}).get("share_of_missing_values"),
                        "reference_missing_share": mresult.get("reference", {}).get("share_of_missing_values"),
                    }

            self._last_report = drift_summary
            self._last_run = datetime.now(timezone.utc)

            # Also save HTML
            self._last_html = report.get_html()

            log.info(
                "Drift run complete: drift_detected=%s share=%.2f%%",
                drift_summary["metrics"].get("dataset_drift", {}).get("dataset_drift_detected"),
                (drift_summary["metrics"].get("dataset_drift", {}).get("drift_share") or 0) * 100,
            )

            return drift_summary

    def run_html(self, current_df: pd.DataFrame) -> str:
        """Run drift report and return the full Evidently HTML string."""
        self.run(current_df)
        return self._last_html or "<p>No report available.</p>"

    @property
    def last_report(self) -> Optional[Dict[str, Any]]:
        return self._last_report

    @property
    def last_run(self) -> Optional[datetime]:
        return self._last_run


# ── Per-disease monitor singletons ────────────────────────────────────────────

_monitors: Dict[str, DriftMonitor] = {}

_REFERENCE_PATHS: Dict[str, str] = {
    "heart_disease": "data/heart_disease/processed/final_ready_data.csv",
    "diabetes":      "data/diabetes/interim/false_negatives_profile.csv",
}


def get_monitor(disease: str) -> DriftMonitor:
    if disease not in _monitors:
        ref = _REFERENCE_PATHS.get(disease, f"data/{disease}/processed/reference.csv")
        _monitors[disease] = DriftMonitor(reference_csv=ref)
    return _monitors[disease]
