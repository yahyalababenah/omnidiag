"""
OmniDiag — Model Drift Monitor (Feature 4.1)
==============================================
THE LIVE PATH IS `ProfileDriftMonitor` (Gate 8.7c): KS, chi-square and PSI on
scipy, against a frozen reference profile. `get_monitor()` returns it.

The Evidently `DriftMonitor` below is kept as the record of what was there and
why it never ran. It could not run for TWO independent reasons, and fixing
either alone would have changed nothing:

  1. evidently 0.4's API is gone -- requirements asks for >=0.4.0, which resolves
     to 0.7, and 0.7 removed both ColumnMapping and evidently.report.Report
     (docs/EVIDENTLY_COST.md);
  2. the heart reference CSV it pointed at,
     data/heart_disease/processed/final_ready_data.csv, DOES NOT EXIST in this
     repository, so `self._reference` was None and `is_ready` was False anyway.

The old reference for diabetes was the 50/50 BALANCED BRFSS sample, which would
have reported drift on every feature correlated with diabetes: the reference was
balanced by construction and real people are not.

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

import numpy as np
import pandas as pd

log = logging.getLogger("omnidiag.drift")

# Lazy-import evidently so the module loads either way.
#
# These are the evidently 0.4 imports. requirements.txt asks for
# `evidently>=0.4.0`, which resolves to 0.7.x, and 0.7 removed both
# ColumnMapping and evidently.report.Report. So on the Space evidently IS
# installed and this still raises ImportError.
#
# The old message here read "evidently not installed", which sent anyone
# reading the log looking for a missing dependency that was in fact present.
# Report the real reason. See docs/EVIDENTLY_COST.md for the options and
# why neither is worth taking before the demo.
_evidently_available = False
_evidently_unavailable_reason: Optional[str] = None
try:
    from evidently import ColumnMapping
    from evidently.metrics import (
        DatasetDriftMetric,
        DatasetMissingValuesMetric,
        ColumnDriftMetric,
    )
    from evidently.report import Report
    _evidently_available = True
except ImportError as _exc:
    try:
        from importlib.metadata import version as _pkg_version
        _installed = _pkg_version("evidently")
    except Exception:
        _installed = None
    if _installed:
        _evidently_unavailable_reason = (
            f"evidently {_installed} is installed but its API is incompatible — "
            f"backend/monitoring/drift.py targets the 0.4 API ({_exc}). "
            f"Drift monitoring is unavailable; see docs/EVIDENTLY_COST.md"
        )
    else:
        _evidently_unavailable_reason = (
            f"evidently is not installed — drift monitoring unavailable ({_exc})"
        )
    log.warning(_evidently_unavailable_reason)


def evidently_unavailable_reason() -> Optional[str]:
    """Why the Evidently path cannot run, or None when it can. Kept for the record."""
    return None if _evidently_available else _evidently_unavailable_reason


def drift_unavailable_reason(disease: str = "heart_disease") -> Optional[str]:
    """Why the LIVE drift monitor cannot run for this disease, or None when it can.

    This used to report Evidently's state, which was the wrong question once the
    live path stopped being Evidently -- and it hid that the reference file was
    missing too.
    """
    return get_monitor(disease).unavailable_reason()


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
    "diabetes":      "data/diabetes/raw/diabetes_binary_5050split_health_indicators_BRFSS2015.csv",
}


def get_evidently_monitor(disease: str) -> DriftMonitor:
    """The Evidently monitor. Kept for the record; not the live path."""
    if disease not in _monitors:
        ref = _REFERENCE_PATHS.get(disease, f"data/{disease}/processed/reference.csv")
        _monitors[disease] = DriftMonitor(reference_csv=ref)
    return _monitors[disease]


# ── The live path: a frozen profile plus scipy ─────────────────────────────────

_PROFILE_PATHS: Dict[str, str] = {
    "heart_disease": "models/heart_disease/drift_reference.json",
    "diabetes": "models/diabetes/drift_reference.json",
}


class ProfileDriftMonitor:
    """Input drift against a frozen reference profile, on scipy alone.

    The profile is built by scripts/build_drift_reference.py and verified at image
    build time. Nothing here reads a CSV: the bin edges belong to the reference
    and must not move with the incoming sample.

    What this reports is INPUT drift -- the arriving population has stopped
    looking like the fitted one. That is not model degradation, which would need
    follow-up labels this system does not have. The status says so.
    """

    def __init__(self, disease: str, profile_path: str | Path) -> None:
        self._lock = threading.Lock()
        self.disease = disease
        self._path = Path(profile_path)
        self._profile: Optional[Dict[str, Any]] = None
        self._reason: Optional[str] = None
        self._last_report: Optional[Dict[str, Any]] = None
        self._last_run: Optional[datetime] = None
        if not self._path.exists():
            self._reason = (f"drift reference profile missing: {self._path} — build it with "
                            f"`python scripts/build_drift_reference.py`")
            log.warning(self._reason)
        else:
            try:
                self._profile = json.loads(self._path.read_text())
            except Exception as exc:  # noqa: BLE001
                self._reason = f"drift reference profile unreadable ({type(exc).__name__}: {exc})"
                log.warning(self._reason)

    @property
    def is_ready(self) -> bool:
        return self._profile is not None

    def unavailable_reason(self) -> Optional[str]:
        return None if self.is_ready else self._reason

    def _to_monitored_space(self, current: "pd.DataFrame") -> "pd.DataFrame":
        """Put incoming rows in the same space as the reference.

        For heart that means encoding them, because the chest-pain coding is
        INVERTED between training and the API (ASY/NAP/ATA/TA -> 3/2/1/0 versus
        TA/ATA/NAP/ASY -> 3/2/1/0). Comparing the raw strings would compare
        opposite meanings and report drift on a population that had not moved.
        """
        if self.disease != "heart_disease":
            return current
        from backend.heart_glm.stack import encode_for_inference
        needed = ["Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol",
                  "FastingBS", "RestingECG"]
        missing = [c for c in needed if c not in current.columns]
        if missing:
            raise ValueError(f"cannot encode incoming rows — missing input columns: {missing}")
        return encode_for_inference(current[needed])

    def run(self, current_df: "pd.DataFrame") -> Dict[str, Any]:
        from backend.monitoring.drift_stats import (
            MIN_CURRENT_ROWS, compare_feature, flag_features,
        )
        if not self.is_ready:
            return {"status": "reference_missing", "detail": self._reason,
                    "metrics": {"dataset_drift": {"drift_share": None}}}

        profile = self._profile
        rule = profile.get("rule", {})
        min_rows = int(rule.get("min_current_rows", MIN_CURRENT_ROWS))

        with self._lock:
            n_current = int(len(current_df))
            base: Dict[str, Any] = {
                "run_at": datetime.now(timezone.utc).isoformat(),
                "disease": self.disease,
                "reference_rows": profile["source"]["rows"],
                "reference_sha256": profile["source"]["sha256"],
                "reference_reweighting": profile.get("reweighting"),
                "space": profile.get("space"),
                "current_rows": n_current,
                "rule": rule,
                "measures": "input drift only — not model degradation, which would need "
                            "follow-up labels that do not exist for any patient here",
            }
            # Under the floor nothing is judged. A drift share of 0.0 on 12 rows
            # reads as "measured, no drift" -- the same lie `count: 0` told about
            # MLflow before Gate 8.6.
            if n_current < min_rows:
                report = {**base, "status": "insufficient_data",
                          "detail": f"{n_current} current rows, {min_rows} needed to judge",
                          "metrics": {"dataset_drift": {"drift_share": None,
                                                        "number_of_columns": len(profile["monitored"])}}}
                self._last_report, self._last_run = report, datetime.now(timezone.utc)
                return report

            try:
                space = self._to_monitored_space(current_df)
            except Exception as exc:  # noqa: BLE001
                report = {**base, "status": "cannot_encode", "detail": str(exc),
                          "metrics": {"dataset_drift": {"drift_share": None}}}
                self._last_report, self._last_run = report, datetime.now(timezone.utc)
                return report

            from backend.monitoring.drift_stats import expand_ecdf
            comparisons, skipped = [], {}
            for name, ref in profile["monitored"].items():
                if name not in space.columns:
                    skipped[name] = "absent from the incoming rows"
                    continue
                if ref["kind"] != "numeric":
                    comparisons.append(_compare_categorical(name, ref, space[name].tolist()))
                    continue
                # KS gets the exact reference sample, rebuilt from its run-length
                # encoding; PSI gets the frozen bins. Passing bins alone here left
                # every numeric p-value None -- found by running the seeded test.
                comparisons.append(compare_feature(
                    name, "numeric",
                    reference=expand_ecdf(ref["ecdf"]["values"], ref["ecdf"]["counts"]),
                    current=space[name].tolist(),
                    edges=[(-np.inf if i == 0 else np.inf) if e is None else e
                           for i, e in enumerate(ref["edges"])],
                    reference_counts=ref["counts"],
                ))

            summary = flag_features(comparisons,
                                    alpha=float(rule.get("alpha", 0.05)),
                                    psi_major=float(rule.get("psi_major", 0.2)))
            report = {
                **base,
                "status": "ok",
                "detail": "",
                "features": summary["features"],
                "skipped_features": skipped,
                "drifted_features": summary["drifted_features"],
                "not_monitored": list(profile.get("profiled_but_not_monitored", {})),
                "not_monitored_reason": profile.get("not_monitored_reason", ""),
                "metrics": {"dataset_drift": {
                    "dataset_drift_detected": summary["n_drifted"] > 0,
                    "drift_share": summary["drift_share"],
                    "number_of_drifted_columns": summary["n_drifted"],
                    "number_of_columns": summary["n_monitored"],
                }},
            }
            self._last_report, self._last_run = report, datetime.now(timezone.utc)
            log.info("drift run %s: status=%s share=%s drifted=%s",
                     self.disease, report["status"],
                     report["metrics"]["dataset_drift"]["drift_share"],
                     report.get("drifted_features"))
            return report

    @property
    def last_report(self) -> Optional[Dict[str, Any]]:
        return self._last_report

    @property
    def last_run(self) -> Optional[datetime]:
        return self._last_run


def _compare_categorical(name: str, ref: Dict[str, Any], current: list) -> Dict[str, Any]:
    """Categorical comparison against STORED reference counts.

    The reference is counts, not rows -- for diabetes they are reweighted and
    there are no rows to hand back. So the contingency table is built here from
    the stored counts and the incoming values, on the union of both category sets.
    """
    from scipy import stats as _stats
    from backend.monitoring.drift_stats import PSI_MAJOR, PSI_MINOR, psi

    cats = list(ref["categories"])
    ref_counts = list(ref["counts"])
    cur: Dict[str, int] = {}
    for v in current:
        k = "__missing__" if v is None or (isinstance(v, float) and not np.isfinite(v)) else str(v)
        cur[k] = cur.get(k, 0) + 1
    # Categories the reference never saw are kept: they are the drift.
    extra = [k for k in sorted(cur) if k not in cats]
    all_cats = cats + extra
    r = np.array(ref_counts + [0] * len(extra), dtype=float)
    c = np.array([cur.get(k, 0) for k in all_cats], dtype=float)

    keep = (r + c) > 0
    table = np.vstack([r[keep], c[keep]])
    out: Dict[str, Any] = {"feature": name, "kind": "categorical",
                           "test": "chi2_contingency", "categories": [k for k, m in
                                                                      zip(all_cats, keep) if m],
                           "reference_counts": r[keep].astype(int).tolist(),
                           "current_counts": c[keep].astype(int).tolist()}
    if table.shape[1] < 2 or c.sum() < 1:
        out.update({"statistic": None, "p_value": None,
                    "detail": "fewer than two non-empty categories"})
    else:
        chi2, p, dof, expected = _stats.chi2_contingency(table)
        low = int(np.sum(expected < 5))
        out.update({"statistic": float(chi2), "p_value": float(p), "dof": int(dof),
                    "low_expected_cells": low,
                    "detail": (f"{low} expected cell(s) below 5 — the chi-square approximation "
                               f"is unreliable here; PSI carries the judgement") if low else ""})
    try:
        value = psi(out["reference_counts"], out["current_counts"])
    except ValueError as exc:
        out.update({"psi": None, "psi_detail": str(exc), "psi_band": None})
    else:
        out.update({"psi": value, "psi_detail": "",
                    "psi_band": ("major" if value >= PSI_MAJOR else
                                 "minor" if value >= PSI_MINOR else "none")})
    return out


_profile_monitors: Dict[str, ProfileDriftMonitor] = {}


def get_monitor(disease: str) -> ProfileDriftMonitor:
    """The live drift monitor for a disease: profile + scipy."""
    if disease not in _profile_monitors:
        _profile_monitors[disease] = ProfileDriftMonitor(
            disease, _PROFILE_PATHS.get(disease, f"models/{disease}/drift_reference.json"))
    return _profile_monitors[disease]
