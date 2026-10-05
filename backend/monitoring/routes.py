"""
OmniDiag — Monitoring Routes (Features 4.1 + 4.6)
===================================================
Mounted at /api/v4/admin/drift/* in main.py.

Endpoints:
    GET  /admin/drift/{disease}/status  — latest drift metrics JSON
    GET  /admin/drift/{disease}/report  — full Evidently HTML report
    POST /admin/drift/{disease}/run     — trigger a fresh drift run
    GET  /metrics                       — Prometheus metrics (mounted at root)
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel

from sqlalchemy.ext.asyncio import AsyncSession

from backend.auth.rbac import require_role, ADMIN_ROLES
from backend.retired_diseases import reject_if_retired
from backend.database import get_db
from backend.monitoring.drift import get_monitor, drift_unavailable_reason
from backend.monitoring.metrics import get_metrics_response
from backend.monitoring.mlflow_tracker import log_model_info, tracking_state

router = APIRouter()


# ── Schemas ───────────────────────────────────────────────────────────────────

class DriftStatusResponse(BaseModel):
    disease: str
    monitor_ready: bool
    last_run: Optional[str]
    report: Optional[Dict[str, Any]]
    message: str
    # How the sampled rows split by predictions.probability_scale. Keys are
    # 'corrected', 'raw' and 'unknown' (NULL — written before the column
    # existed). Only rows on `confidence_scale` contribute a confidence value;
    # the rest contribute their input features only. None when not computed.
    rows_by_probability_scale: Optional[Dict[str, int]] = None
    confidence_scale: Optional[str] = None


# ── GET /admin/drift/{disease}/status ─────────────────────────────────────────

@router.get(
    "/drift/{disease}/status",
    response_model=DriftStatusResponse,
    summary="Latest drift metrics for a disease (super_admin only)",
)
async def drift_status(
    disease: str,
    _: object = Depends(require_role(*ADMIN_ROLES)),
) -> DriftStatusResponse:
    reject_if_retired(disease)  # a retired module has no live model to monitor
    monitor = get_monitor(disease)
    return DriftStatusResponse(
        disease=disease,
        monitor_ready=monitor.is_ready,
        last_run=monitor.last_run.isoformat() if monitor.last_run else None,
        report=monitor.last_report,
        # When the monitor cannot run at all, say why rather than implying a
        # report merely has not been triggered yet. On the Space evidently is
        # installed but exposes an incompatible API (see docs/EVIDENTLY_COST.md),
        # and "no report yet" hid that behind a message about a missing run.
        message=(
            drift_unavailable_reason(disease)
            or ("No drift report run yet. POST /admin/drift/{disease}/run to trigger."
                if monitor.last_report is None else "Drift report available.")
        ),
    )


# ── POST /admin/drift/{disease}/run ───────────────────────────────────────────

class DriftRunRequest(BaseModel):
    sample_size: int = Query(default=1000, ge=10, le=10000, description="Number of recent predictions to sample")


@router.post(
    "/drift/{disease}/run",
    response_model=DriftStatusResponse,
    summary="Trigger a drift computation run (super_admin only)",
)
async def run_drift(
    disease: str,
    body: DriftRunRequest = None,
    _: object = Depends(require_role(*ADMIN_ROLES)),
    db: AsyncSession = Depends(get_db),
) -> DriftStatusResponse:
    reject_if_retired(disease)
    # Previously imported `async_session_maker` from backend.database — a name
    # that does not exist there (the factory is AsyncSessionLocal), so every
    # call to this route raised ImportError. The session now comes from the
    # standard dependency, which also makes it overridable in tests.
    from backend.db_models.prediction import Prediction
    from sqlalchemy import select
    import json as _json

    monitor = get_monitor(disease)
    if not monitor.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"error": f"Drift monitor not ready for {disease}", "code": "MONITOR_NOT_READY"},
        )

    sample_size = (body.sample_size if body else 1000)

    rows = (
        await db.execute(
            select(Prediction)
            .where(Prediction.disease == disease)
            .order_by(Prediction.created_at.desc())
            .limit(sample_size)
        )
    ).scalars().all()

    if len(rows) < 10:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"error": f"Need at least 10 predictions for {disease}, found {len(rows)}", "code": "INSUFFICIENT_DATA"},
        )

    # Build DataFrame from stored input_features.
    #
    # Input features are scale-free, so every sampled row contributes them.
    # `confidence` is not: a window can hold raw-prior rows written before the
    # prevalence correction, corrected rows written after it, and NULL rows
    # whose scale was never recorded. Pooling them would register the
    # correction itself as drift. So confidence is kept only for rows on the
    # scale this module currently reports, and set to NaN — explicitly, not
    # by accident — for every other row, including NULL.
    from configs.config_loader import load_config
    from backend.probability_scale import scale_of_disease_config

    # The scale this disease reports is a property of its config (both
    # prevalence priors declared -> corrected). Read the YAML directly rather
    # than importing backend.main, which would re-run app start-up if the
    # server was launched under a different module name.
    try:
        disease_config = load_config(disease)
    except Exception:
        disease_config = None
    current_scale = scale_of_disease_config(disease_config).value
    scale_counts: Dict[str, int] = {"corrected": 0, "raw": 0, "unknown": 0}
    records = []
    for p in rows:
        row_scale = p.probability_scale if p.probability_scale is not None else "unknown"
        scale_counts[row_scale] = scale_counts.get(row_scale, 0) + 1
        row = dict(p.input_features or {})
        row["prediction"] = p.prediction
        row["confidence"] = p.confidence if row_scale == current_scale else float("nan")
        records.append(row)

    current_df = pd.DataFrame(records)

    # Run drift (sync, but fast on small samples)
    import asyncio
    loop = asyncio.get_event_loop()
    report = await loop.run_in_executor(None, monitor.run, current_df)

    # Update the Prometheus gauge.
    #
    # `or 0.0` used to stand here, which published a drift share of ZERO whenever
    # the run could not judge -- too few rows, an unencodable window, a missing
    # reference. Zero means "measured, nothing drifted". NaN is the honest export
    # for "not measured": Prometheus carries it and a dashboard shows a gap
    # instead of a reassuring flat line at zero.
    from backend.monitoring.metrics import record_drift
    share = report.get("metrics", {}).get("dataset_drift", {}).get("drift_share")
    record_drift(disease, float("nan") if share is None else float(share))

    return DriftStatusResponse(
        disease=disease,
        monitor_ready=True,
        last_run=monitor.last_run.isoformat() if monitor.last_run else None,
        report=report,
        message="Drift report computed successfully.",
        rows_by_probability_scale=scale_counts,
        confidence_scale=current_scale,
    )


# ── GET /admin/drift/{disease}/report ─────────────────────────────────────────

@router.get(
    "/drift/{disease}/report",
    response_class=HTMLResponse,
    summary="Full Evidently HTML drift report (super_admin only)",
)
async def drift_html_report(
    disease: str,
    _: object = Depends(require_role(*ADMIN_ROLES)),
) -> HTMLResponse:
    reject_if_retired(disease)
    monitor = get_monitor(disease)
    if monitor.last_report is None:
        return HTMLResponse(
            content=("<html><body><h2>No report yet. "
                     f"POST /admin/drift/{disease}/run first.</h2></body></html>"),
            status_code=200,
        )
    # Rendered from the live report. This is NOT Evidently's report -- that path
    # never ran (see backend/monitoring/drift.py) and asking the new monitor for
    # `_last_html` raised AttributeError.
    return HTMLResponse(content=_render_drift_html(monitor.last_report))


def _render_drift_html(report: Dict[str, Any]) -> str:
    """A plain table of the drift numbers. No new dependency, no template engine."""
    from html import escape

    def cell(value, digits=4):
        if value is None:
            return "<td class='na'>not measured</td>"
        if isinstance(value, float):
            return f"<td>{value:.{digits}g}</td>"
        return f"<td>{escape(str(value))}</td>"

    status = escape(str(report.get("status", "")))
    head = (f"<h2>Input drift — {escape(str(report.get('disease', '')))}</h2>"
            f"<p><b>status:</b> {status}"
            + (f" — {escape(str(report.get('detail')))}" if report.get("detail") else "")
            + f"<br><b>reference:</b> {report.get('reference_rows')} rows, sha256 "
              f"{escape(str(report.get('reference_sha256', ''))[:12])}…"
              f"<br><b>current:</b> {report.get('current_rows')} rows"
              f"<br><b>measures:</b> {escape(str(report.get('measures', '')))}</p>")
    rows = ""
    for f in report.get("features", []):
        rows += ("<tr>"
                 f"<td>{escape(str(f['feature']))}</td>"
                 f"<td>{escape(str(f['kind']))}</td>"
                 + cell(f.get("psi")) + f"<td>{escape(str(f.get('psi_band')))}</td>"
                 + cell(f.get("p_value"), 3) + cell(f.get("p_adjusted"), 3)
                 + f"<td>{'<b>yes</b>' if f.get('drifted') else 'no'}</td>"
                 + f"<td>{escape(str(f.get('detail') or ''))}</td>"
                 "</tr>")
    table = ("<table border=1 cellpadding=4 cellspacing=0><thead><tr>"
             "<th>feature</th><th>kind</th><th>PSI</th><th>band</th><th>p</th>"
             "<th>p (Holm)</th><th>drifted</th><th>note</th></tr></thead>"
             f"<tbody>{rows}</tbody></table>")
    rule = escape(str((report.get("rule") or {}).get("flag", "")))
    return ("<html><head><meta charset='utf-8'><title>Drift report</title>"
            "<style>body{font-family:system-ui,sans-serif;margin:2rem}"
            "td,th{font-size:14px}.na{color:#888;font-style:italic}</style></head>"
            f"<body>{head}{table}<p><b>flagged when:</b> {rule}</p></body></html>")


# ── GET /admin/mlflow/runs ────────────────────────────────────────────────────

@router.get(
    "/mlflow/runs",
    summary="List recent MLflow experiment runs (super_admin only)",
)
async def mlflow_runs(
    n: int = Query(20, ge=1, le=100),
    _: object = Depends(require_role(*ADMIN_ROLES)),
) -> Dict[str, Any]:
    # `status` distinguishes "not installed" / "store unreachable" / "working but
    # empty" / "working", which `count: 0` alone never did -- and that ambiguity
    # is why an empty experiment went unexplained (Gate 8.6).
    state = tracking_state(n=n)
    return {
        "experiment": "OmniDiag",
        "status": state["status"],
        "tracking_uri": state["tracking_uri"],
        "detail": state["detail"],
        "count": len(state["runs"]),
        "runs": state["runs"],
    }


# ── POST /admin/mlflow/register-model ─────────────────────────────────────────

class RegisterModelRequest(BaseModel):
    disease: str
    model_version: str
    metrics: Dict[str, float] = {}
    params: Dict[str, Any] = {}


@router.post(
    "/mlflow/register",
    summary="Register model metrics in MLflow (super_admin only)",
)
async def mlflow_register(
    body: RegisterModelRequest,
    _: object = Depends(require_role(*ADMIN_ROLES)),
) -> Dict[str, Any]:
    run_id = log_model_info(
        disease=body.disease,
        model_version=body.model_version,
        metrics=body.metrics,
        params=body.params,
    )
    if run_id is None:
        return {"message": "MLflow unavailable — skipped", "run_id": None}
    return {"message": "Model registered in MLflow", "run_id": run_id}


# ── GET /metrics (Prometheus) ─────────────────────────────────────────────────

@router.get(
    "/metrics",
    include_in_schema=False,
    summary="Prometheus metrics endpoint",
)
async def prometheus_metrics() -> Response:
    body, content_type = get_metrics_response()
    return Response(content=body, media_type=content_type)
