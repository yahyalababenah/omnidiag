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

from backend.auth.rbac import require_role, ADMIN_ROLES
from backend.monitoring.drift import get_monitor
from backend.monitoring.metrics import get_metrics_response
from backend.monitoring.mlflow_tracker import list_recent_runs, log_model_info

router = APIRouter()


# ── Schemas ───────────────────────────────────────────────────────────────────

class DriftStatusResponse(BaseModel):
    disease: str
    monitor_ready: bool
    last_run: Optional[str]
    report: Optional[Dict[str, Any]]
    message: str


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
    monitor = get_monitor(disease)
    return DriftStatusResponse(
        disease=disease,
        monitor_ready=monitor.is_ready,
        last_run=monitor.last_run.isoformat() if monitor.last_run else None,
        report=monitor.last_report,
        message="No drift report run yet. POST /admin/drift/{disease}/run to trigger." if monitor.last_report is None else "Drift report available.",
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
) -> DriftStatusResponse:
    from backend.database import async_session_maker
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

    async with async_session_maker() as db:
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

    # Build DataFrame from stored input_features
    records = []
    for p in rows:
        row = dict(p.input_features or {})
        row["prediction"] = p.prediction
        row["confidence"] = p.confidence
        records.append(row)

    current_df = pd.DataFrame(records)

    # Run drift (sync, but fast on small samples)
    import asyncio
    loop = asyncio.get_event_loop()
    report = await loop.run_in_executor(None, monitor.run, current_df)

    # Update Prometheus gauge
    from backend.monitoring.metrics import record_drift
    drift_share = report.get("metrics", {}).get("dataset_drift", {}).get("drift_share") or 0.0
    record_drift(disease, drift_share)

    return DriftStatusResponse(
        disease=disease,
        monitor_ready=True,
        last_run=monitor.last_run.isoformat() if monitor.last_run else None,
        report=report,
        message="Drift report computed successfully.",
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
    monitor = get_monitor(disease)
    if monitor.last_report is None:
        return HTMLResponse(
            content="<html><body><h2>No report yet. POST /admin/drift/{disease}/run first.</h2></body></html>",
            status_code=200,
        )
    html = monitor._last_html or "<p>Report HTML not available.</p>"
    return HTMLResponse(content=html)


# ── GET /admin/mlflow/runs ────────────────────────────────────────────────────

@router.get(
    "/mlflow/runs",
    summary="List recent MLflow experiment runs (super_admin only)",
)
async def mlflow_runs(
    n: int = Query(20, ge=1, le=100),
    _: object = Depends(require_role(*ADMIN_ROLES)),
) -> Dict[str, Any]:
    runs = list_recent_runs(n=n)
    return {"experiment": "OmniDiag", "count": len(runs), "runs": runs}


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
