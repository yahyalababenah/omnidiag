"""
OmniDiag — Active Learning / Review Queue API
==============================================
Mounted at /api/v4/review in main.py.

Endpoints:
    GET  /api/v4/review/queue          — List pending review items (doctors)
    POST /api/v4/review/{id}/annotate  — Submit expert label
    POST /api/v4/review/{id}/skip      — Skip/dismiss a review item
    GET  /api/v4/review/stats          — Queue statistics
"""

import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel
from sqlalchemy import and_, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from backend.auth.rbac import CLINICAL_ROLES, ADMIN_ROLES, require_role
from backend.database import get_db
from backend.db_models.prediction import Prediction
from backend.db_models.review_queue import ReviewQueue
from backend.db_models.user import User

router = APIRouter()


class AnnotateRequest(BaseModel):
    label: int  # 0 or 1 — the expert's ground-truth annotation
    notes: Optional[str] = None


@router.get(
    "/queue",
    summary="List pending predictions awaiting expert review",
    tags=["Active Learning"],
)
async def list_review_queue(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    disease: Optional[str] = Query(None),
    current_user: User = Depends(require_role(*CLINICAL_ROLES)),
    db: AsyncSession = Depends(get_db),
):
    filters = [ReviewQueue.status == "pending"]
    if disease:
        # Join Prediction to filter by disease
        q_count = (
            select(func.count())
            .select_from(ReviewQueue)
            .join(Prediction, ReviewQueue.prediction_id == Prediction.id)
            .where(and_(*filters, Prediction.disease == disease))
        )
        q_rows = (
            select(ReviewQueue)
            .join(Prediction, ReviewQueue.prediction_id == Prediction.id)
            .where(and_(*filters, Prediction.disease == disease))
            .options(selectinload(ReviewQueue.prediction))
            .order_by(ReviewQueue.created_at.desc())
            .offset((page - 1) * limit)
            .limit(limit)
        )
    else:
        q_count = (
            select(func.count())
            .select_from(ReviewQueue)
            .where(and_(*filters))
        )
        q_rows = (
            select(ReviewQueue)
            .where(and_(*filters))
            .options(selectinload(ReviewQueue.prediction))
            .order_by(ReviewQueue.created_at.desc())
            .offset((page - 1) * limit)
            .limit(limit)
        )

    total = (await db.execute(q_count)).scalar_one()
    rows = (await db.execute(q_rows)).scalars().all()

    items = []
    for rq in rows:
        pred = rq.prediction
        items.append({
            "id": rq.id,
            "prediction_id": rq.prediction_id,
            "uncertainty_score": rq.uncertainty_score,
            "status": rq.status,
            "created_at": rq.created_at.isoformat() if rq.created_at else None,
            "disease": pred.disease if pred else None,
            "model_prediction": pred.prediction if pred else None,
            "confidence": pred.confidence if pred else None,
            "features": pred.input_features if pred else None,
        })

    return {
        "total": total,
        "page": page,
        "limit": limit,
        "pages": max(1, -(-total // limit)),
        "items": items,
    }


@router.post(
    "/{review_id}/annotate",
    summary="Submit expert annotation for a review item",
    tags=["Active Learning"],
    status_code=200,
)
async def annotate_review(
    review_id: str,
    body: AnnotateRequest,
    current_user: User = Depends(require_role(*CLINICAL_ROLES)),
    db: AsyncSession = Depends(get_db),
):
    rq = (await db.execute(select(ReviewQueue).where(ReviewQueue.id == review_id))).scalar_one_or_none()
    if rq is None:
        raise HTTPException(status_code=404, detail="Review item not found")
    if rq.status != "pending":
        raise HTTPException(status_code=409, detail=f"Item already {rq.status}")
    if body.label not in (0, 1):
        raise HTTPException(status_code=422, detail="label must be 0 or 1")

    rq.label = body.label
    rq.reviewer_id = current_user.id
    rq.reviewed_at = datetime.now(timezone.utc)
    rq.status = "reviewed"
    await db.commit()
    return {"id": review_id, "status": "reviewed", "label": body.label}


@router.post(
    "/{review_id}/skip",
    summary="Skip / dismiss a review item",
    tags=["Active Learning"],
    status_code=200,
)
async def skip_review(
    review_id: str,
    current_user: User = Depends(require_role(*CLINICAL_ROLES)),
    db: AsyncSession = Depends(get_db),
):
    rq = (await db.execute(select(ReviewQueue).where(ReviewQueue.id == review_id))).scalar_one_or_none()
    if rq is None:
        raise HTTPException(status_code=404, detail="Review item not found")
    if rq.status != "pending":
        raise HTTPException(status_code=409, detail=f"Item already {rq.status}")
    rq.status = "skipped"
    rq.reviewer_id = current_user.id
    rq.reviewed_at = datetime.now(timezone.utc)
    await db.commit()
    return {"id": review_id, "status": "skipped"}


@router.get(
    "/stats",
    summary="Review queue statistics",
    tags=["Active Learning"],
)
async def review_stats(
    _user: User = Depends(require_role(*CLINICAL_ROLES)),
    db: AsyncSession = Depends(get_db),
):
    rows = (await db.execute(
        select(ReviewQueue.status, func.count().label("count"))
        .group_by(ReviewQueue.status)
    )).all()
    stats = {r.status: r.count for r in rows}
    return {
        "pending": stats.get("pending", 0),
        "reviewed": stats.get("reviewed", 0),
        "skipped": stats.get("skipped", 0),
        "total": sum(stats.values()),
    }
