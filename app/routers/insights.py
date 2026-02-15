"""
Insights Router
───────────────
GET  /insights/{survey_id}        Get generated insights for a closed survey
POST /insights/{survey_id}/trigger  Manually trigger insight generation
"""
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import Survey, SurveyInsight
from app.routers.auth import require_admin
from app.schemas import InsightResult
from app.tasks.celery_app import generate_survey_insights

router = APIRouter(prefix="/insights", tags=["insights"])


@router.get("/{survey_id}")
async def get_insights(
    survey_id: UUID,
    admin=Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Get the latest insight report for a survey."""
    survey = await db.get(Survey, survey_id)
    if not survey or survey.admin_id != admin.id:
        raise HTTPException(status_code=404, detail="Survey not found")

    insight = await db.scalar(
        select(SurveyInsight)
        .where(SurveyInsight.survey_id == survey_id)
        .order_by(SurveyInsight.generated_at.desc())
    )

    if not insight:
        if survey.status == "closed":
            return {
                "status": "pending",
                "message": "Insights are being generated. Check back in a few minutes.",
            }
        raise HTTPException(status_code=404, detail="No insights yet — survey may still be active")

    return {
        "survey_id": str(survey_id),
        "survey_title": survey.title,
        "generated_at": insight.generated_at.isoformat(),
        "completion_rate": insight.completion_rate,
        "executive_summary": insight.executive_summary,
        "themes": insight.themes,
        "action_items": insight.action_items,
        "sentiment_breakdown": insight.sentiment_breakdown,
    }


@router.post("/{survey_id}/trigger", status_code=202)
async def trigger_insights(
    survey_id: UUID,
    admin=Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Manually trigger insight generation for a survey."""
    survey = await db.get(Survey, survey_id)
    if not survey or survey.admin_id != admin.id:
        raise HTTPException(status_code=404, detail="Survey not found")

    task = generate_survey_insights.apply_async(args=[str(survey_id)], queue="insights")
    return {"status": "queued", "task_id": task.id, "survey_id": str(survey_id)}
