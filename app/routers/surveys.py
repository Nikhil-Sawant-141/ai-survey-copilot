"""
Surveys Router
──────────────
Admin-facing CRUD for surveys.

POST   /surveys              Create survey
GET    /surveys              List admin's surveys
GET    /surveys/{id}         Get survey details
PATCH  /surveys/{id}         Update survey (draft only)
POST   /surveys/{id}/launch  Launch survey (changes status → active)
POST   /surveys/{id}/close   Close survey (triggers insight generation)
DELETE /surveys/{id}         Delete draft survey
"""
from datetime import datetime
import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import Survey
from app.routers.auth import require_admin
from app.safety.moderator import safety_moderator
from app.schemas import SurveyCreate, SurveyResponse, SurveyStatus, SurveyUpdate
from app.tasks.celery_app import generate_survey_insights
from app.utils.logger import get_logger

router = APIRouter(prefix="/surveys", tags=["surveys"])
logger = logging.getLogger(__name__)


@router.post("", response_model=SurveyResponse, status_code=201)
async def create_survey(
    payload: SurveyCreate,
    admin=Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Create a new survey (status=draft)."""
    # Safety: check for PHI in questions
    violations = safety_moderator.validate_survey_for_phi(
        [q.model_dump() for q in payload.questions]
    )
    if violations:
        raise HTTPException(
            status_code=422,
            detail={"message": "Survey contains potential PHI", "violations": violations},
        )

    survey = Survey(
        admin_id=admin.id,
        title=payload.title,
        description=payload.description,
        questions=[q.model_dump() for q in payload.questions],
        targeting_rules=payload.targeting_rules,
        # Rough estimate: 18 seconds per question
        estimated_time_seconds=len(payload.questions) * 18,
        status="draft",
    )
    db.add(survey)
    await db.flush()

    logger.info("survey.created", survey_id=str(survey.id), admin_id=str(admin.id))
    return survey


@router.get("", response_model=list[SurveyResponse])
async def list_surveys(
    status: SurveyStatus | None = None,
    admin=Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    q = select(Survey).where(Survey.admin_id == admin.id)
    if status:
        q = q.where(Survey.status == status.value)
    q = q.order_by(Survey.created_at.desc())
    result = await db.scalars(q)
    return list(result)


@router.get("/{survey_id}", response_model=SurveyResponse)
async def get_survey(
    survey_id: UUID,
    admin=Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    survey = await _get_admin_survey(survey_id, admin.id, db)
    return survey


@router.patch("/{survey_id}", response_model=SurveyResponse)
async def update_survey(
    survey_id: UUID,
    payload: SurveyUpdate,
    admin=Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    survey = await _get_admin_survey(survey_id, admin.id, db)

    if survey.status != "draft":
        raise HTTPException(status_code=409, detail="Only draft surveys can be edited")

    if payload.title is not None:
        survey.title = payload.title
    if payload.description is not None:
        survey.description = payload.description
    if payload.questions is not None:
        violations = safety_moderator.validate_survey_for_phi(
            [q.model_dump() for q in payload.questions]
        )
        if violations:
            raise HTTPException(status_code=422, detail={"violations": violations})
        survey.questions = [q.model_dump() for q in payload.questions]
        survey.estimated_time_seconds = len(payload.questions) * 18
    if payload.targeting_rules is not None:
        survey.targeting_rules = payload.targeting_rules

    survey.version += 1
    await db.flush()
    return survey


@router.post("/{survey_id}/launch", response_model=SurveyResponse)
async def launch_survey(
    survey_id: UUID,
    admin=Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    survey = await _get_admin_survey(survey_id, admin.id, db)

    if survey.status != "draft":
        raise HTTPException(status_code=409, detail="Only draft surveys can be launched")
    if not survey.quality_score:
        raise HTTPException(
            status_code=422,
            detail="Run AI quality check before launching (/agents/quality-check)",
        )

    survey.status = "active"
    survey.launched_at = datetime.utcnow()
    await db.flush()

    logger.info("survey.launched", survey_id=str(survey_id))
    return survey


@router.post("/{survey_id}/close", response_model=SurveyResponse)
async def close_survey(
    survey_id: UUID,
    admin=Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    survey = await _get_admin_survey(survey_id, admin.id, db)

    if survey.status != "active":
        raise HTTPException(status_code=409, detail="Survey is not active")

    survey.status = "closed"
    survey.closed_at = datetime.utcnow()
    await db.flush()

    # Trigger async insight generation
    generate_survey_insights.apply_async(args=[str(survey_id)], queue="insights")
    logger.info("survey.closed.insights_queued", survey_id=str(survey_id))

    return survey


@router.delete("/{survey_id}", status_code=204)
async def delete_survey(
    survey_id: UUID,
    admin=Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    survey = await _get_admin_survey(survey_id, admin.id, db)
    if survey.status != "draft":
        raise HTTPException(status_code=409, detail="Only draft surveys can be deleted")
    await db.delete(survey)


# ─── Helpers ──────────────────────────────────────────────────────────────────

async def _get_admin_survey(survey_id: UUID, admin_id, db: AsyncSession) -> Survey:
    survey = await db.get(Survey, survey_id)
    if not survey:
        raise HTTPException(status_code=404, detail="Survey not found")
    if survey.admin_id != admin_id:
        raise HTTPException(status_code=403, detail="Not your survey")
    return survey
