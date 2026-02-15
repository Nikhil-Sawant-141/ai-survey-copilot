"""
Responses Router
────────────────
POST /responses           Submit survey response (partial or complete)
GET  /responses/{id}      Get response status
GET  /surveys/{id}/responses  Admin: list all responses for a survey
"""
from datetime import datetime
import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import Response, Survey, SurveyEvent
from app.routers.auth import require_admin, require_doctor
from app.safety.moderator import safety_moderator
from app.schemas import ResponseCreate, ResponseOut
from app.tasks.celery_app import send_completion_reminder
from app.utils.logger import get_logger

router = APIRouter(prefix="/responses", tags=["responses"])
logger = logging.getLogger(__name__)

@router.post("", response_model=ResponseOut, status_code=201)
async def submit_response(
    payload: ResponseCreate,
    doctor=Depends(require_doctor),
    db: AsyncSession = Depends(get_db),
):
    """
    Submit survey answers. Can be called multiple times for partial saves.
    Final submission: set is_complete=True.
    """
    # Verify survey exists and is active
    survey = await db.get(Survey, payload.survey_id)
    if not survey:
        raise HTTPException(status_code=404, detail="Survey not found")
    if survey.status != "active":
        raise HTTPException(status_code=409, detail="Survey is not accepting responses")

    # Sanitize open-text answers for PHI
    answers_dict = {}
    for answer in payload.answers:
        value = answer.value
        if isinstance(value, str):
            value = safety_moderator.check_response_for_phi(value)
        answers_dict[answer.question_id] = value

    # Check if response already exists (partial save → update)
    existing = await db.scalar(
        select(Response).where(
            Response.survey_id == payload.survey_id,
            Response.doctor_id == doctor.id,
        )
    )

    if existing:
        existing.answers = answers_dict
        existing.is_complete = payload.is_complete
        existing.device_type = payload.device_type
        existing.time_spent_seconds = payload.time_spent_seconds
        if payload.is_complete and not existing.completed_at:
            existing.completed_at = datetime.utcnow()
        response_obj = existing
    else:
        response_obj = Response(
            survey_id=payload.survey_id,
            doctor_id=doctor.id,
            answers=answers_dict,
            is_complete=payload.is_complete,
            device_type=payload.device_type,
            time_spent_seconds=payload.time_spent_seconds,
            completed_at=datetime.utcnow() if payload.is_complete else None,
        )
        db.add(response_obj)

    # Log event
    event_type = "survey_completed" if payload.is_complete else "survey_partial_save"
    db.add(SurveyEvent(
        survey_id=payload.survey_id,
        doctor_id=doctor.id,
        event_type=event_type,
        metadata={"is_complete": payload.is_complete, "answers_count": len(answers_dict)},
    ))

    await db.flush()

    # If partial: schedule a reminder (24h later) if not already sent
    if not payload.is_complete:
        send_completion_reminder.apply_async(
            args=[str(doctor.id), str(payload.survey_id)],
            countdown=86400,  # 24 hours
            queue="reminders",
        )

    logger.info(
        "response.submitted",
        survey_id=str(payload.survey_id),
        doctor_id=str(doctor.id),
        is_complete=payload.is_complete,
    )
    return response_obj


@router.get("/{response_id}", response_model=ResponseOut)
async def get_response(
    response_id: UUID,
    doctor=Depends(require_doctor),
    db: AsyncSession = Depends(get_db),
):
    response = await db.get(Response, response_id)
    if not response or response.doctor_id != doctor.id:
        raise HTTPException(status_code=404, detail="Response not found")
    return response


# ─── Admin: view survey responses ─────────────────────────────────────────────

responses_admin_router = APIRouter(prefix="/surveys", tags=["admin-responses"])


@responses_admin_router.get("/{survey_id}/responses")
async def list_survey_responses(
    survey_id: UUID,
    complete_only: bool = False,
    admin=Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Admin: list all responses for a survey."""
    survey = await db.get(Survey, survey_id)
    if not survey or survey.admin_id != admin.id:
        raise HTTPException(status_code=404, detail="Survey not found")

    q = select(Response).where(Response.survey_id == survey_id)
    if complete_only:
        q = q.where(Response.is_complete == True)

    responses = await db.scalars(q)
    results = list(responses)

    return {
        "survey_id": str(survey_id),
        "total": len(results),
        "complete": sum(1 for r in results if r.is_complete),
        "completion_rate": (
            round(sum(1 for r in results if r.is_complete) / len(results) * 100, 1)
            if results else 0.0
        ),
        "responses": [
            {
                "id": str(r.id),
                "is_complete": r.is_complete,
                "started_at": r.started_at.isoformat(),
                "completed_at": r.completed_at.isoformat() if r.completed_at else None,
                "time_spent_seconds": r.time_spent_seconds,
                "device_type": r.device_type,
            }
            for r in results
        ],
    }
