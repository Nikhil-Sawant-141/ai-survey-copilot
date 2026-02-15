"""
Agents Router
─────────────
Exposes all AI agent functionality via REST.

ADMIN endpoints:
  POST /agents/quality-check           Run Design Agent quality check
  POST /agents/improve-question        Improve a single question
  POST /agents/generate-variants       Generate A/B test variants
  POST /agents/suggest-questions       Suggest questions from survey goal

DOCTOR endpoints:
  POST /agents/clarify                 Clarify a survey question
  GET  /agents/progress                Get progress message
  POST /agents/completion-summary      Generate post-completion summary
  POST /agents/save-progress           Save partial answers
  GET  /agents/restore/{session_id}    Restore in-progress session
"""
import logging
import uuid
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.orchestrator import orchestrator
from app.database import get_db
from app.models import User
from app.routers.auth import get_current_user, require_admin, require_doctor
from app.schemas import (
    ClarificationRequest,
    CompletionSummary,
    GenerateVariantsResult,
    InsightResult,
    ProgressMessage,
    QualityCheckResult,
)
from app.utils.logger import get_logger

router = APIRouter(prefix="/agents", tags=["agents"])
logger = logging.getLogger(__name__)


# ─── Admin Agent Endpoints ────────────────────────────────────────────────────


@router.post("/quality-check", response_model=QualityCheckResult)
async def quality_check(
    body: dict,
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """
    Run a full quality check on a survey's questions.

    Body:
      survey_title: str
      questions: list[dict]
      specialty: str | None
    """
    survey_title = body.get("survey_title", "")
    questions = body.get("questions", [])
    specialty = body.get("specialty")

    if not survey_title or not questions:
        raise HTTPException(status_code=422, detail="survey_title and questions are required")

    result = await orchestrator.run_quality_check(
        survey_title=survey_title,
        questions=questions,
        admin_id=str(admin.id),
        specialty=specialty,
        db=db,
    )

    # Optionally persist quality score back to survey
    survey_id = body.get("survey_id")
    if survey_id:
        from app.models import Survey
        survey = await db.get(Survey, uuid.UUID(survey_id))
        if survey and survey.admin_id == admin.id:
            survey.quality_score = result.overall_quality_score
            survey.predicted_completion_rate = result.estimated_completion_rate
            survey.estimated_time_seconds = result.estimated_time_seconds
            await db.flush()

    return result


@router.post("/improve-question")
async def improve_question(
    body: dict,
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """
    Improve a single question for clarity and neutrality.

    Body:
      question: dict (question object)
    """
    question = body.get("question")
    if not question:
        raise HTTPException(status_code=422, detail="question is required")

    from app.agents.design_agent import design_agent
    improved = await design_agent.improve_question(question)
    return {"improved_question": improved}


@router.post("/generate-variants", response_model=GenerateVariantsResult)
async def generate_variants(
    body: dict,
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """
    Generate A/B test variants of a survey.

    Body:
      title: str
      questions: list[dict]
      num_variants: int (default 2)
    """
    title = body.get("title", "")
    questions = body.get("questions", [])
    num_variants = body.get("num_variants", 2)

    if not title or not questions:
        raise HTTPException(status_code=422, detail="title and questions are required")

    result = await orchestrator.run_generate_variants(
        title=title,
        questions=questions,
        admin_id=str(admin.id),
        num_variants=num_variants,
        db=db,
    )
    return result


@router.post("/suggest-questions")
async def suggest_questions(
    body: dict,
    admin: User = Depends(require_admin),
):
    """
    Given a survey goal, suggest question structure.

    Body:
      survey_goal: str
    """
    survey_goal = body.get("survey_goal", "")
    if not survey_goal:
        raise HTTPException(status_code=422, detail="survey_goal is required")

    from app.agents.design_agent import design_agent
    suggestions = await design_agent.suggest_question_types(survey_goal)
    return {"suggested_questions": suggestions}


# ─── Doctor Agent Endpoints ───────────────────────────────────────────────────


@router.post("/clarify")
async def clarify_question(
    payload: ClarificationRequest,
    doctor: User = Depends(require_doctor),
    db: AsyncSession = Depends(get_db),
):
    """Clarify a survey question for the doctor."""
    # Load the question from survey
    from app.models import Survey
    survey = await db.get(Survey, payload.survey_id)
    if not survey or survey.status != "active":
        raise HTTPException(status_code=404, detail="Survey not found or not active")

    # Find the question in the survey
    question = next(
        (q for q in survey.questions if q.get("id") == payload.question_id), None
    )
    if not question:
        raise HTTPException(status_code=404, detail="Question not found in survey")

    # Log event
    from app.models import SurveyEvent
    event = SurveyEvent(
        survey_id=payload.survey_id,
        doctor_id=doctor.id,
        event_type="clarification_requested",
        question_id=payload.question_id,
    )
    db.add(event)

    result = await orchestrator.run_clarify_question(
        session_id=payload.session_id,
        survey_id=str(payload.survey_id),
        question=question,
        doctor_id=str(doctor.id),
        doctor_context={
            "specialty": doctor.specialty,
            "years_experience": doctor.years_experience,
        },
        db=db,
    )
    return result


@router.get("/progress")
async def get_progress(
    session_id: str,
    questions_total: int,
    questions_answered: int,
    doctor: User = Depends(require_doctor),
) -> ProgressMessage:
    """Get progress message for a doctor mid-survey."""
    return await orchestrator.run_get_progress(
        session_id=session_id,
        questions_total=questions_total,
        questions_answered=questions_answered,
    )


@router.post("/completion-summary", response_model=CompletionSummary)
async def completion_summary(
    body: dict,
    doctor: User = Depends(require_doctor),
    db: AsyncSession = Depends(get_db),
):
    """
    Generate personalized thank-you after survey completion.

    Body:
      responses: list[dict]
      survey_title: str
      total_responses: int
    """
    responses = body.get("responses", [])
    survey_title = body.get("survey_title", "")
    total_responses = body.get("total_responses", 1)

    result = await orchestrator.run_completion_summary(
        responses=responses,
        survey_title=survey_title,
        total_responses=total_responses,
        doctor_id=str(doctor.id),
        db=db,
    )
    return result


@router.post("/save-progress")
async def save_progress(
    body: dict,
    doctor: User = Depends(require_doctor),
):
    """Auto-save partial survey answers to Redis."""
    session_id = body.get("session_id")
    survey_id = body.get("survey_id")
    answers = body.get("answers", {})

    if not session_id or not survey_id:
        raise HTTPException(status_code=422, detail="session_id and survey_id required")

    from app.agents.attempt_agent import attempt_agent
    await attempt_agent.save_partial_progress(session_id, survey_id, answers)
    return {"status": "saved"}


@router.get("/restore/{session_id}")
async def restore_session(
    session_id: str,
    doctor: User = Depends(require_doctor),
):
    """Restore doctor's in-progress survey answers."""
    from app.agents.attempt_agent import attempt_agent
    session = await attempt_agent.restore_session(session_id)
    if not session:
        return {"found": False}
    return {"found": True, "session": session}
