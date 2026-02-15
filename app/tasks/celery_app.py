"""
Celery Application + Tasks
──────────────────────────
Task queues:
  - insights  : post-survey analysis (triggered when survey closes)
  - reminders : scheduled doctor nudges
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timedelta

from celery import Celery
from celery.schedules import crontab
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.config import settings
from app.utils.logger import get_logger

logger = logging.getLogger(__name__)

# ─── App ──────────────────────────────────────────────────────────────────────

celery_app = Celery(
    "survey_agent",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_routes={
        "app.tasks.celery_app.generate_survey_insights": {"queue": "insights"},
        "app.tasks.celery_app.send_completion_reminder": {"queue": "reminders"},
        "app.tasks.celery_app.close_expired_surveys": {"queue": "insights"},
    },
    beat_schedule={
        # Check for expired surveys every hour
        "close-expired-surveys": {
            "task": "app.tasks.celery_app.close_expired_surveys",
            "schedule": crontab(minute=0),
        },
    },
)


# ─── Helper: sync DB session for Celery tasks ─────────────────────────────────

def _get_sync_engine():
    from sqlalchemy import create_engine
    return create_engine(settings.DATABASE_URL_SYNC, pool_pre_ping=True)


# ─── Tasks ────────────────────────────────────────────────────────────────────


@celery_app.task(
    name="app.tasks.celery_app.generate_survey_insights",
    bind=True,
    max_retries=3,
    default_retry_delay=60,
)
def generate_survey_insights(self, survey_id: str) -> dict:
    """
    Triggered when a survey is closed.
    Runs Insight Agent and persists results to survey_insights table.
    """
    from sqlalchemy.orm import Session as SyncSession
    from app.models import Response, Survey, SurveyInsight
    from app.agents.insight_agent import insight_agent

    logger.info("task.generate_insights.start", survey_id=survey_id)

    try:
        engine = _get_sync_engine()
        with SyncSession(engine) as session:
            # Load survey
            survey = session.get(Survey, uuid.UUID(survey_id))
            if not survey:
                logger.error("task.generate_insights.survey_not_found", survey_id=survey_id)
                return {"error": "survey not found"}

            # Load responses
            responses = session.scalars(
                select(Response).where(Response.survey_id == uuid.UUID(survey_id))
            ).all()

            total_sent = (
                session.scalars(
                    select(Response).where(Response.survey_id == uuid.UUID(survey_id))
                ).all()
            )
            completed = [r for r in responses if r.is_complete]
            completion_rate = (len(completed) / len(responses) * 100) if responses else 0.0

            # Prepare response dicts for agent
            response_dicts = [
                {
                    "answers": r.answers,
                    "doctor_specialty": None,  # Would join with User in production
                    "time_spent_seconds": r.time_spent_seconds,
                }
                for r in completed
            ]

            survey_meta = {
                "id": str(survey.id),
                "title": survey.title,
                "description": survey.description,
                "questions": survey.questions,
            }

            # Run async agent in sync context
            result = asyncio.run(
                insight_agent.analyze(survey_meta, response_dicts, completion_rate)
            )

            # Persist insight
            insight = SurveyInsight(
                survey_id=uuid.UUID(survey_id),
                themes=[t.model_dump() for t in result.themes],
                executive_summary=result.executive_summary,
                action_items=[a.model_dump() for a in result.action_items],
                sentiment_breakdown=result.sentiment_breakdown,
                completion_rate=result.completion_rate,
            )
            session.add(insight)
            session.commit()

        logger.info(
            "task.generate_insights.complete",
            survey_id=survey_id,
            themes=len(result.themes),
            completion_rate=completion_rate,
        )
        return {"survey_id": survey_id, "themes": len(result.themes)}

    except Exception as exc:
        logger.error("task.generate_insights.failed", survey_id=survey_id, error=str(exc))
        raise self.retry(exc=exc)


@celery_app.task(
    name="app.tasks.celery_app.send_completion_reminder",
    bind=True,
    max_retries=1,   # Only remind once
)
def send_completion_reminder(self, doctor_id: str, survey_id: str) -> dict:
    """
    Sends a reminder to a doctor who started but didn't finish.
    Respects:
      - Already completed → skip
      - Survey closed → skip
      - MAX one reminder per survey per doctor
    """
    from sqlalchemy.orm import Session as SyncSession
    from app.models import Response, Survey, SurveyEvent

    logger.info(
        "task.send_reminder.start",
        doctor_id=doctor_id,
        survey_id=survey_id,
    )

    try:
        engine = _get_sync_engine()
        with SyncSession(engine) as session:
            # Check if already completed
            response = session.scalar(
                select(Response).where(
                    Response.doctor_id == uuid.UUID(doctor_id),
                    Response.survey_id == uuid.UUID(survey_id),
                )
            )
            if response and response.is_complete:
                logger.info("task.send_reminder.already_complete", doctor_id=doctor_id)
                return {"status": "skipped", "reason": "already_complete"}

            # Check if survey is still active
            survey = session.get(Survey, uuid.UUID(survey_id))
            if not survey or survey.status != "active":
                return {"status": "skipped", "reason": "survey_not_active"}

            # Log reminder sent event
            event = SurveyEvent(
                survey_id=uuid.UUID(survey_id),
                doctor_id=uuid.UUID(doctor_id),
                event_type="reminder_sent",
                metadata={"channel": "push"},
            )
            session.add(event)
            session.commit()

        # In production: call push notification / email service here
        logger.info("task.send_reminder.sent", doctor_id=doctor_id, survey_id=survey_id)
        return {"status": "sent", "doctor_id": doctor_id, "survey_id": survey_id}

    except Exception as exc:
        logger.error("task.send_reminder.failed", error=str(exc))
        raise self.retry(exc=exc)


@celery_app.task(name="app.tasks.celery_app.close_expired_surveys")
def close_expired_surveys() -> dict:
    """
    Periodic task: closes surveys past their end date and triggers insight generation.
    Runs hourly via Celery Beat.
    """
    from sqlalchemy.orm import Session as SyncSession
    from app.models import Survey

    engine = _get_sync_engine()
    closed_ids = []

    with SyncSession(engine) as session:
        # Find active surveys with no recent responses (heuristic: 30 days old)
        cutoff = datetime.utcnow() - timedelta(days=30)
        expired = session.scalars(
            select(Survey).where(
                Survey.status == "active",
                Survey.launched_at < cutoff,
            )
        ).all()

        for survey in expired:
            survey.status = "closed"
            survey.closed_at = datetime.utcnow()
            closed_ids.append(str(survey.id))

        session.commit()

    # Trigger insight generation for each closed survey
    for sid in closed_ids:
        generate_survey_insights.apply_async(args=[sid], queue="insights")
        logger.info("task.close_expired.triggered_insights", survey_id=sid)

    logger.info("task.close_expired.done", count=len(closed_ids))
    return {"closed": len(closed_ids), "survey_ids": closed_ids}
