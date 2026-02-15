"""
Agent Orchestrator
──────────────────
Central router: validates context, applies rate limits, dispatches to the
correct specialized agent, logs all interactions for auditing.
"""
from __future__ import annotations

import time
import uuid
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.attempt_agent import attempt_agent
from app.agents.design_agent import design_agent
from app.agents.insight_agent import insight_agent
from app.models import AgentInteractionLog
from app.redis_client import check_rate_limit
from app.safety.moderator import safety_moderator
from app.schemas import (
    ClarificationResult,
    CompletionSummary,
    GenerateVariantsResult,
    InsightResult,
    ProgressMessage,
    QualityCheckResult,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)


class AgentOrchestrator:
    """
    Routes agent tasks, enforces rate limits, runs safety checks,
    and persists interaction logs.
    """

    async def run_quality_check(
        self,
        survey_title: str,
        questions: list[dict],
        admin_id: str,
        specialty: str | None,
        db: AsyncSession,
    ) -> QualityCheckResult:
        # Rate limit: 100 quality checks/hour per admin
        allowed = await check_rate_limit(
            f"design:{admin_id}", limit=100, window_seconds=3600
        )
        if not allowed:
            raise ValueError("Rate limit exceeded: try again in an hour")

        t0 = time.monotonic()
        result = await design_agent.quality_check(
            survey_title, questions, specialty, admin_id
        )
        await self._log(
            db=db,
            agent_type="design",
            user_id=admin_id,
            input_ctx={"action": "quality_check", "title": survey_title},
            output=result.model_dump(),
            latency_ms=int((time.monotonic() - t0) * 1000),
        )
        return result

    async def run_generate_variants(
        self,
        title: str,
        questions: list[dict],
        admin_id: str,
        num_variants: int,
        db: AsyncSession,
    ) -> GenerateVariantsResult:
        t0 = time.monotonic()
        result = await design_agent.generate_variants(title, questions, num_variants)
        await self._log(
            db=db,
            agent_type="design",
            user_id=admin_id,
            input_ctx={"action": "generate_variants", "title": title},
            output={"variants_count": len(result.variants)},
            latency_ms=int((time.monotonic() - t0) * 1000),
        )
        return result

    async def run_clarify_question(
        self,
        session_id: str,
        survey_id: str,
        question: dict,
        doctor_id: str,
        doctor_context: dict | None,
        db: AsyncSession,
    ) -> ClarificationResult:
        # Rate limit: 10 clarifications per survey per doctor
        allowed = await check_rate_limit(
            f"clarify:{doctor_id}:{survey_id}", limit=10, window_seconds=86400
        )
        if not allowed:
            raise ValueError("Clarification limit reached for this survey")

        t0 = time.monotonic()
        result = await attempt_agent.clarify_question(session_id, question, doctor_context)

        # Safety check on output
        safe, filtered_text = await safety_moderator.check_output(result.clarification)
        if not safe:
            result.clarification = filtered_text

        await self._log(
            db=db,
            agent_type="attempt",
            user_id=doctor_id,
            input_ctx={
                "action": "clarify_question",
                "question_id": question.get("id"),
                "survey_id": survey_id,
            },
            output={"clarification_length": len(result.clarification)},
            latency_ms=int((time.monotonic() - t0) * 1000),
        )
        return result

    async def run_get_progress(
        self,
        session_id: str,
        questions_total: int,
        questions_answered: int,
    ) -> ProgressMessage:
        return await attempt_agent.get_progress(session_id, questions_total, questions_answered)

    async def run_completion_summary(
        self,
        responses: list[dict],
        survey_title: str,
        total_responses: int,
        doctor_id: str,
        db: AsyncSession,
    ) -> CompletionSummary:
        t0 = time.monotonic()
        result = await attempt_agent.generate_completion_summary(
            responses, survey_title, total_responses
        )
        await self._log(
            db=db,
            agent_type="attempt",
            user_id=doctor_id,
            input_ctx={"action": "completion_summary", "responses_count": len(responses)},
            output=result.model_dump(),
            latency_ms=int((time.monotonic() - t0) * 1000),
        )
        return result

    async def run_insight_analysis(
        self,
        survey_metadata: dict,
        responses: list[dict],
        completion_rate: float,
        admin_id: str,
        db: AsyncSession,
    ) -> InsightResult:
        t0 = time.monotonic()
        result = await insight_agent.analyze(survey_metadata, responses, completion_rate)
        await self._log(
            db=db,
            agent_type="insight",
            user_id=admin_id,
            input_ctx={
                "action": "analyze",
                "survey_id": survey_metadata.get("id"),
                "responses": len(responses),
            },
            output={
                "themes": len(result.themes),
                "action_items": len(result.action_items),
            },
            latency_ms=int((time.monotonic() - t0) * 1000),
        )
        return result

    # ─── Private ──────────────────────────────────────────────────────────────

    async def _log(
        self,
        db: AsyncSession,
        agent_type: str,
        user_id: str,
        input_ctx: dict,
        output: dict,
        latency_ms: int,
    ) -> None:
        try:
            log = AgentInteractionLog(
                agent_type=agent_type,
                user_id=uuid.UUID(user_id) if user_id else None,
                input_context=input_ctx,
                output_response=output,
                latency_ms=latency_ms,
            )
            db.add(log)
            await db.flush()
        except Exception as e:
            # Non-critical — don't fail the request over logging
            logger.error("orchestrator.log_failed", error=str(e))


orchestrator = AgentOrchestrator()
