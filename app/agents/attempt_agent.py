"""
Attempt Agent
─────────────
Doctor-facing agent. Explains confusing questions (without changing meaning),
tracks progress, generates completion summaries, and manages session state.
"""
from __future__ import annotations

import json
import logging
import time
import uuid

from openai import AsyncOpenAI

from app.config import settings
from app.redis_client import cache_get, cache_set, get_session, set_session
from app.schemas import ClarificationResult, CompletionSummary, ProgressMessage
# from app.utils.logger import get_logger

logger = logging.getLogger(__name__)

client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

# ─── Tool schema ──────────────────────────────────────────────────────────────

CLARIFICATION_TOOL = {
    "type": "function",
    "function": {
        "name": "clarification_result",
        "description": "Returns a plain-English clarification for a survey question",
        "parameters": {
            "type": "object",
            "properties": {
                "clarification": {
                    "type": "string",
                    "description": "Clear explanation of what the question is asking (2-3 sentences max)",
                },
                "examples": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "1-2 anonymized example responses to help the doctor understand",
                },
                "did_change_meaning": {
                    "type": "boolean",
                    "description": "Safety flag – MUST always be false. Clarification must not change question intent.",
                },
            },
            "required": ["clarification", "did_change_meaning"],
        },
    },
}


# ─── Attempt Agent ────────────────────────────────────────────────────────────


class AttemptAgent:
    """
    Doctor-facing agent.

    Methods
    -------
    clarify_question(session_id, question, doctor_context) → ClarificationResult
    get_progress(session_id, questions_total, questions_answered) → ProgressMessage
    generate_completion_summary(responses, survey_title, total_responses) → CompletionSummary
    save_partial_progress(session_id, answers) → None
    restore_session(session_id) → dict | None
    """

    SYSTEM_PROMPT = """You are a helpful assistant inside a survey app for busy doctors.
Your role is to help doctors UNDERSTAND survey questions — not to answer questions for them.

CRITICAL RULES:
1. NEVER change the meaning of the original question
2. NEVER provide medical advice, diagnosis, or clinical recommendations
3. NEVER tell the doctor what answer to choose
4. Keep clarifications SHORT (2-3 sentences max)
5. Use simple, plain English — no jargon
6. If the question mentions a specific term (NPS, Likert, etc.), define it briefly
7. Provide 1-2 anonymized example responses to guide format, never content

FORBIDDEN RESPONSES:
- "Based on your symptoms..." (medical advice)
- "You should select option X because..." (influencing answer)
- "This question is poorly worded..." (criticizing the survey)

GOOD RESPONSE PATTERN:
Question: "What is your NPS for the current EHR workflow?"
Clarification: "This asks how likely you are to recommend the current EHR workflow to a colleague, 
on a scale of 0 (not at all) to 10 (extremely likely). Focus on your overall experience, 
not any single feature."
Examples: ["I'd give it a 7 because it handles scheduling well", "I'd rate it 4 due to frequent slowdowns"]
"""

    async def clarify_question(
        self,
        session_id: str,
        question: dict,
        doctor_context: dict | None = None,
    ) -> ClarificationResult:
        """Return a plain-English clarification for a survey question."""
        t0 = time.monotonic()

        # Check cache first — same question text often asked by many doctors
        cache_key = f"clarification:{hash(question.get('text', ''))}"
        cached = await cache_get(cache_key)
        if cached:
            logger.info("attempt_agent.clarify_question.cache_hit", question_id=question.get("id"))
            return ClarificationResult(**cached)

        specialty = (doctor_context or {}).get("specialty", "General")
        experience = (doctor_context or {}).get("years_experience", "unknown")

        prompt = f"""A doctor needs help understanding this survey question.

Doctor context: {specialty} specialty, {experience} years experience

Question to clarify:
{json.dumps(question, indent=2)}

Provide a clarification using the clarification_result function.
Remember: explain the question, do NOT suggest an answer."""

        response = await client.chat.completions.create(
            model=settings.OPENAI_LLM_MODEL,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            tools=[CLARIFICATION_TOOL],
            tool_choice={"type": "function", "function": {"name": "clarification_result"}},
            temperature=0.3,
        )

        latency_ms = int((time.monotonic() - t0) * 1000)
        tool_call = response.choices[0].message.tool_calls[0]
        data = json.loads(tool_call.function.arguments)

        # Safety assertion — clarification must never change meaning
        if data.get("did_change_meaning"):
            logger.warning(
                "attempt_agent.clarify_question.meaning_changed",
                question_id=question.get("id"),
            )
            data["did_change_meaning"] = False

        result = ClarificationResult(
            question_id=question.get("id", ""),
            **data,
        )

        # Cache for 24h
        await cache_set(cache_key, result.model_dump(), ttl=86400)

        logger.info(
            "attempt_agent.clarify_question",
            session_id=session_id,
            question_id=question.get("id"),
            latency_ms=latency_ms,
        )
        return result

    async def get_progress(
        self,
        session_id: str,
        questions_total: int,
        questions_answered: int,
        avg_seconds_per_question: float = 18.0,
    ) -> ProgressMessage:
        """Calculate progress and return a motivational message."""
        remaining_questions = questions_total - questions_answered
        estimated_seconds_remaining = int(remaining_questions * avg_seconds_per_question)
        percent_complete = round((questions_answered / questions_total) * 100, 1)

        # Choose motivational message based on progress tier
        if percent_complete == 0:
            message = f"This survey takes about {int(questions_total * avg_seconds_per_question / 60)} min. Let's go!"
        elif percent_complete < 33:
            message = "Great start! Keep going."
        elif percent_complete < 66:
            message = f"Halfway there — only {remaining_questions} questions left!"
        elif percent_complete < 90:
            message = "Almost done! Your input makes a difference."
        else:
            message = f"Just {remaining_questions} more question{'s' if remaining_questions > 1 else ''}!"

        return ProgressMessage(
            questions_total=questions_total,
            questions_answered=questions_answered,
            estimated_seconds_remaining=estimated_seconds_remaining,
            motivational_message=message,
            percent_complete=percent_complete,
        )

    async def generate_completion_summary(
        self,
        responses: list[dict],
        survey_title: str,
        total_responses: int,
    ) -> CompletionSummary:
        """Generate a personalized thank-you + aggregate insight after completion."""
        prompt = f"""A doctor just completed a survey. Generate a brief, warm thank-you message.

Survey: {survey_title}
Total responses from all doctors so far: {total_responses}
This doctor's responses:
{json.dumps(responses, indent=2)}

Return JSON with:
{{
  "thank_you_message": "Warm 1-sentence thank you personalized to the survey topic",
  "aggregate_insight": "1 sentence about what collective responses are showing (make it feel impactful)",
  "next_steps": "1 sentence on what happens with this data"
}}

Rules:
- No medical advice
- Keep it under 3 sentences total
- Make the doctor feel their input mattered"""

        response = await client.chat.completions.create(
            model=settings.OPENAI_LLM_MODEL,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.6,
        )

        data = json.loads(response.choices[0].message.content)
        return CompletionSummary(**data)

    # ─── Session management ───────────────────────────────────────────────────

    async def save_partial_progress(
        self, session_id: str, survey_id: str, answers: dict
    ) -> None:
        """Persist partial answers to Redis so doctor can resume later."""
        session = await get_session(session_id) or {}
        session.update(
            {
                "survey_id": survey_id,
                "answers": answers,
                "last_saved": time.time(),
            }
        )
        await set_session(session_id, session, ttl=604800)  # 7 days
        logger.info("attempt_agent.save_progress", session_id=session_id, answers_count=len(answers))

    async def restore_session(self, session_id: str) -> dict | None:
        """Restore doctor's in-progress answers from Redis."""
        session = await get_session(session_id)
        if session:
            logger.info(
                "attempt_agent.restore_session",
                session_id=session_id,
                answers_count=len(session.get("answers", {})),
            )
        return session


# Module-level singleton
attempt_agent = AttemptAgent()
