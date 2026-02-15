"""
Design Agent
─────────────
Helps admins craft high-quality surveys: bias detection, clarity checks,
question-type recommendations, length optimization, and A/B variant generation.
Uses OpenAI function calling for structured output.
"""
from __future__ import annotations

import json
import time
import uuid
from typing import Any

from openai import AsyncOpenAI

from app.config import settings
from app.rag.knowledge_base import retrieve_guidelines
from app.schemas import (
    BiasFlag,
    GenerateVariantsResult,
    QualityCheckResult,
    Question,
    QuestionType,
    VariantSurvey,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)

client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

# ─── Tool schemas (OpenAI function calling) ───────────────────────────────────

QUALITY_CHECK_TOOL = {
    "type": "function",
    "function": {
        "name": "quality_check_result",
        "description": "Returns quality analysis of a survey",
        "parameters": {
            "type": "object",
            "properties": {
                "overall_quality_score": {
                    "type": "number",
                    "description": "Survey quality 0-10",
                },
                "estimated_completion_rate": {
                    "type": "number",
                    "description": "Predicted % of doctors who will complete",
                },
                "estimated_time_seconds": {
                    "type": "integer",
                    "description": "Estimated time to complete in seconds",
                },
                "bias_flags": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "question_id": {"type": "string"},
                            "bias_type": {
                                "type": "string",
                                "enum": [
                                    "leading_question",
                                    "loaded_term",
                                    "false_dichotomy",
                                    "double_barreled",
                                    "ambiguous",
                                    "jargon_heavy",
                                ],
                            },
                            "severity": {"type": "string", "enum": ["low", "medium", "high"]},
                            "original_text": {"type": "string"},
                            "suggestion": {"type": "string"},
                            "explanation": {"type": "string"},
                        },
                        "required": [
                            "question_id",
                            "bias_type",
                            "severity",
                            "original_text",
                            "suggestion",
                            "explanation",
                        ],
                    },
                },
                "clarity_issues": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "question_id": {"type": "string"},
                            "issue": {"type": "string"},
                            "suggestion": {"type": "string"},
                        },
                    },
                },
                "length_recommendation": {"type": "string"},
                "audience_suggestion": {"type": "string"},
            },
            "required": [
                "overall_quality_score",
                "estimated_completion_rate",
                "estimated_time_seconds",
                "bias_flags",
                "clarity_issues",
                "length_recommendation",
            ],
        },
    },
}

GENERATE_VARIANTS_TOOL = {
    "type": "function",
    "function": {
        "name": "generate_variants_result",
        "description": "Returns A/B survey variants",
        "parameters": {
            "type": "object",
            "properties": {
                "variants": {
                    "type": "array",
                    "minItems": 2,
                    "maxItems": 3,
                    "items": {
                        "type": "object",
                        "properties": {
                            "variant_label": {"type": "string"},
                            "questions": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "id": {"type": "string"},
                                        "text": {"type": "string"},
                                        "type": {
                                            "type": "string",
                                            "enum": ["mcq", "likert", "text", "boolean", "ranking"],
                                        },
                                        "options": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                            "nullable": True,
                                        },
                                        "required": {"type": "boolean"},
                                    },
                                    "required": ["id", "text", "type", "required"],
                                },
                            },
                            "hypothesis": {"type": "string"},
                            "predicted_completion_rate": {"type": "number"},
                            "key_differences": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": [
                            "variant_label",
                            "questions",
                            "hypothesis",
                            "predicted_completion_rate",
                            "key_differences",
                        ],
                    },
                }
            },
            "required": ["variants"],
        },
    },
}


# ─── Design Agent ─────────────────────────────────────────────────────────────


class DesignAgent:
    """
    Admin-facing agent.

    Methods
    -------
    quality_check(survey_title, questions, specialty) → QualityCheckResult
    improve_question(question) → Question
    generate_variants(title, questions, num_variants) → GenerateVariantsResult
    """

    SYSTEM_PROMPT = """You are an expert survey methodologist helping healthcare platform
admins create high-quality surveys for busy doctors.

CORE RESPONSIBILITIES:
1. Detect bias: leading questions, loaded terms, false dichotomies, double-barreled questions
2. Improve clarity: reduce jargon, simplify phrasing, ensure each question has one clear purpose
3. Optimize length: ideal survey = 5-10 questions, max 3 minutes to complete
4. Recommend question types: Likert for attitudes, MCQ for discrete choices, open-text sparingly
5. Suggest answer options: complete, mutually exclusive, balanced (no loaded order)

BIAS EXAMPLES:
BAD: "How much do you love our new EHR?" → Assumes positive sentiment (leading)
GOOD: "How satisfied are you with the new EHR?" [1-5 Likert]

BAD: "Don't you think staffing is the main issue?" → Double negative, leads to agreement
GOOD: "What is the most significant staffing challenge?" [open / MCQ]

BAD: "Do you prefer video or phone?" → False dichotomy, ignores context
GOOD: "Which consultation format do you use most?" + "Depends on situation" option

MANDATORY RULES:
- NEVER suggest collecting PHI (names, DOB, SSN, MRN, diagnosis, medications)
- Keep surveys under 10 questions for busy doctors
- Suggested completion time MUST be ≤ 3 minutes (180 seconds)
- All function outputs must be complete and valid JSON
"""

    async def quality_check(
        self,
        survey_title: str,
        questions: list[dict],
        specialty: str | None = None,
        admin_id: str | None = None,
    ) -> QualityCheckResult:
        """Run full quality check on a survey."""
        t0 = time.monotonic()

        # Retrieve relevant best-practice guidelines from RAG
        guidelines = await retrieve_guidelines(survey_title)

        prompt = f"""Analyze this survey for quality, bias, and clarity.

Survey Title: {survey_title}
Target Specialty: {specialty or "All specialties"}
Questions:
{json.dumps(questions, indent=2)}

Relevant Platform Guidelines:
{guidelines}

Run a comprehensive quality check using the quality_check_result function."""

        response = await client.chat.completions.create(
            model=settings.OPENAI_LLM_MODEL,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            tools=[QUALITY_CHECK_TOOL],
            tool_choice={"type": "function", "function": {"name": "quality_check_result"}},
            temperature=0.2,
        )

        latency_ms = int((time.monotonic() - t0) * 1000)
        tool_call = response.choices[0].message.tool_calls[0]
        data = json.loads(tool_call.function.arguments)

        logger.info(
            "design_agent.quality_check",
            survey_title=survey_title,
            latency_ms=latency_ms,
            tokens=response.usage.total_tokens,
            score=data.get("overall_quality_score"),
            bias_count=len(data.get("bias_flags", [])),
        )

        return QualityCheckResult(**data)

    async def improve_question(self, question: dict) -> dict:
        """Return an improved version of a single question."""
        prompt = f"""Improve this survey question for clarity, neutrality, and mobile-friendliness.
Return ONLY a JSON object matching the Question schema (same fields as input).

Original question:
{json.dumps(question, indent=2)}

Improvements to apply:
- Remove bias or leading language
- Simplify wording (reading level ≤ grade 8)
- If MCQ: ensure options are complete, mutually exclusive, balanced
- Add a brief 'hint' field (1 sentence) the doctor can reveal if confused"""

        response = await client.chat.completions.create(
            model=settings.OPENAI_LLM_MODEL,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
        )

        improved = json.loads(response.choices[0].message.content)
        logger.info("design_agent.improve_question", question_id=question.get("id"))
        return improved

    async def generate_variants(
        self,
        title: str,
        questions: list[dict],
        num_variants: int = 2,
    ) -> GenerateVariantsResult:
        """Generate A/B test variants with predicted completion rates."""
        prompt = f"""Create {num_variants} A/B test variants of this survey.

Survey: {title}
Original Questions:
{json.dumps(questions, indent=2)}

Variant strategy:
- Variant A: Keep original order, polish wording
- Variant B: Reorder to most engaging questions first, trim to shortest viable set
- Each variant must have its own hypothesis and predicted completion rate

Use generate_variants_result function."""

        response = await client.chat.completions.create(
            model=settings.OPENAI_LLM_MODEL,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            tools=[GENERATE_VARIANTS_TOOL],
            tool_choice={"type": "function", "function": {"name": "generate_variants_result"}},
            temperature=0.4,
        )

        tool_call = response.choices[0].message.tool_calls[0]
        data = json.loads(tool_call.function.arguments)
        logger.info("design_agent.generate_variants", title=title, num_variants=num_variants)
        return GenerateVariantsResult(**data)

    async def suggest_question_types(self, survey_goal: str) -> list[dict]:
        """Given a survey goal, suggest the best question structure."""
        prompt = f"""A healthcare admin wants to run a survey with this goal:
"{survey_goal}"

Suggest 5-8 questions with ideal question types, options (if MCQ/Likert), and a brief rationale.
Return JSON: {{"questions": [...]}} where each item has: text, type, options, rationale"""

        response = await client.chat.completions.create(
            model=settings.OPENAI_LLM_MODEL,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.5,
        )

        data = json.loads(response.choices[0].message.content)
        return data.get("questions", [])


# Module-level singleton
design_agent = DesignAgent()
