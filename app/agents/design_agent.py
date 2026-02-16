"""
Design Agent
─────────────
Helps admins craft high-quality surveys: bias detection, clarity checks,
question-type recommendations, length optimization, and A/B variant generation.
Uses Anthropic tool calling for structured output.
"""
from __future__ import annotations

import json
import logging
import time

from anthropic import AsyncAnthropic

from app.config import settings
from app.rag.knowledge_base import retrieve_guidelines
from app.schemas import (
    GenerateVariantsResult,
    QualityCheckResult,
)

logger = logging.getLogger(__name__)

client = AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)

# ─── Tool schemas (Anthropic format) ──────────────────────────────────────────
# Key differences from OpenAI:
#   - No outer {"type": "function", "function": {...}} wrapper — tools are flat
#   - "input_schema" instead of "parameters"
#   - tool_choice is {"type": "tool", "name": "..."} not {"type": "function", ...}
#   - Response: find block with type=="tool_use" in response.content; .input is already a dict

QUALITY_CHECK_TOOL = {
    "name": "quality_check_result",
    "description": "Returns quality analysis of a survey",
    "input_schema": {
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
                        "question_id", "bias_type", "severity",
                        "original_text", "suggestion", "explanation",
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
            "overall_quality_score", "estimated_completion_rate",
            "estimated_time_seconds", "bias_flags",
            "clarity_issues", "length_recommendation",
        ],
    },
}

GENERATE_VARIANTS_TOOL = {
    "name": "generate_variants_result",
    "description": "Returns A/B survey variants",
    "input_schema": {
        "type": "object",
        "properties": {
            "variants": {
                "type": "array",
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
                                    "options": {"type": "array", "items": {"type": "string"}},
                                    "required": {"type": "boolean"},
                                },
                                "required": ["id", "text", "type", "required"],
                            },
                        },
                        "hypothesis": {"type": "string"},
                        "predicted_completion_rate": {"type": "number"},
                        "key_differences": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": [
                        "variant_label", "questions", "hypothesis",
                        "predicted_completion_rate", "key_differences",
                    ],
                },
            }
        },
        "required": ["variants"],
    },
}


# ─── Design Agent ─────────────────────────────────────────────────────────────


class DesignAgent:
    """
    Admin-facing agent.

    Methods
    -------
    quality_check(survey_title, questions, specialty) → QualityCheckResult
    improve_question(question) → dict
    generate_variants(title, questions, num_variants) → GenerateVariantsResult
    suggest_question_types(survey_goal) → list[dict]
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
        guidelines = await retrieve_guidelines(survey_title)

        response = await client.messages.create(
            model=settings.ANTHROPIC_MODEL,
            max_tokens=4096,
            system=self.SYSTEM_PROMPT,
            tools=[QUALITY_CHECK_TOOL],
            tool_choice={"type": "tool", "name": "quality_check_result"},
            messages=[{
                "role": "user",
                "content": f"""Analyze this survey for quality, bias, and clarity.

Survey Title: {survey_title}
Target Specialty: {specialty or "All specialties"}
Questions:
{json.dumps(questions, indent=2)}

Relevant Platform Guidelines:
{guidelines}

Run a comprehensive quality check using the quality_check_result tool.""",
            }],
        )

        latency_ms = int((time.monotonic() - t0) * 1000)
        # Anthropic: find the tool_use block and read .input (already a dict)
        tool_use = next(b for b in response.content if b.type == "tool_use")
        data = tool_use.input

        logger.info(
            f"design_agent.quality_check survey={survey_title} latency_ms={latency_ms} "
            f"score={data.get('overall_quality_score')} bias_count={len(data.get('bias_flags', []))}"
        )
        return QualityCheckResult(**data)

    async def improve_question(self, question: dict) -> dict:
        """Return an improved version of a single question."""
        response = await client.messages.create(
            model=settings.ANTHROPIC_MODEL,
            max_tokens=1024,
            system=self.SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": f"""Improve this survey question for clarity, neutrality, and mobile-friendliness.

Original question:
{json.dumps(question, indent=2)}

Improvements to apply:
- Remove bias or leading language
- Simplify wording (reading level ≤ grade 8)
- If MCQ: ensure options are complete, mutually exclusive, balanced
- Add a brief 'hint' field (1 sentence) the doctor can reveal if confused

Respond with ONLY a JSON object (no markdown fences).""",
            }],
        )

        text = response.content[0].text.strip()
        if text.startswith("```"):
            text = text.split("```")[1].lstrip("json").strip()
        improved = json.loads(text)
        logger.info(f"design_agent.improve_question question_id={question.get('id')}")
        return improved

    async def generate_variants(
        self,
        title: str,
        questions: list[dict],
        num_variants: int = 2,
    ) -> GenerateVariantsResult:
        """Generate A/B test variants with predicted completion rates."""
        response = await client.messages.create(
            model=settings.ANTHROPIC_MODEL,
            max_tokens=4096,
            system=self.SYSTEM_PROMPT,
            tools=[GENERATE_VARIANTS_TOOL],
            tool_choice={"type": "tool", "name": "generate_variants_result"},
            messages=[{
                "role": "user",
                "content": f"""Create {num_variants} A/B test variants of this survey.

Survey: {title}
Original Questions:
{json.dumps(questions, indent=2)}

Variant strategy:
- Variant A: Keep original order, polish wording
- Variant B: Reorder to most engaging questions first, trim to shortest viable set
- Each variant must have its own hypothesis and predicted completion rate

Use the generate_variants_result tool.""",
            }],
        )

        tool_use = next(b for b in response.content if b.type == "tool_use")
        data = tool_use.input
        logger.info(f"design_agent.generate_variants title={title} num_variants={num_variants}")
        return GenerateVariantsResult(**data)

    async def suggest_question_types(self, survey_goal: str) -> list[dict]:
        """Given a survey goal, suggest the best question structure."""
        response = await client.messages.create(
            model=settings.ANTHROPIC_MODEL,
            max_tokens=2048,
            system=self.SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": f"""A healthcare admin wants to run a survey with this goal:
"{survey_goal}"

Suggest 5-8 questions with ideal question types, options (if MCQ/Likert), and a brief rationale.
Return ONLY a JSON object in this exact format (no markdown fences):
{{"questions": [{{"text": "...", "type": "...", "options": [...], "rationale": "..."}}]}}""",
            }],
        )

        text = response.content[0].text.strip()
        if text.startswith("```"):
            text = text.split("```")[1].lstrip("json").strip()
        data = json.loads(text)
        return data.get("questions", [])


# Module-level singleton
design_agent = DesignAgent()