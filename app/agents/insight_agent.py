"""
Insight Agent
─────────────
Post-survey analysis: theme extraction from open-ended responses,
sentiment analysis, executive summary, and actionable recommendations.
Designed to run asynchronously after surveys close.
Uses Anthropic tool calling for structured output.
"""
from __future__ import annotations

import json
import logging
import time

from anthropic import AsyncAnthropic

from app.config import settings
from app.schemas import ActionItem, InsightResult

logger = logging.getLogger(__name__)

client = AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)

# ─── Tool schema (Anthropic format) ───────────────────────────────────────────

INSIGHT_TOOL = {
    "name": "insight_result",
    "description": "Returns structured analysis of survey responses",
    "input_schema": {
        "type": "object",
        "properties": {
            "executive_summary": {
                "type": "string",
                "description": "3-5 sentence summary of key findings, suitable for leadership",
            },
            "completion_rate": {
                "type": "number",
                "description": "% of recipients who completed the survey",
            },
            "themes": {
                "type": "array",
                "description": "3-5 major themes extracted from open-ended responses",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "description": {"type": "string"},
                        "prevalence_pct": {"type": "number"},
                        "sentiment": {
                            "type": "string",
                            "enum": ["positive", "negative", "neutral", "mixed"],
                        },
                        "representative_quotes": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["title", "description", "prevalence_pct", "sentiment"],
                },
            },
            "action_items": {
                "type": "array",
                "description": "Prioritized recommendations based on findings",
                "items": {
                    "type": "object",
                    "properties": {
                        "priority": {"type": "string", "enum": ["high", "medium", "low"]},
                        "description": {"type": "string"},
                        "owner_suggestion": {"type": "string"},
                    },
                    "required": ["priority", "description", "owner_suggestion"],
                },
            },
            "sentiment_breakdown": {
                "type": "object",
                "properties": {
                    "positive": {"type": "number"},
                    "negative": {"type": "number"},
                    "neutral": {"type": "number"},
                },
            },
            "segment_insights": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "segment": {"type": "string"},
                        "insight": {"type": "string"},
                    },
                },
            },
        },
        "required": [
            "executive_summary", "completion_rate", "themes",
            "action_items", "sentiment_breakdown", "segment_insights",
        ],
    },
}


# ─── Insight Agent ────────────────────────────────────────────────────────────


class InsightAgent:
    """
    Post-survey analysis agent.

    Methods
    -------
    analyze(survey_metadata, responses, completion_rate) → InsightResult
    """

    SYSTEM_PROMPT = """You are a healthcare survey analyst helping organizations
understand feedback from doctors and improve their operations.

RESPONSIBILITIES:
1. Extract 3-5 major themes from open-ended responses using semantic clustering
2. Assess sentiment per theme and overall
3. Calculate prevalence of each theme (% of responses mentioning it)
4. Generate prioritized, actionable recommendations
5. Identify differences between doctor segments (specialties, experience levels)
6. Write an executive summary that non-technical leaders can act on

ANALYSIS STANDARDS:
- Themes must be grounded in actual responses — no speculation
- Representative quotes must be paraphrased (never include identifiable info)
- Action items must be specific, measurable, and assigned to a realistic owner
- Segment insights only if sample size per segment ≥ 10 responses

SAFETY RULES:
- No medical advice or clinical recommendations
- No identification of individual respondents
- No reference to patient data
- Flag if responses suggest a compliance or safety concern (without diagnosing)
"""

    async def analyze(
        self,
        survey_metadata: dict,
        responses: list[dict],
        completion_rate: float,
    ) -> InsightResult:
        """
        Full post-survey analysis. Called after survey closes.
        Handles large response sets by chunking open-ended answers.
        """
        t0 = time.monotonic()

        if not responses:
            return self._empty_result(completion_rate)

        open_responses = self._extract_open_responses(responses)
        quant_summary = self._summarize_quantitative(responses, survey_metadata)

        response = await client.messages.create(
            model=settings.ANTHROPIC_MODEL,
            max_tokens=4096,
            system=self.SYSTEM_PROMPT,
            tools=[INSIGHT_TOOL],
            tool_choice={"type": "tool", "name": "insight_result"},
            messages=[{
                "role": "user",
                "content": f"""Analyze these survey results and generate comprehensive insights.

Survey: {survey_metadata.get('title')}
Survey Goal: {survey_metadata.get('description', 'Collect doctor feedback')}
Total Respondents: {len(responses)}
Completion Rate: {completion_rate:.1f}%

Quantitative Summary:
{json.dumps(quant_summary, indent=2)}

Open-Ended Responses (sample of up to 200):
{json.dumps(open_responses[:200], indent=2)}

Segment Distribution:
{json.dumps(self._get_segments(responses), indent=2)}

Generate full insights using the insight_result tool.
Focus on actionable findings. Paraphrase quotes — never include identifiable info.""",
            }],
        )

        latency_ms = int((time.monotonic() - t0) * 1000)

        # Anthropic: find tool_use block, .input is already a dict
        tool_use = next(b for b in response.content if b.type == "tool_use")
        data = tool_use.input

        # Always override with actual completion rate — don't trust LLM math
        data["completion_rate"] = completion_rate

        logger.info(
            f"insight_agent.analyze survey_id={survey_metadata.get('id')} "
            f"responses_count={len(responses)} themes_found={len(data.get('themes', []))} "
            f"latency_ms={latency_ms}"
        )
        return InsightResult(**data)

    # ─── Private helpers ──────────────────────────────────────────────────────

    def _extract_open_responses(self, responses: list[dict]) -> list[str]:
        """Pull all text-type answers into a flat list.

        answers is a dict of {question_id: value}, not a list —
        iterate over .values() to get the actual answer values.
        """
        texts = []
        for r in responses:
            answers = r.get("answers", {})
            if not isinstance(answers, dict):
                continue
            for value in answers.values():
                if isinstance(value, str) and len(value) > 10:
                    texts.append(value)
        return texts

    def _summarize_quantitative(
        self, responses: list[dict], survey_metadata: dict
    ) -> dict:
        """Compute mean/distribution for Likert/MCQ questions."""
        questions = survey_metadata.get("questions", [])
        summary: dict[str, dict] = {}

        for q in questions:
            qid = q.get("id")
            qtype = q.get("type")

            if qtype not in ("likert", "mcq", "boolean"):
                continue

            values = [
                r["answers"].get(qid)
                for r in responses
                if isinstance(r.get("answers"), dict) and r["answers"].get(qid) is not None
            ]

            if not values:
                continue

            if qtype == "likert":
                numeric = [v for v in values if isinstance(v, (int, float))]
                if numeric:
                    summary[qid] = {
                        "type": "likert",
                        "question": q.get("text"),
                        "mean": round(sum(numeric) / len(numeric), 2),
                        "n": len(numeric),
                    }
            elif qtype in ("mcq", "boolean"):
                counts: dict = {}
                for v in values:
                    counts[str(v)] = counts.get(str(v), 0) + 1
                summary[qid] = {
                    "type": qtype,
                    "question": q.get("text"),
                    "distribution": counts,
                    "n": len(values),
                }

        return summary

    def _get_segments(self, responses: list[dict]) -> dict:
        """Summarize response counts by doctor segment."""
        segments: dict[str, int] = {}
        for r in responses:
            specialty = r.get("doctor_specialty", "Unknown")
            segments[specialty] = segments.get(specialty, 0) + 1
        return segments

    def _empty_result(self, completion_rate: float) -> InsightResult:
        return InsightResult(
            executive_summary="No responses were received for this survey.",
            completion_rate=completion_rate,
            themes=[],
            action_items=[
                ActionItem(
                    priority="high",
                    description="Investigate why no responses were received — check targeting and send timing.",
                    owner_suggestion="Survey Administrator",
                )
            ],
            sentiment_breakdown={"positive": 0.0, "negative": 0.0, "neutral": 1.0},
            segment_insights=[],
        )


# Module-level singleton
insight_agent = InsightAgent()