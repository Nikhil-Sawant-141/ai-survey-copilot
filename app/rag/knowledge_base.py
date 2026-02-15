"""
Knowledge Base
──────────────
Manages the RAG pipeline:
  1. Seeds Pinecone indexes with best-practice guidelines
  2. Provides retrieve_guidelines() used by Design Agent
  3. Provides retrieve_templates() for template inspiration

Run seed_knowledge_base() once at startup / via CLI.
"""
from __future__ import annotations

import uuid

from app.config import settings
from app.rag.embeddings import embed_batch, embed_text
from app.rag.pinecone_client import query_index, upsert_vectors
import logging

logger = logging.getLogger(__name__)

# ─── Static knowledge: survey best practices ──────────────────────────────────
# In production, load these from a database or document store.

SURVEY_GUIDELINES: list[dict] = [
    {
        "id": "guide-001",
        "title": "Avoiding Leading Questions",
        "content": (
            "A leading question suggests a desired answer. Rephrase to be neutral. "
            "BAD: 'How much do you enjoy our platform?' "
            "GOOD: 'How would you rate your experience with the platform?'"
        ),
        "category": "bias",
    },
    {
        "id": "guide-002",
        "title": "Double-Barreled Questions",
        "content": (
            "Never ask about two things in one question. "
            "BAD: 'Are you satisfied with the speed and accuracy of the EHR?' "
            "GOOD: Split into two separate questions."
        ),
        "category": "clarity",
    },
    {
        "id": "guide-003",
        "title": "Optimal Survey Length for Doctors",
        "content": (
            "Doctors have limited time. Target 5-8 questions, under 3 minutes. "
            "Completion rates drop 50% past 10 questions. "
            "Use skip logic to hide irrelevant questions."
        ),
        "category": "length",
    },
    {
        "id": "guide-004",
        "title": "Likert Scale Best Practices",
        "content": (
            "Use odd-numbered scales (5 or 7 points) with labeled endpoints. "
            "Always include a midpoint. "
            "Example: 1=Very Dissatisfied, 3=Neutral, 5=Very Satisfied."
        ),
        "category": "question_types",
    },
    {
        "id": "guide-005",
        "title": "Multiple Choice Option Design",
        "content": (
            "MCQ options must be mutually exclusive and collectively exhaustive. "
            "Include 'Other (please specify)' when options might not cover all cases. "
            "Avoid ordered lists that could bias toward first or last items."
        ),
        "category": "question_types",
    },
    {
        "id": "guide-006",
        "title": "HIPAA Compliance in Surveys",
        "content": (
            "Never collect PHI: names, dates of birth, SSNs, MRNs, diagnosis codes, "
            "treatment details, or any patient-identifiable information. "
            "Anonymize all responses at rest. Retain for maximum 2 years."
        ),
        "category": "compliance",
    },
    {
        "id": "guide-007",
        "title": "Mobile-First Survey Design",
        "content": (
            "Over 60% of doctors complete surveys on mobile. "
            "Use single-column layouts. Limit open-text questions (voice input helps). "
            "Show progress bar. Enable auto-save every 10 seconds."
        ),
        "category": "ux",
    },
    {
        "id": "guide-008",
        "title": "Avoiding Loaded Language",
        "content": (
            "Loaded terms carry implicit assumptions. "
            "BAD: 'When did you stop struggling with documentation?' (assumes struggle) "
            "GOOD: 'How would you describe your documentation experience?'"
        ),
        "category": "bias",
    },
    {
        "id": "guide-009",
        "title": "Question Order Effects",
        "content": (
            "Place engaging, easy questions first to build momentum. "
            "Sensitive or open-ended questions should come later. "
            "Never place demographic questions first — they cause early drop-off."
        ),
        "category": "flow",
    },
    {
        "id": "guide-010",
        "title": "Telemedicine Survey Best Practices",
        "content": (
            "When surveying about telemedicine, ask separately about: "
            "technology (video quality, ease of use), clinical impact (patient outcomes), "
            "and workflow integration. Avoid conflating these dimensions."
        ),
        "category": "domain",
    },
    {
        "id": "guide-011",
        "title": "EHR Feedback Survey Design",
        "content": (
            "EHR surveys should separate: usability (navigation, speed), "
            "clinical workflow integration, documentation burden, and interoperability. "
            "Use Likert scales for ratings, open-text for specific pain points."
        ),
        "category": "domain",
    },
    {
        "id": "guide-012",
        "title": "Survey Fatigue Prevention",
        "content": (
            "Send no more than 1 survey per week per doctor. "
            "Rotate survey recipients across segments. "
            "Always communicate: why this survey, how long it takes, how data is used."
        ),
        "category": "engagement",
    },
]


async def seed_knowledge_base() -> None:
    """
    Embed all guidelines and upsert to Pinecone.
    Safe to run multiple times (upsert is idempotent).
    """
    logger.info(f"knowledge_base.seeding count={len(SURVEY_GUIDELINES)}")

    texts = [f"{g['title']}. {g['content']}" for g in SURVEY_GUIDELINES]
    embeddings = await embed_batch(texts)

    vectors = [
        {
            "id": g["id"],
            "values": emb,
            "metadata": {
                "title": g["title"],
                "content": g["content"],
                "category": g["category"],
            },
        }
        for g, emb in zip(SURVEY_GUIDELINES, embeddings)
    ]

    await upsert_vectors(settings.PINECONE_INDEX_GUIDELINES, vectors)
    logger.info(f"knowledge_base.seeded count={len(vectors)}")


async def retrieve_guidelines(query: str, top_k: int = 4) -> str:
    """
    Retrieve the most relevant guidelines for a survey topic.
    Returns formatted text for injection into agent prompts.
    """
    query_vector = await embed_text(query)
    matches = await query_index(
        settings.PINECONE_INDEX_GUIDELINES,
        vector=query_vector,
        top_k=top_k,
    )

    if not matches:
        return "No specific guidelines found — apply general best practices."

    sections = []
    for m in matches:
        meta = m["metadata"]
        sections.append(f"[{meta.get('category', 'general').upper()}] {meta.get('title')}\n{meta.get('content')}")

    return "\n\n".join(sections)


async def index_survey_template(survey: dict, completion_rate: float) -> None:
    """
    Index a completed survey as a template for future inspiration.
    Only indexes surveys with completion_rate ≥ 40%.
    """
    if completion_rate < 40:
        return

    text = f"{survey['title']}. {survey.get('description', '')}. " + " ".join(
        q.get("text", "") for q in survey.get("questions", [])
    )
    embedding = await embed_text(text)

    vector = {
        "id": str(survey.get("id", uuid.uuid4())),
        "values": embedding,
        "metadata": {
            "title": survey["title"],
            "question_count": len(survey.get("questions", [])),
            "completion_rate": completion_rate,
            "specialty": survey.get("targeting_rules", {}).get("specialty", "all"),
        },
    }

    await upsert_vectors(settings.PINECONE_INDEX_TEMPLATES, [vector])
    logger.info(
        f"knowledge_base.template_indexed survey_id={survey.get('id')} "
        f"completion_rate={completion_rate}"
    )


async def retrieve_similar_templates(query: str, top_k: int = 3) -> list[dict]:
    """Find high-performing similar surveys for admin inspiration."""
    query_vector = await embed_text(query)
    matches = await query_index(
        settings.PINECONE_INDEX_TEMPLATES,
        vector=query_vector,
        top_k=top_k,
    )
    return [m["metadata"] for m in matches]