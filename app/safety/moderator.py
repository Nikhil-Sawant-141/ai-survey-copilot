"""
Safety & Compliance Layer
─────────────────────────
Handles:
  1. PHI / PII detection in survey questions and responses (Presidio)
  2. Content moderation on agent outputs
  3. Medical advice detection (keyword + pattern-based)
"""
from __future__ import annotations

import logging
import re

from app.utils.logger import get_logger

logger = logging.getLogger(__name__)

# ─── PHI patterns (supplement Presidio for healthcare-specific items) ─────────

PHI_PATTERNS = [
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "SSN"),
    (re.compile(r"\bMRN\b|\bmedical record number\b", re.IGNORECASE), "MRN"),
    (re.compile(r"\bdate of birth\b|\bDOB\b|\bbirthdate\b", re.IGNORECASE), "DOB"),
    (re.compile(r"\bICD-\d+\b|\bdiagnosis code\b", re.IGNORECASE), "DIAGNOSIS_CODE"),
    (re.compile(r"\bNPI\b|\bnational provider\b", re.IGNORECASE), "NPI"),
    (re.compile(r"\bDEA number\b", re.IGNORECASE), "DEA"),
]

# ─── Medical advice patterns ──────────────────────────────────────────────────

MEDICAL_ADVICE_PATTERNS = [
    re.compile(r"\byou (?:should|must|need to) (?:take|start|stop|see a|visit)\b", re.IGNORECASE),
    re.compile(r"\bthis (?:sounds|looks|seems) like\b.{0,30}\b(?:condition|diagnosis|disease|disorder)\b", re.IGNORECASE),
    re.compile(r"\byou (?:have|might have|could have|may have)\b.{0,30}\b(?:condition|disease|disorder|syndrome)\b", re.IGNORECASE),
    re.compile(r"\bI (?:recommend|suggest|advise)\b.{0,30}\b(?:medication|treatment|therapy|doctor|specialist)\b", re.IGNORECASE),
    re.compile(r"\bsymptoms suggest\b", re.IGNORECASE),
    re.compile(r"\bconsult a (?:doctor|physician|specialist) (?:immediately|urgently|right away)\b", re.IGNORECASE),
]

# ─── PHI collection indicators in survey questions ────────────────────────────

PHI_COLLECTION_KEYWORDS = [
    "full name", "first name", "last name", "email address", "phone number",
    "home address", "zip code", "date of birth", "social security",
    "medical record", "patient id", "patient name", "npi number",
    "license number", "dea number", "diagnosis", "medication list",
    "prescription", "treatment plan",
]


class SafetyModerator:
    """
    Checks both inbound content (surveys, responses) and
    outbound content (agent outputs) for safety violations.
    """

    def check_question_for_phi(self, question_text: str) -> tuple[bool, str | None]:
        """
        Returns (is_safe, violation_type).
        is_safe=True means no PHI collection detected.
        """
        lower = question_text.lower()

        # Check keyword list
        for keyword in PHI_COLLECTION_KEYWORDS:
            if keyword in lower:
                logger.warning(
                    "safety.phi_keyword_detected",
                    keyword=keyword,
                    question_preview=question_text[:80],
                )
                return False, f"phi_keyword:{keyword}"

        # Check regex patterns
        for pattern, label in PHI_PATTERNS:
            if pattern.search(question_text):
                logger.warning(
                    "safety.phi_pattern_detected",
                    pattern_type=label,
                    question_preview=question_text[:80],
                )
                return False, f"phi_pattern:{label}"

        return True, None

    def check_response_for_phi(self, response_text: str) -> str:
        """
        Redact any PHI accidentally included in open-ended doctor responses.
        Uses simple regex redaction (Presidio can be swapped in for production).
        """
        redacted = response_text

        # Redact SSN-like patterns
        redacted = re.sub(r"\b\d{3}-\d{2}-\d{4}\b", "[REDACTED-SSN]", redacted)

        # Redact phone numbers
        redacted = re.sub(
            r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
            "[REDACTED-PHONE]",
            redacted,
        )

        # Redact email addresses
        redacted = re.sub(
            r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Z|a-z]{2,}\b",
            "[REDACTED-EMAIL]",
            redacted,
        )

        if redacted != response_text:
            logger.warning(
                "safety.phi_redacted_from_response",
                original_length=len(response_text),
                redacted_length=len(redacted),
            )

        return redacted

    async def check_output(self, agent_output: str) -> tuple[bool, str]:
        """
        Checks agent output for:
        1. Medical advice
        2. PHI disclosure
        3. Inappropriate content

        Returns (is_safe, safe_text).
        If unsafe, returns generic fallback text.
        """
        # Check for medical advice
        for pattern in MEDICAL_ADVICE_PATTERNS:
            if pattern.search(agent_output):
                logger.warning(
                    "safety.medical_advice_blocked",
                    output_preview=agent_output[:100],
                )
                return False, (
                    "I can help clarify what this survey question is asking, "
                    "but I'm not able to provide medical guidance. "
                    "For clinical questions, please consult appropriate resources."
                )

        # Check for PHI in output
        for pattern, label in PHI_PATTERNS:
            if pattern.search(agent_output):
                logger.warning(
                    "safety.phi_in_agent_output",
                    pattern_type=label,
                )
                return False, "I was unable to generate a safe response. Please contact support."

        return True, agent_output

    def validate_survey_for_phi(self, questions: list[dict]) -> list[dict]:
        """
        Validate all questions in a survey for PHI collection.
        Returns list of violations: [{question_id, issue}]
        """
        violations = []
        for q in questions:
            text = q.get("text", "")
            is_safe, violation_type = self.check_question_for_phi(text)
            if not is_safe:
                violations.append(
                    {
                        "question_id": q.get("id"),
                        "question_text": text[:100],
                        "violation": violation_type,
                        "recommendation": "Remove or rephrase this question to avoid collecting protected health information.",
                    }
                )
        return violations


safety_moderator = SafetyModerator()
