"""
All Pydantic v2 schemas for request / response validation.
"""
import uuid
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, EmailStr, Field, field_validator


# ─── Enums ────────────────────────────────────────────────────────────────────

class UserRole(str, Enum):
    admin = "admin"
    doctor = "doctor"


class QuestionType(str, Enum):
    mcq = "mcq"
    likert = "likert"
    text = "text"
    boolean = "boolean"
    ranking = "ranking"


class SurveyStatus(str, Enum):
    draft = "draft"
    active = "active"
    closed = "closed"


# ─── Auth ─────────────────────────────────────────────────────────────────────

class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8)
    role: UserRole
    specialty: str | None = None
    years_experience: int | None = None


class UserResponse(BaseModel):
    id: uuid.UUID
    email: str
    role: UserRole
    specialty: str | None
    years_experience: int | None
    created_at: datetime

    model_config = {"from_attributes": True}


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


# ─── Question ─────────────────────────────────────────────────────────────────

class SkipLogicRule(BaseModel):
    condition_question_id: str
    condition_value: Any
    action: str = "skip_to"     # skip_to | hide
    target_question_id: str


class Question(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    text: str = Field(min_length=5, max_length=500)
    type: QuestionType
    options: list[str] | None = None    # For MCQ / ranking
    required: bool = True
    skip_logic: list[SkipLogicRule] | None = None
    hint: str | None = None             # Populated by Attempt Agent


# ─── Survey ───────────────────────────────────────────────────────────────────

class SurveyCreate(BaseModel):
    title: str = Field(min_length=5, max_length=255)
    description: str | None = None
    questions: list[Question] = Field(min_length=1)
    targeting_rules: dict | None = None


class SurveyUpdate(BaseModel):
    title: str | None = None
    description: str | None = None
    questions: list[Question] | None = None
    targeting_rules: dict | None = None


class SurveyResponse(BaseModel):
    id: uuid.UUID
    title: str
    description: str | None
    questions: list[dict]
    targeting_rules: dict | None
    estimated_time_seconds: int | None
    quality_score: float | None
    predicted_completion_rate: float | None
    status: SurveyStatus
    version: int
    created_at: datetime
    launched_at: datetime | None

    model_config = {"from_attributes": True}


# ─── Survey Response (Doctor) ─────────────────────────────────────────────────

class AnswerItem(BaseModel):
    question_id: str
    value: Any                  # String / int / list[str] depending on type


class ResponseCreate(BaseModel):
    survey_id: uuid.UUID
    answers: list[AnswerItem]
    is_complete: bool = False
    device_type: str | None = None
    time_spent_seconds: int | None = None


class ResponseOut(BaseModel):
    id: uuid.UUID
    survey_id: uuid.UUID
    is_complete: bool
    started_at: datetime
    completed_at: datetime | None

    model_config = {"from_attributes": True}


# ─── Agent Payloads ───────────────────────────────────────────────────────────

# Design Agent
class BiasFlag(BaseModel):
    question_id: str
    bias_type: str
    severity: str               # low | medium | high
    original_text: str
    suggestion: str
    explanation: str


class QualityCheckResult(BaseModel):
    overall_quality_score: float = Field(ge=0, le=10)
    estimated_completion_rate: float = Field(ge=0, le=100)
    estimated_time_seconds: int
    bias_flags: list[BiasFlag]
    clarity_issues: list[dict]
    length_recommendation: str
    audience_suggestion: str | None


class VariantSurvey(BaseModel):
    variant_label: str          # "A" | "B"
    questions: list[Question]
    hypothesis: str
    predicted_completion_rate: float
    key_differences: list[str]


class GenerateVariantsResult(BaseModel):
    variants: list[VariantSurvey]


# Attempt Agent
class ClarificationRequest(BaseModel):
    session_id: str
    survey_id: uuid.UUID
    question_id: str
    doctor_context: dict | None = None   # specialty, experience


class ClarificationResult(BaseModel):
    question_id: str
    clarification: str
    examples: list[str] | None
    did_change_meaning: bool = False     # Safety flag – must always be False


class ProgressMessage(BaseModel):
    questions_total: int
    questions_answered: int
    estimated_seconds_remaining: int
    motivational_message: str
    percent_complete: float


class CompletionSummary(BaseModel):
    thank_you_message: str
    aggregate_insight: str      # "You're 1 of 342 doctors who flagged X"
    next_steps: str


# Insight Agent
class Theme(BaseModel):
    title: str
    description: str
    prevalence_pct: float
    sentiment: str              # positive | negative | neutral | mixed
    representative_quotes: list[str]


class ActionItem(BaseModel):
    priority: str               # high | medium | low
    description: str
    owner_suggestion: str


class InsightResult(BaseModel):
    executive_summary: str
    completion_rate: float
    themes: list[Theme]
    action_items: list[ActionItem]
    sentiment_breakdown: dict   # {positive: 0.4, negative: 0.3, neutral: 0.3}
    segment_insights: list[dict]
