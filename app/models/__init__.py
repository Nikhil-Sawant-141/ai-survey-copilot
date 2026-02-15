import uuid
from datetime import datetime

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


def uuid_pk() -> Mapped[uuid.UUID]:
    return mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)


# ─── User ─────────────────────────────────────────────────────────────────────

class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = uuid_pk()
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(String(20), nullable=False)   # admin | doctor
    specialty: Mapped[str | None] = mapped_column(String(100))      # doctor-only
    years_experience: Mapped[int | None] = mapped_column(Integer)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    surveys: Mapped[list["Survey"]] = relationship(back_populates="admin")
    responses: Mapped[list["Response"]] = relationship(back_populates="doctor")


# ─── Survey ───────────────────────────────────────────────────────────────────

class Survey(Base):
    __tablename__ = "surveys"

    id: Mapped[uuid.UUID] = uuid_pk()
    admin_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"), nullable=False)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    questions: Mapped[dict] = mapped_column(JSON, nullable=False, default=list)
    targeting_rules: Mapped[dict | None] = mapped_column(JSON)
    estimated_time_seconds: Mapped[int | None] = mapped_column(Integer)
    quality_score: Mapped[float | None] = mapped_column(Float)
    predicted_completion_rate: Mapped[float | None] = mapped_column(Float)
    version: Mapped[int] = mapped_column(Integer, default=1)
    parent_survey_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("surveys.id"))
    status: Mapped[str] = mapped_column(String(20), default="draft")  # draft|active|closed
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    launched_at: Mapped[datetime | None] = mapped_column(DateTime)
    closed_at: Mapped[datetime | None] = mapped_column(DateTime)

    admin: Mapped["User"] = relationship(back_populates="surveys")
    responses: Mapped[list["Response"]] = relationship(back_populates="survey")
    insights: Mapped[list["SurveyInsight"]] = relationship(back_populates="survey")
    events: Mapped[list["SurveyEvent"]] = relationship(back_populates="survey")


# ─── Response ─────────────────────────────────────────────────────────────────

class Response(Base):
    __tablename__ = "responses"

    id: Mapped[uuid.UUID] = uuid_pk()
    survey_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("surveys.id"), nullable=False)
    doctor_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"), nullable=False)
    answers: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    is_complete: Mapped[bool] = mapped_column(Boolean, default=False)
    time_spent_seconds: Mapped[int | None] = mapped_column(Integer)
    device_type: Mapped[str | None] = mapped_column(String(20))
    started_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    completed_at: Mapped[datetime | None] = mapped_column(DateTime)

    survey: Mapped["Survey"] = relationship(back_populates="responses")
    doctor: Mapped["User"] = relationship(back_populates="responses")


# ─── Insight ──────────────────────────────────────────────────────────────────

class SurveyInsight(Base):
    __tablename__ = "survey_insights"

    id: Mapped[uuid.UUID] = uuid_pk()
    survey_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("surveys.id"), nullable=False)
    themes: Mapped[dict | None] = mapped_column(JSON)
    executive_summary: Mapped[str | None] = mapped_column(Text)
    action_items: Mapped[dict | None] = mapped_column(JSON)
    sentiment_breakdown: Mapped[dict | None] = mapped_column(JSON)
    completion_rate: Mapped[float | None] = mapped_column(Float)
    generated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    survey: Mapped["Survey"] = relationship(back_populates="insights")


# ─── Analytics Events ─────────────────────────────────────────────────────────

class SurveyEvent(Base):
    __tablename__ = "survey_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    survey_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("surveys.id"), nullable=False)
    doctor_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id"))
    event_type: Mapped[str] = mapped_column(String(60), nullable=False)
    question_id: Mapped[str | None] = mapped_column(String(60))
    survey_metadata: Mapped[dict | None] = mapped_column(JSON)
    timestamp: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    survey: Mapped["Survey"] = relationship(back_populates="events")


# ─── Agent Interaction Log ────────────────────────────────────────────────────

class AgentInteractionLog(Base):
    __tablename__ = "agent_interaction_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    agent_type: Mapped[str] = mapped_column(String(20), nullable=False)
    user_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id"))
    input_context: Mapped[dict | None] = mapped_column(JSON)
    output_response: Mapped[dict | None] = mapped_column(JSON)
    tokens_used: Mapped[int | None] = mapped_column(Integer)
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    timestamp: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
