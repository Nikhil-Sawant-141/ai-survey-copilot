"""initial tables

Revision ID: 001_initial
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSONB


def upgrade() -> None:
    # Users
    op.create_table(
        "users",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("email", sa.String(255), unique=True, nullable=False),
        sa.Column("hashed_password", sa.String(255), nullable=False),
        sa.Column("role", sa.String(20), nullable=False),
        sa.Column("specialty", sa.String(100)),
        sa.Column("years_experience", sa.Integer),
        sa.Column("is_active", sa.Boolean, default=True),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
    )

    # Surveys
    op.create_table(
        "surveys",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("admin_id", UUID(as_uuid=True), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("title", sa.String(255), nullable=False),
        sa.Column("description", sa.Text),
        sa.Column("questions", JSONB, nullable=False),
        sa.Column("targeting_rules", JSONB),
        sa.Column("estimated_time_seconds", sa.Integer),
        sa.Column("quality_score", sa.Float),
        sa.Column("predicted_completion_rate", sa.Float),
        sa.Column("version", sa.Integer, default=1),
        sa.Column("parent_survey_id", UUID(as_uuid=True)),
        sa.Column("status", sa.String(20), default="draft"),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
        sa.Column("launched_at", sa.DateTime),
        sa.Column("closed_at", sa.DateTime),
    )

    # Responses
    op.create_table(
        "responses",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("survey_id", UUID(as_uuid=True), sa.ForeignKey("surveys.id"), nullable=False),
        sa.Column("doctor_id", UUID(as_uuid=True), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("answers", JSONB, nullable=False),
        sa.Column("is_complete", sa.Boolean, default=False),
        sa.Column("time_spent_seconds", sa.Integer),
        sa.Column("device_type", sa.String(20)),
        sa.Column("started_at", sa.DateTime, server_default=sa.func.now()),
        sa.Column("completed_at", sa.DateTime),
        sa.UniqueConstraint("survey_id", "doctor_id", name="uq_response_per_doctor"),
    )

    # Insights
    op.create_table(
        "survey_insights",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("survey_id", UUID(as_uuid=True), sa.ForeignKey("surveys.id"), nullable=False),
        sa.Column("themes", JSONB),
        sa.Column("executive_summary", sa.Text),
        sa.Column("action_items", JSONB),
        sa.Column("sentiment_breakdown", JSONB),
        sa.Column("completion_rate", sa.Float),
        sa.Column("generated_at", sa.DateTime, server_default=sa.func.now()),
    )

    # Events
    op.create_table(
        "survey_events",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("survey_id", UUID(as_uuid=True), sa.ForeignKey("surveys.id"), nullable=False),
        sa.Column("doctor_id", UUID(as_uuid=True), sa.ForeignKey("users.id")),
        sa.Column("event_type", sa.String(60), nullable=False),
        sa.Column("question_id", sa.String(60)),
        sa.Column("metadata", JSONB),
        sa.Column("timestamp", sa.DateTime, server_default=sa.func.now()),
    )

    # Agent interaction logs
    op.create_table(
        "agent_interaction_logs",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("agent_type", sa.String(20), nullable=False),
        sa.Column("user_id", UUID(as_uuid=True), sa.ForeignKey("users.id")),
        sa.Column("input_context", JSONB),
        sa.Column("output_response", JSONB),
        sa.Column("tokens_used", sa.Integer),
        sa.Column("latency_ms", sa.Integer),
        sa.Column("timestamp", sa.DateTime, server_default=sa.func.now()),
    )

    # Indexes for common queries
    op.create_index("ix_surveys_admin_id", "surveys", ["admin_id"])
    op.create_index("ix_surveys_status", "surveys", ["status"])
    op.create_index("ix_responses_survey_id", "responses", ["survey_id"])
    op.create_index("ix_survey_events_survey_id", "survey_events", ["survey_id"])
    op.create_index("ix_survey_events_type", "survey_events", ["event_type"])


def downgrade() -> None:
    for table in [
        "agent_interaction_logs", "survey_events", "survey_insights",
        "responses", "surveys", "users"
    ]:
        op.drop_table(table)
