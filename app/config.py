from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env.example", extra="ignore")

    # ── App ──────────────────────────────────────────────────────────────────
    APP_ENV: str = "development"
    SECRET_KEY: str = "change-me-in-production"
    DEBUG: bool = True

    # ── Database ─────────────────────────────────────────────────────────────
    DATABASE_URL: str = "postgresql+asyncpg://survey_user:survey_pass@localhost:5432/survey_db"
    DATABASE_URL_SYNC: str = "postgresql://survey_user:survey_pass@localhost:5432/survey_db"

    # ── Redis ─────────────────────────────────────────────────────────────────
    REDIS_URL: str = "redis://localhost:6379/0"
    CELERY_BROKER_URL: str = "redis://localhost:6379/1"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/2"

    # ── OpenAI ───────────────────────────────────────────────────────────────
    OPENAI_API_KEY: str = ""
    OPENAI_LLM_MODEL: str = "gpt-4o"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"

    # ── Pinecone ─────────────────────────────────────────────────────────────
    PINECONE_API_KEY: str = ""
    PINECONE_ENVIRONMENT: str = "us-east-1-aws"
    PINECONE_INDEX_GUIDELINES: str = "survey-guidelines"
    PINECONE_INDEX_TEMPLATES: str = "survey-templates"

    # ── Rate Limits ──────────────────────────────────────────────────────────
    RATE_LIMIT_CLARIFICATION_PER_SURVEY: int = 10
    RATE_LIMIT_AI_SUGGESTIONS_PER_HOUR: int = 100


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
