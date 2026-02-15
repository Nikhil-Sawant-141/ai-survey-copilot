# Survey Agent — AI-Powered Survey Engagement Platform

End-to-end Python system with three specialized AI agents that improve survey
quality (admin side) and completion rates (doctor side).

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│   FastAPI (app/main.py)                                      │
│   Auth | Surveys | Responses | Agents | Insights             │
└───────────────────────┬──────────────────────────────────────┘
                        │
┌───────────────────────▼──────────────────────────────────────┐
│   Agent Orchestrator (app/agents/orchestrator.py)            │
│   Rate limiting · Safety checks · Audit logging              │
├───────────────────────┬──────────────────────────────────────┤
│  Design Agent         │  Attempt Agent  │  Insight Agent     │
│  Admin survey quality │  Doctor UX aid  │  Post-survey NLP   │
└───────────────────────┴──────────────────────────────────────┘
                        │
┌───────────────────────▼──────────────────────────────────────┐
│   OpenAI GPT-4o  |  Pinecone  |  PostgreSQL  |  Redis        │
│   LLM + Tools    |  RAG       |  Storage     |  Cache/Queue  │
└──────────────────────────────────────────────────────────────┘
```

## Project Structure

```
survey_agent/
├── app/
│   ├── main.py                  # FastAPI app + lifespan startup
│   ├── config.py                # Pydantic settings
│   ├── database.py              # Async SQLAlchemy setup
│   ├── redis_client.py          # Redis helpers
│   ├── models/__init__.py       # SQLAlchemy ORM models
│   ├── schemas/__init__.py      # Pydantic request/response schemas
│   ├── agents/
│   │   ├── design_agent.py      # Admin: quality check, variants, suggestions
│   │   ├── attempt_agent.py     # Doctor: clarify, progress, completion
│   │   ├── insight_agent.py     # Async: themes, sentiment, action items
│   │   └── orchestrator.py      # Routes tasks, enforces rate limits
│   ├── rag/
│   │   ├── embeddings.py        # OpenAI text-embedding-3-small
│   │   ├── pinecone_client.py   # Vector store operations
│   │   └── knowledge_base.py   # Seed guidelines + retrieve for prompts
│   ├── routers/
│   │   ├── auth.py              # Register, login, JWT
│   │   ├── surveys.py           # Survey CRUD + launch/close
│   │   ├── responses.py         # Doctor submit responses
│   │   ├── agents.py            # All AI agent REST endpoints
│   │   └── insights.py          # View insight reports
│   ├── tasks/
│   │   └── celery_app.py        # Async: generate_insights, reminders, close surveys
│   ├── safety/
│   │   └── moderator.py         # PHI detection, medical advice blocking
│   └── utils/
│       └── logger.py            # Structured logging (structlog)
├── cli/
│   └── demo.py                  # Interactive CLI demo (no server needed)
├── alembic/                     # DB migrations
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── .env.example
```

## Quick Start

### Option A: CLI Demo (fastest — no DB/Redis needed)

```bash
# 1. Install dependencies
pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Add OPENAI_API_KEY to .env

# 3. Run full demo
python -m cli.demo full

# Or individual agents:
python -m cli.demo design       # Design Agent: bias check + variants
python -m cli.demo attempt      # Attempt Agent: clarification + progress
python -m cli.demo insights     # Insight Agent: theme + recommendations
```

### Option B: Full Stack with Docker

```bash
# 1. Configure
cp .env.example .env
# Add OPENAI_API_KEY, PINECONE_API_KEY to .env

# 2. Start everything
docker-compose up

# 3. API available at:
#    http://localhost:8000/docs   ← Interactive Swagger UI
#    http://localhost:5555        ← Celery Flower monitoring

# 4. Run DB migrations
docker-compose exec api alembic upgrade head
```

### Option C: Local Development

```bash
# Prerequisites: PostgreSQL + Redis running locally

pip install -r requirements.txt
cp .env.example .env        # Edit with your keys + local DB URL

# Run migrations
alembic upgrade head

# Start API
uvicorn app.main:app --reload

# Start Celery worker (separate terminal)
celery -A app.tasks.celery_app worker --loglevel=info -Q insights,reminders

# Start Celery Beat for scheduled tasks (separate terminal)
celery -A app.tasks.celery_app beat --loglevel=info
```

## API Reference

### Authentication
```http
POST /auth/register    # Create admin or doctor user
POST /auth/login       # Get JWT token
GET  /auth/me          # Current user
```

### Admin: Survey Management
```http
POST   /surveys              # Create survey (status=draft)
GET    /surveys              # List your surveys
GET    /surveys/{id}         # Get survey
PATCH  /surveys/{id}         # Update draft survey
POST   /surveys/{id}/launch  # Launch (requires quality_score)
POST   /surveys/{id}/close   # Close + trigger insights
DELETE /surveys/{id}         # Delete draft
```

### Admin: AI Agents
```http
POST /agents/quality-check          # Bias detection + quality score
POST /agents/improve-question       # Improve a single question
POST /agents/generate-variants      # Generate A/B test variants
POST /agents/suggest-questions      # Suggest questions from goal
```

### Doctor: Survey Taking
```http
POST /responses                     # Submit answers (partial or complete)
POST /agents/clarify                # Explain a confusing question
GET  /agents/progress               # Get progress + motivation message
POST /agents/completion-summary     # Post-completion thank-you
POST /agents/save-progress          # Save partial answers (Redis)
GET  /agents/restore/{session_id}   # Restore in-progress session
```

### Admin: Insights
```http
GET  /insights/{survey_id}          # View insight report
POST /insights/{survey_id}/trigger  # Manually trigger analysis
GET  /surveys/{id}/responses        # List all responses
```

## Agent Details

### Design Agent (`/agents/quality-check`)
```json
POST /agents/quality-check
{
  "survey_title": "EHR Satisfaction Survey",
  "questions": [...],
  "specialty": "Cardiology",
  "survey_id": "uuid-optional"
}
```
Returns: quality score, bias flags with fixes, completion rate prediction, timing.

### Attempt Agent (`/agents/clarify`)
```json
POST /agents/clarify
{
  "session_id": "sess_abc123",
  "survey_id": "uuid",
  "question_id": "q5",
  "doctor_context": {"specialty": "Cardiology", "years_experience": 8}
}
```
Returns: plain-English clarification with examples. Never changes question meaning.

### Insight Agent (async, via Celery)
Triggered automatically when survey closes. Also available via:
```http
POST /insights/{survey_id}/trigger
```
Returns: themes, sentiment, executive summary, prioritized action items.

## Safety & Compliance

- **PHI Prevention**: Questions scanned for 15+ PHI keyword/regex patterns before saving
- **Response Sanitization**: Open-text answers automatically redact SSNs, phones, emails
- **Medical Advice Blocking**: Agent outputs checked with 7 regex patterns before returning
- **Audit Logging**: Every agent call logged with input context, output summary, latency
- **Rate Limiting**: Per-user limits via Redis (10 clarifications/survey, 100 suggestions/hour)
- **HIPAA Notes**: No patient data collected. All responses anonymized at rest.

## Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key (GPT-4o + embeddings) |
| `PINECONE_API_KEY` | Pinecone API key (vector store) |
| `DATABASE_URL` | Async PostgreSQL URL |
| `REDIS_URL` | Redis URL (sessions + cache) |
| `SECRET_KEY` | JWT signing secret |

## Tech Stack

| Layer | Technology |
|-------|-----------|
| API Framework | FastAPI + Uvicorn |
| LLM | OpenAI GPT-4o (function calling) |
| Embeddings | text-embedding-3-small |
| Vector Store | Pinecone (RAG) |
| Database | PostgreSQL + SQLAlchemy async |
| Cache/Sessions | Redis |
| Task Queue | Celery + Redis Broker |
| Migrations | Alembic |
| Safety | Custom moderator + Presidio-ready |
| Logging | structlog |
| CLI | Typer + Rich |
