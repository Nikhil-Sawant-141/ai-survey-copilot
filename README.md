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
│   Claude Sonnet  |  Pinecone  |  PostgreSQL  |  Redis        │
│   LLM + Tools   |  RAG       |  Storage     |  Cache/Queue  │
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
│   │   ├── embeddings.py        # Local sentence-transformers (all-MiniLM-L6-v2)
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
│       └── logger.py            # Logging
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
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Add ANTHROPIC_API_KEY and PINECONE_API_KEY to .env

# 3. Run full demo
python -m cli.demo full

# Or individual agents:
python -m cli.demo design       # Design Agent: bias check + variants
python -m cli.demo attempt      # Attempt Agent: clarification + progress
python -m cli.demo insights     # Insight Agent: themes + recommendations
python -m cli.demo quality      # Quick bias check on sample survey
```

### Option B: Full Stack with Docker

```bash
# 1. Configure
cp .env.example .env
# Add ANTHROPIC_API_KEY, PINECONE_API_KEY to .env

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

# Install PostgreSQL
sudo apt install postgresql postgresql-contrib -y
sudo systemctl start postgresql
sudo systemctl enable postgresql

# Create database
sudo -u postgres psql -c "CREATE USER survey_user WITH PASSWORD 'survey_pass';"
sudo -u postgres psql -c "CREATE DATABASE survey_db OWNER survey_user;"

# Install Redis
sudo apt install redis-server -y
sudo systemctl start redis
sudo systemctl enable redis

# Install Python dependencies
pip install -r requirements.txt

# Configure
cp .env.example .env   # Edit with your keys + local DB URL

# Run migrations
alembic upgrade head

# Start API
uvicorn app.main:app --reload

# Start Celery worker (separate terminal)
celery -A app.tasks.celery_app worker --loglevel=info -Q insights,reminders

# Start Celery Beat for scheduled tasks (separate terminal)
celery -A app.tasks.celery_app beat --loglevel=info
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `ANTHROPIC_API_KEY` | ✅ | Anthropic API key (Claude Sonnet) |
| `ANTHROPIC_MODEL` | ✅ | Model string — `claude-sonnet-4-5-20250929` |
| `PINECONE_API_KEY` | ✅ | Pinecone API key (vector store) |
| `PINECONE_ENVIRONMENT` | ✅ | Pinecone region e.g. `us-east-1` |
| `PINECONE_INDEX_GUIDELINES` | ✅ | Index name e.g. `survey-guidelines` |
| `PINECONE_INDEX_TEMPLATES` | ✅ | Index name e.g. `survey-templates` |
| `DATABASE_URL` | ⚠️ Server only | Async PostgreSQL URL |
| `DATABASE_URL_SYNC` | ⚠️ Server only | Sync PostgreSQL URL (Alembic) |
| `REDIS_URL` | ⚠️ Server only | Redis URL (sessions + cache) |
| `SECRET_KEY` | ⚠️ Server only | JWT signing secret |

> **Note:** `DATABASE_URL` and `REDIS_URL` are only required when running the
> full API server. The CLI demo (`python -m cli.demo`) runs without them.

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
POST /agents/suggest-questions      # Suggest questions from survey goal
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
Returns: quality score (0-10), bias flags with fixes, predicted completion rate, estimated time.

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
Responses are cached in Redis for 24h — same question asked by 100 doctors = 1 API call.

### Insight Agent (async, via Celery)
Triggered automatically when survey closes. Also available via:
```http
POST /insights/{survey_id}/trigger
```
Returns: themes, sentiment breakdown, executive summary, prioritized action items.

## Safety & Compliance

- **PHI Prevention**: Questions scanned for PHI keyword/regex patterns before saving
- **Response Sanitization**: Open-text answers automatically redact SSNs, phones, emails
- **Medical Advice Blocking**: All agent outputs checked before returning to users
- **Audit Logging**: Every agent call logged with input context, output summary, latency
- **Rate Limiting**: Per-user limits via Redis (10 clarifications/survey, 100 suggestions/hour)
- **HIPAA Notes**: No patient data collected. All responses anonymized at rest.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| API Framework | FastAPI + Uvicorn |
| LLM | Anthropic Claude Sonnet (`claude-sonnet-4-5-20250929`) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` — runs locally, no API key |
| Vector Store | Pinecone serverless (RAG) — 384-dim cosine index |
| Database | PostgreSQL + SQLAlchemy async |
| Cache / Sessions | Redis |
| Task Queue | Celery + Redis broker |
| Migrations | Alembic |
| Safety | Custom PHI moderator |
| CLI | Typer + Rich |

## Key Design Decisions

**Why Anthropic Claude instead of OpenAI GPT-4o?**
Claude is used for all LLM calls because it has stronger built-in safety for
medical contexts (refuses medical advice by default), better structured output
reliability for complex tool schemas, and a 200k token context window suited
to the Insight Agent's large response sets.

**Why local embeddings instead of OpenAI embeddings?**
`sentence-transformers/all-MiniLM-L6-v2` runs entirely on-device with no API
calls, no quota limits, and no cost. At 384 dimensions it is fast and accurate
enough for survey template and guideline similarity search.

**Why Pinecone serverless?**
Indexes are created automatically at startup (`ensure_indexes()` in lifespan)
so there is no manual setup required. Region must be set to a valid value
(e.g. `us-east-1`) in `.env` — do not include a cloud suffix like `-aws`.

## Common Issues

| Error | Cause | Fix |
|-------|-------|-----|
| `NotFoundException: survey-guidelines` | Pinecone index not created yet | Ensure `ensure_indexes()` runs at startup before any queries |
| `ConnectionRefusedError: localhost:6379` | Redis not running | `sudo systemctl start redis` |
| `ConnectionRefusedError: localhost:5432` | PostgreSQL not running | `sudo systemctl start postgresql` |
| `NotFoundError: model claude-3-5-sonnet` | Deprecated model name | Use `claude-sonnet-4-5-20250929` |