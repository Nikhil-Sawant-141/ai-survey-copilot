"""
Survey Agent API — Main Entry Point
────────────────────────────────────
FastAPI application that wires together:
  - Auth, Survey, Response, Agent, Insight routers
  - Database + Pinecone initialization at startup
  - CORS, error handling, health check
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.database import Base, engine
from app.routers.agents import router as agents_router
from app.routers.auth import router as auth_router
from app.routers.insights import router as insights_router
from app.routers.responses import responses_admin_router, router as responses_router
from app.routers.surveys import router as surveys_router
from app.utils.logger import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle."""
    logger.info("startup.begin", env=settings.APP_ENV)

    # Create DB tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("startup.database_ready")

    # Initialize Pinecone indexes
    try:
        from app.rag.pinecone_client import ensure_indexes
        ensure_indexes()
        logger.info("startup.pinecone_ready")

        # Seed knowledge base (idempotent)
        from app.rag.knowledge_base import seed_knowledge_base
        await seed_knowledge_base()
        logger.info("startup.knowledge_base_seeded")
    except Exception as e:
        logger.warning("startup.pinecone_skip", reason=str(e))

    logger.info("startup.complete")
    yield

    # Shutdown
    await engine.dispose()
    logger.info("shutdown.complete")


app = FastAPI(
    title="Survey Agent API",
    description="AI-powered survey engagement platform for healthcare",
    version="1.0.0",
    lifespan=lifespan,
)

# ─── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],  # Admin + Doctor frontends
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Routers ───────────────────────────────────────────────────────────────────
app.include_router(auth_router)
app.include_router(surveys_router)
app.include_router(responses_router)
app.include_router(responses_admin_router)
app.include_router(agents_router)
app.include_router(insights_router)

# ─── Error handlers ────────────────────────────────────────────────────────────

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.exception_handler(PermissionError)
async def permission_error_handler(request: Request, exc: PermissionError):
    return JSONResponse(status_code=403, content={"detail": str(exc)})


# ─── Health check ──────────────────────────────────────────────────────────────

@app.get("/health", tags=["system"])
async def health():
    return {
        "status": "healthy",
        "env": settings.APP_ENV,
        "version": "1.0.0",
    }


@app.get("/", tags=["system"])
async def root():
    return {
        "name": "Survey Agent API",
        "docs": "/docs",
        "health": "/health",
    }
