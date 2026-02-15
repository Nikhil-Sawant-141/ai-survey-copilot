"""
Pinecone vector store wrapper.
Manages two indexes:
  - survey-guidelines  : best-practice docs for Design Agent
  - survey-templates   : past high-performing surveys for inspiration
"""
from __future__ import annotations

import logging
import time

from pinecone import Pinecone, ServerlessSpec

from app.config import settings

# Use stdlib logging — custom PrintLogger breaks third-party libs
logger = logging.getLogger(__name__)

_pc: Pinecone | None = None


def get_pinecone() -> Pinecone:
    global _pc
    if _pc is None:
        _pc = Pinecone(api_key=settings.PINECONE_API_KEY)
    return _pc


def get_index(index_name: str):
    """
    Return a handle to a Pinecone index.
    Auto-creates the index if it doesn't exist yet, so callers
    never hit a 404 regardless of call order at startup.
    """
    pc = get_pinecone()
    existing = set(pc.list_indexes().names())
    if index_name not in existing:
        logger.warning(
            f"Index '{index_name}' not found — creating it now. "
            "Call ensure_indexes() at startup to avoid this delay."
        )
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region=settings.PINECONE_ENVIRONMENT,
            ),
        )
        _wait_until_ready(pc, index_name)
    return pc.Index(index_name)


def ensure_indexes() -> None:
    """Create Pinecone indexes if they don't exist (run at startup)."""
    pc = get_pinecone()

    # FIX 1: list_indexes() returns an object with a .names() method,
    # not a plain list of dicts — the old `idx["name"]` would raise a TypeError
    existing = set(pc.list_indexes().names())

    for index_name in [
        settings.PINECONE_INDEX_GUIDELINES,
        settings.PINECONE_INDEX_TEMPLATES,
    ]:
        if index_name not in existing:
            logger.info(f"Creating Pinecone index '{index_name}' ...")
            pc.create_index(
                name=index_name,
                # FIX 2: dimension must be 384 to match local all-MiniLM-L6-v2
                # embeddings — the old value of 1536 was for OpenAI which is removed
                dimension=384,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region=settings.PINECONE_ENVIRONMENT,
                ),
            )
            # FIX 3: wait until the index is ready before returning —
            # querying too early causes a 404 even after create_index() returns
            _wait_until_ready(pc, index_name)
            logger.info(f"Pinecone index '{index_name}' created and ready.")
        else:
            logger.info(f"Pinecone index '{index_name}' already exists.")


def _wait_until_ready(pc: Pinecone, index_name: str, timeout: int = 120) -> None:
    """Poll until the index is ready or timeout is reached."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = pc.describe_index(index_name).status
        if status.get("ready"):
            return
        logger.debug(f"Waiting for index '{index_name}' to become ready ...")
        time.sleep(3)
    raise TimeoutError(
        f"Pinecone index '{index_name}' was not ready within {timeout}s."
    )


async def upsert_vectors(
    index_name: str,
    vectors: list[dict],  # [{"id": str, "values": list[float], "metadata": dict}]
) -> None:
    idx = get_index(index_name)
    idx.upsert(vectors=vectors)
    logger.info(f"Upserted {len(vectors)} vectors into '{index_name}'.")


async def query_index(
    index_name: str,
    vector: list[float],
    top_k: int = 5,
    filter_dict: dict | None = None,
) -> list[dict]:
    idx = get_index(index_name)
    kwargs: dict = {"vector": vector, "top_k": top_k, "include_metadata": True}
    if filter_dict:
        kwargs["filter"] = filter_dict

    response = idx.query(**kwargs)

    # FIX 4: Pinecone SDK returns Match objects, not plain dicts —
    # access attributes directly instead of using dict subscript/get
    return [
        {
            "id": match.id,
            "score": match.score,
            "metadata": match.metadata or {},
        }
        for match in response.matches
    ]