"""
Pinecone vector store wrapper.
Manages two indexes:
  - survey-guidelines  : best-practice docs for Design Agent
  - survey-templates   : past high-performing surveys for inspiration
"""
from __future__ import annotations

from pinecone import Pinecone, ServerlessSpec

from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

_pc: Pinecone | None = None


def get_pinecone() -> Pinecone:
    global _pc
    if _pc is None:
        _pc = Pinecone(api_key=settings.PINECONE_API_KEY)
    return _pc


def get_index(index_name: str):
    pc = get_pinecone()
    return pc.Index(index_name)


def ensure_indexes() -> None:
    """Create Pinecone indexes if they don't exist (run at startup)."""
    pc = get_pinecone()
    existing = [idx["name"] for idx in pc.list_indexes()]

    for index_name in [
        settings.PINECONE_INDEX_GUIDELINES,
        settings.PINECONE_INDEX_TEMPLATES,
    ]:
        if index_name not in existing:
            pc.create_index(
                name=index_name,
                dimension=1536,          # text-embedding-3-small
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region=settings.PINECONE_ENVIRONMENT,
                ),
            )
            logger.info("pinecone.index_created", index=index_name)
        else:
            logger.info("pinecone.index_exists", index=index_name)


async def upsert_vectors(
    index_name: str,
    vectors: list[dict],  # [{"id": str, "values": list[float], "metadata": dict}]
) -> None:
    idx = get_index(index_name)
    idx.upsert(vectors=vectors)
    logger.info("pinecone.upsert", index=index_name, count=len(vectors))


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
    return [
        {
            "id": match["id"],
            "score": match["score"],
            "metadata": match.get("metadata", {}),
        }
        for match in response.get("matches", [])
    ]
