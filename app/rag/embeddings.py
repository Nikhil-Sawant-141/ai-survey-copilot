"""
OpenAI embeddings helper.
Uses text-embedding-3-small (1536 dims, low cost).
"""
from __future__ import annotations

from openai import AsyncOpenAI

from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

_client: AsyncOpenAI | None = None


def get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    return _client


async def embed_text(text: str) -> list[float]:
    """Embed a single text string."""
    client = get_client()
    response = await client.embeddings.create(
        model=settings.OPENAI_EMBEDDING_MODEL,
        input=text.replace("\n", " "),
    )
    return response.data[0].embedding


async def embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed multiple texts in a single API call (up to 2048 inputs)."""
    client = get_client()
    response = await client.embeddings.create(
        model=settings.OPENAI_EMBEDDING_MODEL,
        input=[t.replace("\n", " ") for t in texts],
    )
    return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]
