"""
Local embeddings helper using sentence-transformers.
Uses all-MiniLM-L6-v2 (384 dims, runs locally, no API key needed).

Drop-in replacement for the previous OpenAI embeddings module —
embed_text() and embed_batch() signatures are identical.
"""
from __future__ import annotations

import asyncio
import logging
from functools import lru_cache

from sentence_transformers import SentenceTransformer

# Use stdlib logging directly — sentence_transformers internally calls
# logger.name which custom PrintLogger objects don't implement.
logger = logging.getLogger(__name__)

# Embedding dimension for all-MiniLM-L6-v2
# Update your vector store index to match if you're migrating from OpenAI (1536 → 384)
EMBEDDING_DIM = 384
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    """
    Load the model once and cache it for the lifetime of the process.
    Thread-safe due to lru_cache + GIL for model loading.
    """
    logger.info(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    logger.info(f"Embedding model loaded (dim={EMBEDDING_DIM})")
    return model


async def embed_text(text: str) -> list[float]:
    """
    Embed a single text string.
    Runs the CPU-bound encoding in a thread pool to keep the event loop free.
    """
    cleaned = text.replace("\n", " ").strip()
    loop = asyncio.get_event_loop()
    embedding = await loop.run_in_executor(
        None,  # uses default ThreadPoolExecutor
        lambda: _get_model().encode(cleaned, normalize_embeddings=True).tolist()
    )
    return embedding


async def embed_batch(texts: list[str]) -> list[list[float]]:
    """
    Embed multiple texts in a single encoding call.
    Order of returned embeddings matches order of input texts.
    """
    cleaned = [t.replace("\n", " ").strip() for t in texts]
    loop = asyncio.get_event_loop()
    embeddings = await loop.run_in_executor(
        None,
        lambda: _get_model().encode(
            cleaned,
            normalize_embeddings=True,
            batch_size=64,        # tune based on available RAM
            show_progress_bar=False,
        ).tolist()
    )
    return embeddings