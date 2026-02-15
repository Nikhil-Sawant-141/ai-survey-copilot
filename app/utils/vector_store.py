"""
Pinecone vector store helper.

Handles index creation automatically if the index doesn't exist yet,
so you never hit a 404 NotFoundException on first run.

Indexes:
  - survey-guidelines   : best practices, compliance rules (RAG for Design Agent)
  - survey-templates    : historical high-performing surveys (RAG for Design Agent)
  - question-clarifications : past Q&A pairs (RAG for Attempt Agent)
"""
from __future__ import annotations

import logging
import time
from enum import Enum

from pinecone import Pinecone, ServerlessSpec

from app.rag.embeddings import EMBEDDING_DIM, embed_text, embed_batch

logger = logging.getLogger(__name__)

# ── Index names ────────────────────────────────────────────────────────────────

class IndexName(str, Enum):
    GUIDELINES      = "survey-guidelines"
    TEMPLATES       = "survey-templates"
    CLARIFICATIONS  = "question-clarifications"


# ── Pinecone client singleton ──────────────────────────────────────────────────

_pc: Pinecone | None = None


def get_pinecone(api_key: str) -> Pinecone:
    global _pc
    if _pc is None:
        _pc = Pinecone(api_key=api_key)
    return _pc


# ── Index bootstrap ────────────────────────────────────────────────────────────

def ensure_index(
    pc: Pinecone,
    index_name: str,
    dimension: int = EMBEDDING_DIM,
    metric: str = "cosine",
    cloud: str = "aws",
    region: str = "us-east-1",
    wait_ready: bool = True,
) -> None:
    """
    Create a Pinecone serverless index if it doesn't already exist.
    Blocks until the index is ready when wait_ready=True.
    """
    existing = {idx.name for idx in pc.list_indexes()}

    if index_name in existing:
        logger.info(f"Pinecone index '{index_name}' already exists — skipping creation.")
        return

    logger.info(f"Creating Pinecone index '{index_name}' (dim={dimension}, metric={metric}) ...")
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric=metric,
        spec=ServerlessSpec(cloud=cloud, region=region),
    )

    if wait_ready:
        _wait_until_ready(pc, index_name)

    logger.info(f"Pinecone index '{index_name}' is ready.")


def ensure_all_indexes(pc: Pinecone) -> None:
    """Create all required indexes in one call. Run this at app startup."""
    for index in IndexName:
        ensure_index(pc, index.value)


def _wait_until_ready(pc: Pinecone, index_name: str, timeout: int = 120) -> None:
    """Poll until the index status is ready or timeout is reached."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        description = pc.describe_index(index_name)
        if description.status.get("ready"):
            return
        logger.debug(f"Waiting for index '{index_name}' to become ready ...")
        time.sleep(3)
    raise TimeoutError(f"Pinecone index '{index_name}' was not ready within {timeout}s.")


# ── Vector store class ─────────────────────────────────────────────────────────

class VectorStore:
    """
    Thin async wrapper around a single Pinecone index.
    Handles upsert and similarity search.
    """

    def __init__(self, pc: Pinecone, index_name: IndexName) -> None:
        self.index_name = index_name.value
        self._index = pc.Index(self.index_name)

    # ── Write ──────────────────────────────────────────────────────────────────

    async def upsert_text(
        self,
        doc_id: str,
        text: str,
        metadata: dict | None = None,
    ) -> None:
        """Embed a single text and upsert into the index."""
        vector = await embed_text(text)
        self._index.upsert(vectors=[{
            "id": doc_id,
            "values": vector,
            "metadata": metadata or {},
        }])
        logger.debug(f"Upserted doc '{doc_id}' into '{self.index_name}'.")

    async def upsert_batch(
        self,
        documents: list[dict],  # each: {"id": str, "text": str, "metadata": dict}
        batch_size: int = 100,
    ) -> None:
        """
        Embed and upsert a batch of documents.
        documents = [{"id": "...", "text": "...", "metadata": {...}}, ...]
        """
        texts = [d["text"] for d in documents]
        vectors = await embed_batch(texts)

        records = [
            {
                "id": doc["id"],
                "values": vec,
                "metadata": doc.get("metadata", {}),
            }
            for doc, vec in zip(documents, vectors)
        ]

        # Pinecone recommends batches ≤ 100 vectors
        for i in range(0, len(records), batch_size):
            chunk = records[i : i + batch_size]
            self._index.upsert(vectors=chunk)
            logger.debug(f"Upserted batch [{i}:{i+len(chunk)}] into '{self.index_name}'.")

    # ── Read ───────────────────────────────────────────────────────────────────

    async def search(
        self,
        query: str,
        top_k: int = 5,
        filter: dict | None = None,
    ) -> list[dict]:
        """
        Semantic search. Returns list of:
          {"id": str, "score": float, "metadata": dict}
        sorted by descending similarity score.
        """
        query_vector = await embed_text(query)
        response = self._index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True,
            filter=filter,
        )
        return [
            {
                "id": match.id,
                "score": match.score,
                "metadata": match.metadata,
            }
            for match in response.matches
        ]

    async def search_and_format(
        self,
        query: str,
        top_k: int = 5,
        filter: dict | None = None,
    ) -> str:
        """
        Convenience method: search and return a formatted string
        ready to inject directly into an LLM prompt as context.
        """
        results = await self.search(query, top_k=top_k, filter=filter)
        if not results:
            return "No relevant guidelines found."

        lines = []
        for i, r in enumerate(results, 1):
            title   = r["metadata"].get("title", f"Document {i}")
            content = r["metadata"].get("content", "")
            score   = r["score"]
            lines.append(f"[{i}] {title} (relevance: {score:.2f})\n{content}")

        return "\n\n".join(lines)


# ── Factory helpers ────────────────────────────────────────────────────────────

def get_guidelines_store(pc: Pinecone) -> VectorStore:
    return VectorStore(pc, IndexName.GUIDELINES)

def get_templates_store(pc: Pinecone) -> VectorStore:
    return VectorStore(pc, IndexName.TEMPLATES)

def get_clarifications_store(pc: Pinecone) -> VectorStore:
    return VectorStore(pc, IndexName.CLARIFICATIONS)
