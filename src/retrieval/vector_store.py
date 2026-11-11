"""PostgreSQL + pgvector based vector store for dense chunk retrieval.

Provides async CRUD operations on document chunks with their embedding
vectors, including cosine-similarity search with optional metadata
filtering and connection pooling.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any

import asyncpg
import structlog
from pgvector.asyncpg import register_vector
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class DocumentChunk(BaseModel):
    """A chunk of a source document ready for vector insertion.

    Attributes:
        chunk_id: Globally unique chunk identifier.
        document_id: Parent document identifier.
        content: Plain-text content of the chunk.
        metadata: Arbitrary key-value metadata (source, page, etc.).
        token_count: Approximate token count of the content.
    """

    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    document_id: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    token_count: int = 0


class SearchResult(BaseModel):
    """A single search hit returned by the vector store.

    Attributes:
        chunk_id: Identifier of the matching chunk.
        content: Text content of the chunk.
        score: Cosine similarity score (higher is more similar).
        metadata: Chunk metadata carried through from ingestion.
    """

    chunk_id: str
    content: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------

# BGE-M3 outputs 1024-dimensional embeddings.
_VECTOR_DIMENSION: int = 1024


class PgVectorStore:
    """Async vector store backed by PostgreSQL with the pgvector extension.

    Uses ``asyncpg`` connection pooling for efficient concurrent access
    and supports cosine-similarity search with dynamic JSONB metadata
    filters.

    Args:
        dsn: PostgreSQL connection string
            (e.g. ``"postgresql://user:pass@localhost:5432/omnicomply"``).
        pool_min: Minimum connections kept open in the pool.
        pool_max: Maximum connections in the pool.
        dimension: Expected embedding vector dimensionality.
    """

    def __init__(
        self,
        dsn: str,
        pool_min: int = 2,
        pool_max: int = 10,
        dimension: int = _VECTOR_DIMENSION,
    ) -> None:
        self._dsn = dsn
        self._pool_min = pool_min
        self._pool_max = pool_max
        self._dimension = dimension
        self._pool: asyncpg.Pool | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Create the asyncpg connection pool and register the vector type."""
        logger.info("pgvector_connect", dsn=self._dsn)
        self._pool = await asyncpg.create_pool(
            dsn=self._dsn,
            min_size=self._pool_min,
            max_size=self._pool_max,
            init=self._init_connection,
        )
        logger.info("pgvector_pool_created", min=self._pool_min, max=self._pool_max)

    async def close(self) -> None:
        """Gracefully close all pooled connections."""
        if self._pool:
            await self._pool.close()
            logger.info("pgvector_pool_closed")

    @staticmethod
    async def _init_connection(conn: asyncpg.Connection) -> None:
        """Per-connection initialiser: register the ``vector`` codec."""
        await register_vector(conn)

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    async def insert_chunks(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
    ) -> None:
        """Insert document chunks with their embedding vectors.

        Args:
            chunks: Chunk data (content, metadata, etc.).
            embeddings: Corresponding embedding vectors. Must be the same
                length as *chunks*.

        Raises:
            ValueError: If *chunks* and *embeddings* have different lengths.
            RuntimeError: If the pool is not connected.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks ({len(chunks)}) and embeddings ({len(embeddings)}) "
                "must have the same length"
            )
        self._assert_connected()

        logger.info("insert_chunks_start", count=len(chunks))

        query = """
            INSERT INTO chunks (id, document_id, content, embedding, metadata, token_count, created_at)
            VALUES ($1, $2, $3, $4::vector, $5::jsonb, $6, $7)
            ON CONFLICT (id) DO UPDATE SET
                content    = EXCLUDED.content,
                embedding  = EXCLUDED.embedding,
                metadata   = EXCLUDED.metadata,
                token_count = EXCLUDED.token_count
        """

        now = datetime.now(timezone.utc)
        records: list[tuple[Any, ...]] = []
        for chunk, emb in zip(chunks, embeddings):
            records.append((
                uuid.UUID(chunk.chunk_id),
                uuid.UUID(chunk.document_id),
                chunk.content,
                _format_vector(emb),
                json.dumps(chunk.metadata),
                chunk.token_count,
                now,
            ))

        async with self._pool.acquire() as conn:  # type: ignore[union-attr]
            await conn.executemany(query, records)

        logger.info("insert_chunks_done", count=len(chunks))

    async def delete_document(self, document_id: str) -> None:
        """Delete all chunks belonging to a document.

        Args:
            document_id: The parent document identifier.
        """
        self._assert_connected()
        logger.info("delete_document", document_id=document_id)

        async with self._pool.acquire() as conn:  # type: ignore[union-attr]
            result = await conn.execute(
                "DELETE FROM chunks WHERE document_id = $1",
                uuid.UUID(document_id),
            )

        logger.info("delete_document_done", document_id=document_id, result=result)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Find the most similar chunks via cosine similarity.

        Args:
            query_embedding: The query vector (same dimensionality as
                stored vectors).
            top_k: Maximum number of results to return.
            filters: Optional metadata key-value pairs. Each pair is
                ANDed as a JSONB containment check
                (``metadata @> '{"key": "value"}'``).

        Returns:
            A list of :class:`SearchResult` sorted by descending
            similarity score.
        """
        self._assert_connected()

        # Build the WHERE clause for optional metadata filters.
        where_clauses: list[str] = []
        params: list[Any] = [_format_vector(query_embedding), top_k]
        param_idx = 3  # $1 = embedding, $2 = top_k

        if filters:
            for key, value in filters.items():
                filter_json = json.dumps({key: value})
                where_clauses.append(f"metadata @> ${param_idx}::jsonb")
                params.append(filter_json)
                param_idx += 1

        where_sql = ""
        if where_clauses:
            where_sql = "WHERE " + " AND ".join(where_clauses)

        query = f"""
            SELECT
                id,
                content,
                1 - (embedding <=> $1::vector) AS score,
                metadata
            FROM chunks
            {where_sql}
            ORDER BY embedding <=> $1::vector
            LIMIT $2
        """

        logger.debug("vector_search", top_k=top_k, filters=filters)

        async with self._pool.acquire() as conn:  # type: ignore[union-attr]
            rows = await conn.fetch(query, *params)

        results = [
            SearchResult(
                chunk_id=str(row["id"]),
                content=row["content"],
                score=float(row["score"]),
                metadata=json.loads(row["metadata"]) if isinstance(row["metadata"], str) else dict(row["metadata"]),
            )
            for row in rows
        ]

        logger.debug("vector_search_done", results=len(results))
        return results

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    async def get_collection_stats(self) -> dict[str, Any]:
        """Return aggregate statistics about the chunks collection.

        Returns:
            Dictionary with ``total_chunks``, ``total_documents``,
            ``avg_token_count``, and ``storage_size_bytes``.
        """
        self._assert_connected()

        query = """
            SELECT
                COUNT(*)                              AS total_chunks,
                COUNT(DISTINCT document_id)           AS total_documents,
                COALESCE(AVG(token_count), 0)         AS avg_token_count,
                COALESCE(pg_total_relation_size('chunks'), 0) AS storage_size_bytes
            FROM chunks
        """

        async with self._pool.acquire() as conn:  # type: ignore[union-attr]
            row = await conn.fetchrow(query)

        stats = {
            "total_chunks": row["total_chunks"],  # type: ignore[index]
            "total_documents": row["total_documents"],  # type: ignore[index]
            "avg_token_count": float(row["avg_token_count"]),  # type: ignore[index]
            "storage_size_bytes": row["storage_size_bytes"],  # type: ignore[index]
        }
        logger.info("collection_stats", **stats)
        return stats

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _assert_connected(self) -> None:
        """Raise if the connection pool has not been initialised."""
        if self._pool is None:
            raise RuntimeError(
                "PgVectorStore is not connected. Call connect() first."
            )


def _format_vector(vec: list[float]) -> str:
    """Format a Python list of floats into the pgvector literal string."""
    return "[" + ",".join(str(v) for v in vec) + "]"
