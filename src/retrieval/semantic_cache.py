"""Semantic cache backed by Redis for near-duplicate query deduplication.

Stores query embeddings as Redis vectors and uses cosine similarity to
determine whether an incoming query is semantically close enough to a
cached entry to serve the cached response, avoiding redundant retrieval
and LLM calls.
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import UTC, datetime
from typing import Any

import numpy as np
import structlog
from pydantic import BaseModel
from redis.asyncio import Redis
from redis.commands.search.field import TagField, TextField, VectorField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class CacheResult(BaseModel):
    """A cache hit returned when a semantically similar query is found.

    Attributes:
        response: The cached response text.
        original_query: The query string that produced the cached response.
        similarity_score: Cosine similarity between the incoming query
            and the cached query embedding.
        cached_at: Timestamp when the entry was originally cached.
    """

    response: str
    original_query: str
    similarity_score: float
    cached_at: datetime


class CacheStats(BaseModel):
    """Aggregate statistics about the semantic cache.

    Attributes:
        total_entries: Number of cached entries.
        hit_rate: Ratio of cache hits to total lookups (0.0-1.0).
        avg_latency: Average cache-lookup latency in milliseconds.
    """

    total_entries: int = 0
    hit_rate: float = 0.0
    avg_latency: float = 0.0


# ---------------------------------------------------------------------------
# Semantic cache
# ---------------------------------------------------------------------------

_INDEX_NAME = "omnicomply:cache:idx"
_KEY_PREFIX = "omnicomply:cache:entry:"
_STATS_KEY = "omnicomply:cache:stats"
_VECTOR_DIM = 1024  # BGE-M3 dimensionality


class SemanticCache:
    """Redis-backed semantic cache with vector similarity lookup.

    Embeddings are stored as Redis hash fields and indexed with
    ``RediSearch`` for efficient approximate-nearest-neighbour queries.
    Metadata is stored alongside via Redis JSON-compatible hash fields.

    Args:
        redis_url: Redis connection URL (e.g. ``"redis://localhost:6379/0"``).
        similarity_threshold: Minimum cosine similarity to treat as a hit.
        vector_dim: Embedding dimensionality.
        ttl_seconds: Default entry time-to-live. ``None`` for no expiry.
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        similarity_threshold: float = 0.92,
        vector_dim: int = _VECTOR_DIM,
        ttl_seconds: int | None = 3600,
    ) -> None:
        self._redis_url = redis_url
        self._similarity_threshold = similarity_threshold
        self._vector_dim = vector_dim
        self._ttl_seconds = ttl_seconds
        self._redis: Redis | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Open the Redis connection and ensure the vector index exists."""
        logger.info("semantic_cache_connect", url=self._redis_url)
        self._redis = Redis.from_url(self._redis_url, decode_responses=False)
        await self._ensure_index()
        logger.info("semantic_cache_connected")

    async def close(self) -> None:
        """Close the Redis connection."""
        if self._redis:
            await self._redis.aclose()
            logger.info("semantic_cache_closed")

    async def _ensure_index(self) -> None:
        """Create the RediSearch vector index if it does not already exist."""
        assert self._redis is not None
        try:
            await self._redis.ft(_INDEX_NAME).info()
            logger.debug("semantic_cache_index_exists")
        except Exception:
            # Index does not exist — create it.
            schema = (
                TextField("$.query", as_name="query"),
                TextField("$.response", as_name="response"),
                TagField("$.status", as_name="status"),
                VectorField(
                    "$.embedding",
                    "FLAT",
                    {
                        "TYPE": "FLOAT32",
                        "DIM": self._vector_dim,
                        "DISTANCE_METRIC": "COSINE",
                    },
                    as_name="embedding",
                ),
            )
            definition = IndexDefinition(
                prefix=[_KEY_PREFIX],
                index_type=IndexType.JSON,
            )
            await self._redis.ft(_INDEX_NAME).create_index(
                schema,
                definition=definition,
            )
            logger.info("semantic_cache_index_created")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get(
        self, query: str, query_embedding: list[float] | None = None
    ) -> CacheResult | None:
        """Look up a semantically similar cached query.

        Args:
            query: The incoming query string (used for logging; the
                actual similarity check uses the embedding).
            query_embedding: Pre-computed query embedding.  If ``None``,
                the caller must supply it; the cache itself does not
                embed queries.

        Returns:
            A :class:`CacheResult` if a match above the threshold is
            found, otherwise ``None``.
        """
        if query_embedding is None:
            logger.warning("semantic_cache_get_no_embedding")
            return None

        self._assert_connected()
        start = time.perf_counter()

        try:
            query_blob = np.array(query_embedding, dtype=np.float32).tobytes()

            redis_query = (
                Query("*=>[KNN 1 @embedding $vec AS score]")
                .sort_by("score")
                .return_fields("query", "response", "cached_at", "score")
                .dialect(2)
            )

            results = await self._redis.ft(_INDEX_NAME).search(  # type: ignore[union-attr]
                redis_query,
                query_params={"vec": query_blob},
            )

            latency_ms = (time.perf_counter() - start) * 1000
            await self._update_stats(hit=False, latency_ms=latency_ms)

            if not results.docs:
                logger.debug("semantic_cache_miss", query=query[:80])
                return None

            doc = results.docs[0]
            # RediSearch COSINE distance: 0 = identical, 2 = opposite.
            # Convert to similarity: sim = 1 - distance.
            distance = float(doc.score)  # type: ignore[arg-type]
            similarity = 1.0 - distance

            if similarity < self._similarity_threshold:
                logger.debug(
                    "semantic_cache_miss_below_threshold",
                    similarity=round(similarity, 4),
                    threshold=self._similarity_threshold,
                )
                return None

            # Cache hit!
            await self._update_stats(hit=True, latency_ms=latency_ms)

            cached_at_str = (
                doc.cached_at if isinstance(doc.cached_at, str) else doc.cached_at.decode()
            )  # type: ignore[union-attr]

            logger.info(
                "semantic_cache_hit",
                similarity=round(similarity, 4),
                original_query=str(doc.query)[:80] if doc.query else "",  # type: ignore[union-attr]
            )

            return CacheResult(
                response=doc.response.decode()
                if isinstance(doc.response, bytes)
                else str(doc.response),  # type: ignore[union-attr]
                original_query=doc.query.decode()
                if isinstance(doc.query, bytes)
                else str(doc.query),  # type: ignore[union-attr]
                similarity_score=similarity,
                cached_at=datetime.fromisoformat(cached_at_str),
            )

        except Exception:
            logger.exception("semantic_cache_get_error")
            return None

    async def set(
        self,
        query: str,
        query_embedding: list[float],
        response: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Store a query-response pair in the cache.

        Args:
            query: The original query text.
            query_embedding: The query's embedding vector.
            response: The response to cache.
            metadata: Optional metadata to store alongside the entry.
        """
        self._assert_connected()

        entry_id = str(uuid.uuid4())
        key = f"{_KEY_PREFIX}{entry_id}"
        now = datetime.now(UTC).isoformat()

        embedding_list = [float(v) for v in query_embedding]

        payload = {
            "query": query,
            "response": response,
            "embedding": embedding_list,
            "metadata": json.dumps(metadata or {}),
            "cached_at": now,
            "status": "active",
        }

        await self._redis.json().set(key, "$", payload)  # type: ignore[union-attr]

        if self._ttl_seconds:
            await self._redis.expire(key, self._ttl_seconds)  # type: ignore[union-attr]

        logger.info("semantic_cache_set", entry_id=entry_id, query=query[:80])

    async def invalidate(self, pattern: str) -> int:
        """Remove cache entries whose keys match a glob pattern.

        Args:
            pattern: Redis key glob (e.g. ``"*regulation*"``).

        Returns:
            Number of entries removed.
        """
        self._assert_connected()

        full_pattern = f"{_KEY_PREFIX}{pattern}"
        count = 0

        async for key in self._redis.scan_iter(match=full_pattern):  # type: ignore[union-attr]
            await self._redis.delete(key)  # type: ignore[union-attr]
            count += 1

        logger.info("semantic_cache_invalidated", pattern=pattern, removed=count)
        return count

    async def get_stats(self) -> CacheStats:
        """Return aggregate cache statistics.

        Returns:
            A :class:`CacheStats` summary.
        """
        self._assert_connected()

        raw = await self._redis.hgetall(_STATS_KEY)  # type: ignore[union-attr]
        if not raw:
            return CacheStats()

        total_lookups = int(raw.get(b"total_lookups", 0))
        total_hits = int(raw.get(b"total_hits", 0))
        total_latency = float(raw.get(b"total_latency_ms", 0.0))

        # Count live entries.
        total_entries = 0
        async for _ in self._redis.scan_iter(match=f"{_KEY_PREFIX}*"):  # type: ignore[union-attr]
            total_entries += 1

        hit_rate = total_hits / total_lookups if total_lookups > 0 else 0.0
        avg_latency = total_latency / total_lookups if total_lookups > 0 else 0.0

        return CacheStats(
            total_entries=total_entries,
            hit_rate=round(hit_rate, 4),
            avg_latency=round(avg_latency, 2),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _update_stats(self, hit: bool, latency_ms: float) -> None:
        """Increment internal hit/miss and latency counters."""
        assert self._redis is not None
        pipe = self._redis.pipeline()
        pipe.hincrby(_STATS_KEY, "total_lookups", 1)
        if hit:
            pipe.hincrby(_STATS_KEY, "total_hits", 1)
        pipe.hincrbyfloat(_STATS_KEY, "total_latency_ms", latency_ms)
        await pipe.execute()

    def _assert_connected(self) -> None:
        """Raise if the Redis client has not been initialised."""
        if self._redis is None:
            raise RuntimeError("SemanticCache is not connected. Call connect() first.")
