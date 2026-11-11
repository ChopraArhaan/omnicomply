"""Hybrid retriever combining dense, sparse, and graph retrieval with RRF fusion.

Orchestrates parallel retrieval from a PgVector dense store, a BM25 sparse
index, and a Neo4j knowledge graph, fuses results with Reciprocal Rank
Fusion, applies cross-encoder reranking, and caches responses semantically.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from enum import Enum
from typing import Any

import structlog
from pydantic import BaseModel, Field
from rank_bm25 import BM25Okapi

from src.retrieval.embeddings import EmbeddingService
from src.retrieval.knowledge_graph import KnowledgeGraphStore
from src.retrieval.reranker import Reranker
from src.retrieval.semantic_cache import SemanticCache
from src.retrieval.vector_store import PgVectorStore, SearchResult

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class RetrievalSource(str, Enum):
    """Enumeration of retrieval sources."""

    DENSE = "dense"
    SPARSE = "sparse"
    GRAPH = "graph"


class RetrievalResult(BaseModel):
    """A single result from the hybrid retriever.

    Attributes:
        chunk_id: Unique chunk identifier.
        content: Text content of the chunk.
        score: Final fused/reranked score.
        source: Which retrieval pathway produced this result.
        metadata: Arbitrary metadata carried from ingestion.
    """

    chunk_id: str
    content: str
    score: float
    source: str  # "dense" | "sparse" | "graph"
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# BM25 corpus wrapper
# ---------------------------------------------------------------------------


class BM25Index:
    """In-memory BM25 sparse index over a pre-loaded corpus.

    Maintains a parallel list of :class:`SearchResult` objects so that
    BM25 scores can be mapped back to the original chunk metadata.
    """

    def __init__(self) -> None:
        self._corpus_tokens: list[list[str]] = []
        self._documents: list[SearchResult] = []
        self._bm25: BM25Okapi | None = None

    @property
    def is_ready(self) -> bool:
        """Whether the index has been built."""
        return self._bm25 is not None and len(self._documents) > 0

    def build(self, documents: list[SearchResult]) -> None:
        """Tokenise and index a list of documents.

        Args:
            documents: The corpus to index. Each document's ``content``
                field is whitespace-tokenised.
        """
        self._documents = documents
        self._corpus_tokens = [doc.content.lower().split() for doc in documents]
        self._bm25 = BM25Okapi(self._corpus_tokens)
        logger.info("bm25_index_built", corpus_size=len(documents))

    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        """Score the corpus against *query* and return the top-k results.

        Args:
            query: The search query.
            top_k: Maximum number of results.

        Returns:
            A list of :class:`SearchResult` sorted by BM25 score
            descending.
        """
        if not self.is_ready:
            logger.warning("bm25_index_not_ready")
            return []

        assert self._bm25 is not None
        tokens = query.lower().split()
        scores = self._bm25.get_scores(tokens)

        # Pair scores with documents and sort.
        scored = sorted(
            zip(scores, self._documents),
            key=lambda x: x[0],
            reverse=True,
        )

        results: list[SearchResult] = []
        for score, doc in scored[:top_k]:
            if score <= 0:
                break
            results.append(
                SearchResult(
                    chunk_id=doc.chunk_id,
                    content=doc.content,
                    score=float(score),
                    metadata=doc.metadata,
                )
            )

        return results


# ---------------------------------------------------------------------------
# Hybrid retriever
# ---------------------------------------------------------------------------


class HybridRetriever:
    """Orchestrates multi-source retrieval with fusion, reranking, and caching.

    Pipeline:
        1. Check the semantic cache for a near-duplicate query.
        2. Run dense (PgVector), sparse (BM25), and graph (Neo4j) retrievals
           in parallel via ``asyncio.gather``.
        3. Fuse results with Reciprocal Rank Fusion (RRF).
        4. Rerank the fused top candidates with a cross-encoder.
        5. Cache the final response.

    Args:
        embedding_service: For encoding queries into dense vectors.
        vector_store: PgVector dense retrieval backend.
        knowledge_graph: Neo4j knowledge-graph backend.
        reranker: Cross-encoder reranker.
        semantic_cache: Redis-backed semantic cache.
        rrf_k: RRF constant (default 60, per the original paper).
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: PgVectorStore,
        knowledge_graph: KnowledgeGraphStore,
        reranker: Reranker,
        semantic_cache: SemanticCache,
        rrf_k: int = 60,
    ) -> None:
        self._embeddings = embedding_service
        self._vector_store = vector_store
        self._kg = knowledge_graph
        self._reranker = reranker
        self._cache = semantic_cache
        self._rrf_k = rrf_k

        # In-memory BM25 index — must be built before sparse retrieval works.
        self._bm25_index = BM25Index()

    # ------------------------------------------------------------------
    # Corpus management
    # ------------------------------------------------------------------

    def build_bm25_index(self, documents: list[SearchResult]) -> None:
        """Build (or rebuild) the in-memory BM25 index.

        Should be called after ingestion or on startup with the full
        corpus of chunks.

        Args:
            documents: All document chunks as :class:`SearchResult` objects.
        """
        self._bm25_index.build(documents)

    # ------------------------------------------------------------------
    # Main retrieval API
    # ------------------------------------------------------------------

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        strategy: str = "hybrid",
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """Run the full hybrid retrieval pipeline.

        Args:
            query: Natural-language search query.
            top_k: Number of final results to return.
            strategy: One of ``"hybrid"``, ``"dense"``, ``"sparse"``,
                or ``"graph"``.
            filters: Optional metadata filters for the dense retriever.

        Returns:
            Ranked list of :class:`RetrievalResult` objects.
        """
        logger.info(
            "retrieve_start",
            query=query[:120],
            top_k=top_k,
            strategy=strategy,
        )
        start = time.perf_counter()

        # 1. Embed the query (needed for dense search and cache check).
        query_embedding = self._embeddings.embed_query(query)

        # 2. Check semantic cache.
        cached = await self._cache.get(query, query_embedding=query_embedding)
        if cached is not None:
            logger.info("retrieve_cache_hit", similarity=cached.similarity_score)
            try:
                cached_results = json.loads(cached.response)
                return [RetrievalResult(**r) for r in cached_results]
            except (json.JSONDecodeError, Exception):
                logger.warning("retrieve_cache_parse_error")
                # Fall through to live retrieval.

        # 3. Run selected retrieval strategies in parallel.
        results = await self._dispatch(
            query=query,
            query_embedding=query_embedding,
            top_k=top_k,
            strategy=strategy,
            filters=filters,
        )

        # 4. Rerank the fused candidates (take more than top_k so the
        #    reranker has room to re-sort).
        rerank_candidates = [
            SearchResult(
                chunk_id=r.chunk_id,
                content=r.content,
                score=r.score,
                metadata=r.metadata,
            )
            for r in results[: top_k * 3]
        ]

        if rerank_candidates:
            reranked = self._reranker.rerank(query, rerank_candidates, top_k=top_k)
            # Map back to RetrievalResult, preserving original source.
            source_map = {r.chunk_id: r.source for r in results}
            results = [
                RetrievalResult(
                    chunk_id=sr.chunk_id,
                    content=sr.content,
                    score=sr.score,
                    source=source_map.get(sr.chunk_id, "dense"),
                    metadata=sr.metadata,
                )
                for sr in reranked
            ]

        # 5. Cache the result.
        try:
            serialised = json.dumps([r.model_dump() for r in results])
            await self._cache.set(
                query=query,
                query_embedding=query_embedding,
                response=serialised,
                metadata={"strategy": strategy, "top_k": top_k},
            )
        except Exception:
            logger.warning("retrieve_cache_set_error", exc_info=True)

        elapsed = time.perf_counter() - start
        logger.info(
            "retrieve_done",
            results=len(results),
            elapsed_seconds=round(elapsed, 3),
        )

        return results

    # ------------------------------------------------------------------
    # Strategy dispatch
    # ------------------------------------------------------------------

    async def _dispatch(
        self,
        query: str,
        query_embedding: list[float],
        top_k: int,
        strategy: str,
        filters: dict[str, Any] | None,
    ) -> list[RetrievalResult]:
        """Run one or more retrieval strategies and fuse results."""
        if strategy == "dense":
            dense = await self._dense_search(query_embedding, top_k, filters)
            return dense

        if strategy == "sparse":
            sparse = self._sparse_search(query, top_k)
            return sparse

        if strategy == "graph":
            graph = await self._graph_search(query, top_k)
            return graph

        # Default: hybrid — run all three in parallel.
        dense_task = self._dense_search(query_embedding, top_k, filters)
        sparse_task = asyncio.coroutine(lambda: self._sparse_search(query, top_k))() if False else asyncio.get_event_loop().run_in_executor(None, self._sparse_search, query, top_k)
        graph_task = self._graph_search(query, top_k)

        dense_results, sparse_results, graph_results = await asyncio.gather(
            dense_task,
            sparse_task,
            graph_task,
            return_exceptions=True,
        )

        # Gracefully handle individual failures.
        if isinstance(dense_results, BaseException):
            logger.error("dense_retrieval_failed", error=str(dense_results))
            dense_results = []
        if isinstance(sparse_results, BaseException):
            logger.error("sparse_retrieval_failed", error=str(sparse_results))
            sparse_results = []
        if isinstance(graph_results, BaseException):
            logger.error("graph_retrieval_failed", error=str(graph_results))
            graph_results = []

        # Fuse with RRF.
        fused = self._reciprocal_rank_fusion(
            [dense_results, sparse_results, graph_results]  # type: ignore[list-item]
        )

        return fused

    # ------------------------------------------------------------------
    # Individual retrieval methods
    # ------------------------------------------------------------------

    async def _dense_search(
        self,
        query_embedding: list[float],
        top_k: int,
        filters: dict[str, Any] | None,
    ) -> list[RetrievalResult]:
        """Run dense vector search via PgVector."""
        results = await self._vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            filters=filters,
        )
        return [
            RetrievalResult(
                chunk_id=r.chunk_id,
                content=r.content,
                score=r.score,
                source=RetrievalSource.DENSE.value,
                metadata=r.metadata,
            )
            for r in results
        ]

    def _sparse_search(self, query: str, top_k: int) -> list[RetrievalResult]:
        """Run BM25 sparse search over the in-memory index."""
        results = self._bm25_index.search(query, top_k=top_k)
        return [
            RetrievalResult(
                chunk_id=r.chunk_id,
                content=r.content,
                score=r.score,
                source=RetrievalSource.SPARSE.value,
                metadata=r.metadata,
            )
            for r in results
        ]

    async def _graph_search(self, query: str, top_k: int) -> list[RetrievalResult]:
        """Extract key terms from the query and traverse the knowledge graph."""
        # Simple heuristic: use the longest non-stopword tokens as entity
        # candidates.  A production system would use an NER model here.
        _STOPWORDS = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "shall", "can",
            "of", "in", "to", "for", "with", "on", "at", "from", "by",
            "about", "as", "into", "through", "during", "before", "after",
            "and", "but", "or", "nor", "not", "so", "yet", "both",
            "what", "which", "who", "whom", "this", "that", "these",
            "those", "it", "its", "how", "when", "where", "why",
        }

        tokens = [
            t for t in query.split()
            if t.lower() not in _STOPWORDS and len(t) > 2
        ]
        tokens.sort(key=len, reverse=True)
        entity_candidates = tokens[:3]  # top 3 longest tokens

        all_results: list[RetrievalResult] = []
        seen_ids: set[str] = set()

        for entity in entity_candidates:
            try:
                graph_results = await self._kg.query_related(entity, hops=2)
            except Exception:
                logger.warning("graph_search_entity_failed", entity=entity)
                continue

            for gr in graph_results:
                for node in gr.nodes:
                    node_id = node.id
                    if node_id in seen_ids:
                        continue
                    seen_ids.add(node_id)

                    # Construct a pseudo-content string from node properties.
                    content_parts = [
                        f"{k}: {v}"
                        for k, v in node.properties.items()
                        if isinstance(v, (str, int, float)) and k not in ("element_id", "id")
                    ]
                    content = "; ".join(content_parts) if content_parts else node_id

                    all_results.append(
                        RetrievalResult(
                            chunk_id=node_id,
                            content=content,
                            score=1.0,  # graph results are unscored; RRF handles ranking
                            source=RetrievalSource.GRAPH.value,
                            metadata=node.properties,
                        )
                    )

                    if len(all_results) >= top_k:
                        break
                if len(all_results) >= top_k:
                    break
            if len(all_results) >= top_k:
                break

        return all_results[:top_k]

    # ------------------------------------------------------------------
    # Reciprocal Rank Fusion
    # ------------------------------------------------------------------

    def _reciprocal_rank_fusion(
        self,
        result_lists: list[list[RetrievalResult]],
    ) -> list[RetrievalResult]:
        """Fuse multiple ranked lists using Reciprocal Rank Fusion (RRF).

        RRF score for a document *d* across *n* rankers:

            RRF(d) = Σ  1 / (k + rank_i(d))

        where *k* is a constant (default 60) and ``rank_i(d)`` is the
        1-based rank of *d* in ranker *i* (∞ if absent).

        Args:
            result_lists: List of ranked result lists from different
                retrieval sources.

        Returns:
            A single fused list sorted by RRF score descending.
        """
        k = self._rrf_k
        scores: dict[str, float] = {}
        docs: dict[str, RetrievalResult] = {}

        for results in result_lists:
            for rank, result in enumerate(results, start=1):
                rrf_score = 1.0 / (k + rank)
                scores[result.chunk_id] = scores.get(result.chunk_id, 0.0) + rrf_score
                # Keep the first occurrence's metadata and source.
                if result.chunk_id not in docs:
                    docs[result.chunk_id] = result

        # Sort by fused RRF score.
        sorted_ids = sorted(scores, key=scores.__getitem__, reverse=True)

        fused: list[RetrievalResult] = []
        for chunk_id in sorted_ids:
            original = docs[chunk_id]
            fused.append(
                RetrievalResult(
                    chunk_id=chunk_id,
                    content=original.content,
                    score=scores[chunk_id],
                    source=original.source,
                    metadata=original.metadata,
                )
            )

        return fused
