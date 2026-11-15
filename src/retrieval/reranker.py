"""Cross-encoder reranker for improving retrieval precision.

Takes an initial set of candidate search results and re-scores them using
a cross-encoder model that jointly encodes the query-document pair, producing
a more accurate relevance estimate than bi-encoder similarity alone.
"""

from __future__ import annotations

import time

import structlog
from sentence_transformers import CrossEncoder

from src.retrieval.vector_store import SearchResult

logger = structlog.get_logger(__name__)


class Reranker:
    """Cross-encoder reranker using ``sentence-transformers.CrossEncoder``.

    Scores each ``(query, document)`` pair with a cross-attention model and
    returns the top-k results sorted by relevance.

    Args:
        model_name: HuggingFace model identifier for the cross-encoder.
        device: Torch device string (``None`` for auto-detect).
        batch_size: Batch size for cross-encoder inference.
        max_length: Maximum token length per query-document pair.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: str | None = None,
        batch_size: int = 32,
        max_length: int = 512,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length

        logger.info(
            "loading_reranker_model",
            model_name=model_name,
            device=device,
        )

        start = time.perf_counter()
        self._model = CrossEncoder(
            model_name,
            device=device,
            max_length=max_length,
        )
        load_time = time.perf_counter() - start

        logger.info(
            "reranker_model_loaded",
            elapsed_seconds=round(load_time, 2),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rerank(
        self,
        query: str,
        documents: list[SearchResult],
        top_k: int = 5,
    ) -> list[SearchResult]:
        """Score and re-sort search results by cross-encoder relevance.

        Args:
            query: The user's search query.
            documents: Candidate results from the initial retrieval stage.
            top_k: Number of top results to return after reranking.

        Returns:
            The *top_k* most relevant :class:`SearchResult` objects,
            sorted by descending cross-encoder score.  The ``score``
            field is updated to reflect the reranker's score.

        Raises:
            ValueError: If *query* is empty.
        """
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string")

        if not documents:
            return []

        logger.debug(
            "rerank_start",
            query_length=len(query),
            candidates=len(documents),
            top_k=top_k,
        )

        start = time.perf_counter()

        # Build (query, document) pairs for the cross-encoder.
        pairs: list[list[str]] = [[query, doc.content] for doc in documents]

        # Score in batches for efficiency.
        scores = self._batch_predict(pairs)

        # Pair each document with its reranker score and sort descending.
        scored: list[tuple[float, SearchResult]] = list(zip(scores, documents, strict=False))
        scored.sort(key=lambda x: x[0], reverse=True)

        # Take top_k and update the score field.
        results: list[SearchResult] = []
        for score, doc in scored[:top_k]:
            results.append(
                SearchResult(
                    chunk_id=doc.chunk_id,
                    content=doc.content,
                    score=float(score),
                    metadata=doc.metadata,
                )
            )

        elapsed = time.perf_counter() - start
        logger.debug(
            "rerank_done",
            candidates=len(documents),
            returned=len(results),
            elapsed_seconds=round(elapsed, 3),
        )

        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _batch_predict(self, pairs: list[list[str]]) -> list[float]:
        """Run cross-encoder prediction in batches.

        Args:
            pairs: List of ``[query, document]`` string pairs.

        Returns:
            List of relevance scores (one per pair).
        """
        all_scores: list[float] = []
        total = len(pairs)

        for batch_start in range(0, total, self.batch_size):
            batch_end = min(batch_start + self.batch_size, total)
            batch = pairs[batch_start:batch_end]

            batch_scores = self._model.predict(
                batch,
                show_progress_bar=False,
            )

            # CrossEncoder.predict returns ndarray or list of floats.
            if hasattr(batch_scores, "tolist"):
                all_scores.extend(batch_scores.tolist())
            else:
                all_scores.extend(batch_scores)

        return all_scores
