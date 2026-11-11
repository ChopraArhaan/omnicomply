"""Embedding service wrapping sentence-transformers for dense vector generation.

Provides a unified interface for generating document and query embeddings
using the BGE-M3 model (or any compatible sentence-transformer model).
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import structlog
from sentence_transformers import SentenceTransformer

logger = structlog.get_logger(__name__)


class EmbeddingService:
    """High-performance embedding service using sentence-transformers.

    Wraps a SentenceTransformer model to produce normalized dense vectors
    suitable for cosine-similarity search. Supports batched encoding for
    throughput and adds a query prefix for asymmetric retrieval models.

    Attributes:
        model_name: HuggingFace model identifier.
        device: Torch device string (e.g. ``"cpu"``, ``"cuda"``).
        batch_size: Number of texts to encode per batch.
        dimension: Output embedding dimensionality (read after model load).
    """

    # BGE-family models expect this prefix for queries in asymmetric retrieval.
    _QUERY_PREFIX: str = "Represent this sentence for searching relevant passages: "

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: str | None = None,
        batch_size: int = 32,
    ) -> None:
        """Initialise the embedding service and warm up the model.

        Args:
            model_name: Name or path of the sentence-transformer model.
            device: Torch device. ``None`` lets the library auto-detect.
            batch_size: Default batch size for :meth:`embed_texts`.
        """
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size

        logger.info(
            "loading_embedding_model",
            model_name=model_name,
            device=device,
            batch_size=batch_size,
        )

        start = time.perf_counter()
        self._model = SentenceTransformer(model_name, device=device)
        load_time = time.perf_counter() - start

        self.dimension: int = self._model.get_sentence_embedding_dimension()  # type: ignore[assignment]

        logger.info(
            "embedding_model_loaded",
            dimension=self.dimension,
            load_seconds=round(load_time, 2),
        )

        self._warmup()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Encode a batch of texts into normalised embedding vectors.

        Args:
            texts: Plain-text strings to embed. May be document chunks,
                passages, or any textual content.

        Returns:
            A list of embedding vectors (one per input text), each a list
            of floats with length equal to :attr:`dimension`.

        Raises:
            ValueError: If *texts* is empty.
        """
        if not texts:
            raise ValueError("texts must be a non-empty list of strings")

        logger.debug("embed_texts_start", count=len(texts))
        start = time.perf_counter()

        embeddings: np.ndarray = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        elapsed = time.perf_counter() - start
        logger.debug(
            "embed_texts_done",
            count=len(texts),
            elapsed_seconds=round(elapsed, 3),
        )

        return self._to_list(embeddings)

    def embed_query(self, query: str) -> list[float]:
        """Encode a single search query with the retrieval prefix.

        Asymmetric retrieval models (BGE, E5, etc.) benefit from a
        task-specific prefix prepended to the query.  This method adds
        the prefix automatically.

        Args:
            query: The user's search query string.

        Returns:
            A normalised embedding vector as a list of floats.

        Raises:
            ValueError: If *query* is empty or blank.
        """
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string")

        prefixed = f"{self._QUERY_PREFIX}{query}"

        logger.debug("embed_query", query_length=len(query))
        start = time.perf_counter()

        embedding: np.ndarray = self._model.encode(
            prefixed,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        elapsed = time.perf_counter() - start
        logger.debug("embed_query_done", elapsed_seconds=round(elapsed, 4))

        return embedding.tolist()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _warmup(self) -> None:
        """Run a dummy embedding to JIT-compile / warm caches."""
        logger.debug("embedding_warmup_start")
        start = time.perf_counter()
        _ = self._model.encode(
            "warmup",
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        elapsed = time.perf_counter() - start
        logger.info("embedding_warmup_done", elapsed_seconds=round(elapsed, 3))

    @staticmethod
    def _to_list(arr: np.ndarray) -> list[list[float]]:
        """Convert a numpy array of embeddings to a nested Python list."""
        return arr.tolist()  # type: ignore[return-value]
