"""Retrieval infrastructure for OmniComply.

This package provides the complete RAG retrieval pipeline including
dense embeddings, vector storage, knowledge-graph traversal, BM25 sparse
search, cross-encoder reranking, semantic caching, and a unified hybrid
retriever that fuses all sources with Reciprocal Rank Fusion.
"""

from src.retrieval.embeddings import EmbeddingService
from src.retrieval.hybrid_retriever import (
    BM25Index,
    HybridRetriever,
    RetrievalResult,
    RetrievalSource,
)
from src.retrieval.knowledge_graph import (
    Entity,
    GraphEdge,
    GraphNode,
    GraphResult,
    KnowledgeGraphStore,
)
from src.retrieval.reranker import Reranker
from src.retrieval.semantic_cache import (
    CacheResult,
    CacheStats,
    SemanticCache,
)
from src.retrieval.vector_store import (
    DocumentChunk,
    PgVectorStore,
    SearchResult,
)

__all__ = [
    # Embeddings
    "EmbeddingService",
    # Vector store
    "PgVectorStore",
    "DocumentChunk",
    "SearchResult",
    # Knowledge graph
    "KnowledgeGraphStore",
    "Entity",
    "GraphEdge",
    "GraphNode",
    "GraphResult",
    # Reranker
    "Reranker",
    # Semantic cache
    "SemanticCache",
    "CacheResult",
    "CacheStats",
    # Hybrid retriever
    "HybridRetriever",
    "BM25Index",
    "RetrievalResult",
    "RetrievalSource",
]
