"""
Centralised application settings for OmniComply.

All values are loaded from environment variables (or an ``.env`` file) via
Pydantic Settings.  Groups are separated by prefix for clarity but live in a
single flat ``Settings`` object so that ``Settings()`` is the only call sites
ever need.
"""

from __future__ import annotations

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide configuration backed by environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM Configuration ────────────────────────────────────────────────
    llm_provider: str = Field(
        default="openai",
        description="LLM provider backend (openai, anthropic, vllm, local).",
    )
    llm_model_name: str = Field(
        default="gpt-4o",
        description="Model identifier passed to the provider.",
    )
    llm_api_key: str = Field(
        default="",
        description="API key for the chosen LLM provider.",
    )
    llm_base_url: str = Field(
        default="https://api.openai.com/v1",
        description="Base URL for the LLM API endpoint.",
    )
    llm_temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Sampling temperature for LLM responses.",
    )
    llm_max_tokens: int = Field(
        default=4096,
        gt=0,
        description="Maximum tokens in an LLM response.",
    )

    # ── Embedding Configuration ──────────────────────────────────────────
    embedding_model_name: str = Field(
        default="BAAI/bge-large-en-v1.5",
        description="Sentence-transformer model for embeddings.",
    )
    embedding_device: str = Field(
        default="cpu",
        description="Torch device for the embedding model (cpu, cuda, mps).",
    )
    embedding_batch_size: int = Field(
        default=32,
        gt=0,
        description="Batch size when encoding embeddings.",
    )

    # ── PostgreSQL ───────────────────────────────────────────────────────
    postgres_host: str = Field(default="localhost")
    postgres_port: int = Field(default=5432)
    postgres_db: str = Field(default="omnicomply")
    postgres_user: str = Field(default="omnicomply")
    postgres_password: str = Field(default="changeme")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def database_url(self) -> str:
        """Construct a full async-compatible DSN from individual fields."""
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    # ── Redis ────────────────────────────────────────────────────────────
    redis_host: str = Field(default="localhost")
    redis_port: int = Field(default=6379)
    redis_db: int = Field(default=0)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def redis_url(self) -> str:
        """Construct a Redis connection URL."""
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"

    # ── Neo4j ────────────────────────────────────────────────────────────
    neo4j_uri: str = Field(default="bolt://localhost:7687")
    neo4j_user: str = Field(default="neo4j")
    neo4j_password: str = Field(default="changeme")

    # ── Celery ───────────────────────────────────────────────────────────
    celery_broker_url: str = Field(default="redis://localhost:6379/1")
    celery_result_backend: str = Field(default="redis://localhost:6379/2")

    # ── API Server ───────────────────────────────────────────────────────
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    api_reload: bool = Field(default=False)
    api_workers: int = Field(default=4, gt=0)
    api_cors_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:3000"],
        description="Allowed CORS origins.",
    )

    # ── External APIs ────────────────────────────────────────────────────
    federal_register_url: str = Field(
        default="https://www.federalregister.gov/api/v1",
        description="Federal Register public API base URL.",
    )
    sec_edgar_url: str = Field(
        default="https://efts.sec.gov/LATEST",
        description="SEC EDGAR full-text search API base URL.",
    )
    fda_url: str = Field(
        default="https://api.fda.gov",
        description="FDA openFDA API base URL.",
    )
    congress_api_key: str = Field(
        default="",
        description="API key for the Congress.gov API.",
    )

    # ── Observability ────────────────────────────────────────────────────
    langsmith_api_key: str = Field(default="")
    langsmith_project: str = Field(default="omnicomply")
    log_level: str = Field(
        default="INFO",
        description="Root log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).",
    )

    # ── Eval Thresholds ──────────────────────────────────────────────────
    eval_faithfulness_threshold: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Minimum RAGAS faithfulness score to pass CI.",
    )
    eval_relevancy_threshold: float = Field(
        default=0.80,
        ge=0.0,
        le=1.0,
        description="Minimum RAGAS answer-relevancy score to pass CI.",
    )
    eval_context_precision_threshold: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="Minimum RAGAS context-precision score to pass CI.",
    )

    # ── RAG Configuration ────────────────────────────────────────────────
    rag_chunk_size: int = Field(
        default=512,
        gt=0,
        description="Target token count per chunk.",
    )
    rag_chunk_overlap: int = Field(
        default=64,
        ge=0,
        description="Token overlap between consecutive chunks.",
    )
    rag_top_k_retrieval: int = Field(
        default=20,
        gt=0,
        description="Number of chunks retrieved before re-ranking.",
    )
    rag_top_k_rerank: int = Field(
        default=5,
        gt=0,
        description="Number of chunks kept after re-ranking.",
    )
    rag_semantic_cache_threshold: float = Field(
        default=0.92,
        ge=0.0,
        le=1.0,
        description="Cosine-similarity threshold for semantic cache hits.",
    )
