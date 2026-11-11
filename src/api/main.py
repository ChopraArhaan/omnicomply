"""
OmniComply FastAPI application entry-point.

Provides:
* Lifespan handler that initialises shared services (embedding service,
  vector store, knowledge graph).
* CORS middleware and structured-logging middleware.
* ``/health`` and ``/ready`` probes.
* Includes all v1 routers.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config.settings import Settings

# ---------------------------------------------------------------------------
# Structured logging bootstrap
# ---------------------------------------------------------------------------

structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(0),
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Shared application state (populated during lifespan)
# ---------------------------------------------------------------------------

_state: dict[str, Any] = {}


def get_state() -> dict[str, Any]:
    """Return the application-wide shared state dict."""
    return _state


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Initialise and tear-down shared services."""
    settings = Settings()
    logger.info("lifespan.start", log_level=settings.log_level)

    # ------ Embedding service ------
    try:
        from src.retrieval.embeddings import EmbeddingService

        embedding_service = EmbeddingService(
            model_name=settings.embedding_model_name,
            device=settings.embedding_device,
            batch_size=settings.embedding_batch_size,
        )
        _state["embedding_service"] = embedding_service
        logger.info("lifespan.embedding_service.ready")
    except Exception:
        logger.warning("lifespan.embedding_service.skipped", exc_info=True)

    # ------ Vector store ------
    try:
        from src.retrieval.vector_store import PgVectorStore

        vector_store = PgVectorStore(database_url=settings.database_url)
        _state["vector_store"] = vector_store
        logger.info("lifespan.vector_store.ready")
    except Exception:
        logger.warning("lifespan.vector_store.skipped", exc_info=True)

    # ------ Knowledge graph ------
    try:
        from src.retrieval.knowledge_graph import KnowledgeGraphStore

        graph = KnowledgeGraphStore(
            uri=settings.neo4j_uri,
            user=settings.neo4j_user,
            password=settings.neo4j_password,
        )
        _state["knowledge_graph"] = graph
        logger.info("lifespan.knowledge_graph.ready")
    except Exception:
        logger.warning("lifespan.knowledge_graph.skipped", exc_info=True)

    yield  # ---- application is running ----

    # ------ Tear-down ------
    logger.info("lifespan.shutdown")
    _state.clear()


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app() -> FastAPI:
    """Build and return the configured FastAPI instance."""
    settings = Settings()

    app = FastAPI(
        title="OmniComply API",
        version="0.1.0",
        description="AI-powered regulatory-compliance platform.",
        lifespan=lifespan,
    )

    # ---- CORS ----
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api_cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ---- Custom middleware ----
    from src.api.middleware.logging_middleware import RequestLoggingMiddleware
    from src.api.middleware.cost_tracking import CostTrackingMiddleware

    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(CostTrackingMiddleware, redis_url=settings.redis_url)

    # ---- Routers ----
    from src.api.routes.documents import router as documents_router
    from src.api.routes.query import router as query_router
    from src.api.routes.analysis import router as analysis_router

    app.include_router(documents_router)
    app.include_router(query_router)
    app.include_router(analysis_router)

    # ---- Health & readiness probes ----

    @app.get("/health", tags=["probes"])
    async def health() -> dict[str, str]:
        """Liveness probe — always returns OK."""
        return {"status": "ok"}

    @app.get("/ready", tags=["probes"])
    async def ready() -> dict[str, Any]:
        """Readiness probe — verifies DB and Redis connectivity."""
        checks: dict[str, str] = {}

        # -- Postgres --
        try:
            import asyncpg

            conn = await asyncio.wait_for(
                asyncpg.connect(dsn=settings.database_url.replace("+asyncpg", "")),
                timeout=3.0,
            )
            await conn.execute("SELECT 1")
            await conn.close()
            checks["postgres"] = "ok"
        except Exception as exc:
            checks["postgres"] = f"error: {exc}"

        # -- Redis --
        try:
            import redis.asyncio as aioredis

            r = aioredis.from_url(settings.redis_url, decode_responses=True)
            await asyncio.wait_for(r.ping(), timeout=3.0)
            await r.aclose()
            checks["redis"] = "ok"
        except Exception as exc:
            checks["redis"] = f"error: {exc}"

        all_ok = all(v == "ok" for v in checks.values())
        return {"ready": all_ok, "checks": checks}

    return app


# Convenience module-level app for ``uvicorn src.api.main:app``
app = create_app()
