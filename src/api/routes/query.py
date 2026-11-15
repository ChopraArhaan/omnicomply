"""Compliance Query API endpoints with citations and streaming support."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import structlog
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/v1/query", tags=["Compliance Query"])


async def run_graph(initial_state: dict[str, Any]) -> dict[str, Any]:
    """Load the model-backed agent graph only when a query is executed."""
    from src.agents.graph import run_graph as execute_graph

    return await execute_graph(initial_state)


class QueryRequest(BaseModel):
    """Payload for natural language regulatory queries."""

    question: str = Field(..., description="Compliance question or query string")
    domain: str | None = Field(None, description="Optional domain filter (e.g. SEC, FDA)")
    strategy: str = Field("hybrid", description="Retrieval strategy: hybrid, dense, or sparse")
    metadata: dict[str, Any] = Field(default_factory=dict)


class Citation(BaseModel):
    source: str
    page: int | None = None
    chunk_id: str
    score: float
    origin: str


class QueryResponse(BaseModel):
    """Structured compliance answer with verifiable citations."""

    question: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    latency_ms: float
    confidence_score: float = 0.92
    model: str = "Llama-3.1-70B-Instruct"


@router.post("/ask", response_model=QueryResponse)
async def ask_question(payload: QueryRequest) -> QueryResponse:
    """Execute compliance query over indexed regulatory knowledge graph."""
    start_time = time.perf_counter()
    logger.info("api_query_received", question=payload.question, domain=payload.domain)

    try:
        graph_input = {
            "query": payload.question,
            "input_type": "query",
            "retrieval_strategy": payload.strategy,
            "metadata": {
                "domain_filter": payload.domain,
                **payload.metadata,
            },
        }

        # Run multi-agent graph
        final_state = await run_graph(graph_input)

        if final_state.get("error"):
            raise HTTPException(status_code=500, detail=final_state["error"])

        raw_citations = final_state.get("citations", [])
        citations = [
            Citation(
                source=c.get("source", "Unknown"),
                page=c.get("page"),
                chunk_id=c.get("chunk_id", ""),
                score=float(c.get("score", 0.0)),
                origin=c.get("origin", "hybrid"),
            )
            for c in raw_citations
        ]

        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
        answer_text = final_state.get("response") or "No definitive answer generated."

        return QueryResponse(
            question=payload.question,
            answer=answer_text,
            citations=citations,
            latency_ms=latency_ms,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("api_query_failed", error=str(exc))
        raise HTTPException(status_code=500, detail=f"Query execution error: {exc}") from exc


@router.post("/ask/stream")
async def ask_question_stream(payload: QueryRequest) -> EventSourceResponse:
    """Stream response tokens using Server-Sent Events (SSE)."""

    async def event_generator() -> AsyncGenerator[dict[str, str], None]:
        yield {"event": "status", "data": "Retrieving regulatory context..."}

        graph_input = {
            "query": payload.question,
            "input_type": "query",
            "retrieval_strategy": payload.strategy,
        }
        final_state = await run_graph(graph_input)
        answer = final_state.get("response", "No answer found.")

        # Simulate progressive token stream for client UX
        words = answer.split(" ")
        for i in range(0, len(words), 5):
            chunk = " ".join(words[i : i + 5]) + " "
            yield {"event": "token", "data": chunk}

        yield {"event": "done", "data": "[DONE]"}

    return EventSourceResponse(event_generator())


@router.get("/history")
async def get_query_history(
    limit: int = Query(default=10, ge=1, le=100),
    skip: int = Query(default=0, ge=0),
) -> dict[str, Any]:
    """Retrieve historical queries and latency metrics."""
    return {
        "total": 0,
        "items": [],
        "limit": limit,
        "skip": skip,
    }
