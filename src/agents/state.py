"""State schema for the LangGraph multi-agent compliance orchestration graph."""

from __future__ import annotations

from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    """Global state passed across all agents in the LangGraph workflow."""

    # General communication
    messages: list[dict[str, Any]]

    # Input routing
    input_type: str  # "pdf", "image", "audio", "text", "query", "monitor"
    file_path: str | None
    query: str | None

    # Ingestion outputs
    parsed_document: dict[str, Any] | None
    chunks: list[dict[str, Any]] | None
    entities: list[dict[str, Any]] | None

    # Retrieval outputs
    retrieval_results: list[dict[str, Any]] | None
    retrieval_strategy: str | None

    # Analysis outputs
    classification: dict[str, Any] | None
    risk_assessment: dict[str, Any] | None
    summary: dict[str, Any] | None
    semantic_diff: dict[str, Any] | None

    # Final response
    response: str | None
    citations: list[dict[str, Any]] | None

    # Routing and control flow
    next_step: str | None
    retry_count: int
    error: str | None
    metadata: dict[str, Any]
