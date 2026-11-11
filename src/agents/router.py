"""Router Agent — Classifies inputs and determines graph execution paths."""

from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import Any

import structlog
from src.agents.state import AgentState
from src.analysis.classifier import RegulatoryClassifier

logger = structlog.get_logger(__name__)


class RouterAgent:
    """Evaluates incoming state and routes to the appropriate specialized agent."""

    def __init__(self, classifier: RegulatoryClassifier | None = None) -> None:
        self.classifier = classifier or RegulatoryClassifier()

    def route(self, state: AgentState) -> str:
        """Determine next node in the graph based on state content."""
        file_path_str = state.get("file_path")
        query_str = state.get("query")
        input_type = state.get("input_type")

        # 1. Direct file input
        if file_path_str:
            detected_type = self.detect_input_type(file_path_str)
            logger.info("routing_file_input", file=file_path_str, detected=detected_type)
            if detected_type in ("audio", "voice"):
                return "audio_agent"
            return "document_agent"

        # 2. Pre-specified input type
        if input_type in ("audio", "voice"):
            return "audio_agent"
        if input_type in ("pdf", "image", "document", "text"):
            return "document_agent"

        # 3. Query or Question Answering
        if query_str:
            # Check if query requests report generation directly
            lower_q = query_str.lower()
            if any(k in lower_q for k in ["generate report", "audit report", "compliance report"]):
                return "report_agent"
            if any(k in lower_q for k in ["compare", "semantic diff", "conflicts between"]):
                return "compliance_agent"
            return "query_agent"

        # Default fallback
        logger.warning("router_falling_back_to_query_agent")
        return "query_agent"

    @staticmethod
    def detect_input_type(file_path: str | Path) -> str:
        """Infer modality from file extension or mime type."""
        path = Path(file_path)
        ext = path.suffix.lower()

        audio_exts = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac"}
        pdf_exts = {".pdf"}
        image_exts = {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"}
        text_exts = {".txt", ".md", ".json", ".csv", ".xml", ".html"}

        if ext in audio_exts:
            return "audio"
        if ext in pdf_exts:
            return "pdf"
        if ext in image_exts:
            return "image"
        if ext in text_exts:
            return "text"

        mime, _ = mimetypes.guess_type(str(file_path))
        if mime:
            if mime.startswith("audio/"):
                return "audio"
            if mime == "application/pdf":
                return "pdf"
            if mime.startswith("image/"):
                return "image"
            if mime.startswith("text/"):
                return "text"

        return "unknown"
