"""Compliance Agent — Performs domain classification, risk scoring, and policy gap analysis."""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from src.analysis.classifier import RegulatoryClassifier
from src.analysis.risk_scorer import ComplianceRiskScorer
from src.analysis.summarizer import RegulatoryDocumentSummarizer

if TYPE_CHECKING:
    from src.agents.state import AgentState

logger = structlog.get_logger(__name__)


class ComplianceAgent:
    """Evaluates regulatory implications, assesses risk levels, and extracts compliance obligations."""

    def __init__(
        self,
        classifier: RegulatoryClassifier | None = None,
        risk_scorer: ComplianceRiskScorer | None = None,
        summarizer: RegulatoryDocumentSummarizer | None = None,
    ) -> None:
        self.classifier = classifier or RegulatoryClassifier()
        self.risk_scorer = risk_scorer or ComplianceRiskScorer()
        self.summarizer = summarizer or RegulatoryDocumentSummarizer()

    async def process(self, state: AgentState) -> AgentState:
        """Run classification, risk assessment, and summarization on document state."""
        chunks = state.get("chunks", [])
        entities = state.get("entities", [])

        # Combine text from chunks or state
        full_text = " ".join([c.get("content", "") for c in chunks]) if chunks else ""
        if not full_text and state.get("query"):
            full_text = state["query"]

        if not full_text:
            state["error"] = "ComplianceAgent found no document content to analyze"
            return state

        logger.info(
            "compliance_agent_analyzing_document", text_len=len(full_text), entities=len(entities)
        )
        try:
            # 1. Domain & Urgency Classification
            domain_result = self.classifier.classify_domain(full_text[:2000])
            urgency = self.classifier.classify_urgency(full_text[:2000])
            doc_type = self.classifier.classify_document_type(full_text[:2000])

            classification_data = {
                "domain": domain_result.domain,
                "domain_confidence": domain_result.confidence,
                "urgency": urgency.value,
                "document_type": doc_type,
            }
            state["classification"] = classification_data

            # 2. Risk Assessment
            assessment = self.risk_scorer.score_document(
                text=full_text,
                entities=entities,
                domain=domain_result.domain,
            )
            state["risk_assessment"] = assessment.model_dump()

            # 3. Executive Summarization
            summary = self.summarizer.executive_summary(full_text[:8000])
            state["summary"] = summary.model_dump()

            # 4. Determine next step
            # If user query requested report or risk is critical, trigger ReportAgent
            query_str = (state.get("query") or "").lower()
            if "report" in query_str or assessment.risk_level.value in ("CRITICAL", "HIGH"):
                state["next_step"] = "report_agent"
            else:
                state["next_step"] = "end"
                state["response"] = (
                    f"### Compliance Analysis Summary\n\n"
                    f"- **Domain:** {domain_result.domain.title()} (Confidence: {domain_result.confidence:.1%})\n"
                    f"- **Urgency:** {urgency.value}\n"
                    f"- **Overall Risk Score:** {assessment.overall_score}/100 ({assessment.risk_level.value})\n\n"
                    f"**Executive Summary:**\n{summary.summary_text}\n\n"
                    f"**Key Recommendations:**\n"
                    + "\n".join([f"- {r}" for r in assessment.recommendations])
                )

            logger.info(
                "compliance_agent_success",
                domain=domain_result.domain,
                risk_level=assessment.risk_level.value,
                overall_score=assessment.overall_score,
            )
        except Exception as exc:
            logger.error("compliance_agent_failed", error=str(exc))
            state["error"] = f"Compliance analysis failed: {exc}"

        return state
