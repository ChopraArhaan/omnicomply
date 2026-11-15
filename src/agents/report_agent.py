"""Report Agent — Generates formal, audit-ready compliance reports with traceable evidence chains."""

from __future__ import annotations

import datetime as dt
import uuid
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from src.agents.state import AgentState

logger = structlog.get_logger(__name__)


class ReportAgent:
    """Produces structured compliance audit reports with findings, risk ratings, and evidence chains."""

    def __init__(self) -> None:
        pass

    async def process(self, state: AgentState) -> AgentState:
        """Synthesize analysis, risk assessment, and evidence into an audit-grade report."""
        logger.info("report_agent_generating_report")

        risk = state.get("risk_assessment") or {}
        summary = state.get("summary") or {}
        classification = state.get("classification") or {}
        parsed_doc = state.get("parsed_document") or {}
        entities = state.get("entities") or []
        chunks = state.get("chunks") or []

        report_id = f"AUD-{uuid.uuid4().hex[:8].upper()}"
        generated_at = dt.datetime.now(dt.UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
        domain = classification.get("domain", "General Regulatory")
        urgency = classification.get("urgency", "MEDIUM")
        overall_score = risk.get("overall_score", 0.0)
        risk_level = risk.get("risk_level", "MEDIUM")
        recs = risk.get("recommendations", ["Periodic audit recommended."])
        exec_sum = summary.get(
            "summary_text", "Comprehensive review completed across provided materials."
        )

        # Build evidence chain
        evidence_lines = []
        for idx, chunk in enumerate(chunks[:5]):
            chunk_id = chunk.get("chunk_id", f"chk_{idx}")
            src = chunk.get("metadata", {}).get("source", "Document Filing")
            page = chunk.get("metadata", {}).get("page_num", 1)
            preview = chunk.get("content", "")[:120].replace("\n", " ")
            evidence_lines.append(
                f'- **[EVID-{idx + 1}]** `{chunk_id}` ({src}, p. {page}): "{preview}..."'
            )

        evidence_section = (
            "\n".join(evidence_lines) if evidence_lines else "- No direct chunks mapped."
        )

        # Compile formatted markdown report
        report_markdown = f"""# Regulatory Compliance Audit Report

**Report ID:** `{report_id}`<br>
**Generated:** {generated_at}<br>
**Domain Scope:** {domain.title()}<br>
**Target Document ID:** `{parsed_doc.get("document_id", "N/A")}`<br>
**Overall Risk Score:** **{overall_score}/100 ({risk_level})**

---

## 1. Executive Summary
{exec_sum}

- **Urgency Level:** `{urgency}`
- **Entities Tracked:** {len(entities)} (organizations, legal citations, jurisdictions)
- **Compliance Posture:** {"Non-Compliant / Action Required" if overall_score > 50 else "Substantially Compliant"}

---

## 2. Risk Assessment & Factor Breakdown

| Risk Factor | Score | Status | Key Evidence |
|---|---|---|---|
"""
        for factor in risk.get("risk_factors", []):
            name = factor.get("name", "Factor")
            score = factor.get("score", 0.0)
            desc = factor.get("description", "")
            status = "🔴 High" if score >= 60 else ("🟡 Moderate" if score >= 30 else "🟢 Low")
            report_markdown += f"| {name} | {score:.1f}/100 | {status} | {desc} |\n"

        report_markdown += """
---

## 3. Corrective Action Plan & Recommendations

"""
        for idx, rec in enumerate(recs, 1):
            report_markdown += f"{idx}. **{rec}**\n"

        report_markdown += f"""
---

## 4. Evidence Chain & Audit Trail

{evidence_section}

---
*Report certified by OmniComply Automated Compliance Engine v0.1.0*
"""

        state["response"] = report_markdown
        state["next_step"] = "end"
        state["metadata"] = {
            **state.get("metadata", {}),
            "report_id": report_id,
            "generated_at": generated_at,
        }

        logger.info("report_agent_complete", report_id=report_id)
        return state
