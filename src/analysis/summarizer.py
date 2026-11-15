"""
Regulatory Document Summarizer.

Provides executive, detailed, and change summaries for regulatory documents
using LangChain's map-reduce summarization chain for handling long documents.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from langchain_core.documents import Document
    from langchain_core.language_models import BaseLLM

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class Summary(BaseModel):
    """Concise executive-level summary of a regulatory document."""

    title: str = Field(..., description="Inferred title of the document.")
    summary_text: str = Field(..., description="Prose summary paragraph(s).")
    key_points: list[str] = Field(
        default_factory=list,
        description="Bullet-point key takeaways.",
    )
    word_count: int = Field(..., description="Word count of the source document.")


class DetailedSummary(BaseModel):
    """Section-by-section detailed summary."""

    title: str = Field(..., description="Inferred title of the document.")
    sections: list[SectionSummary] = Field(
        default_factory=list,
        description="Per-section summaries.",
    )
    overall_summary: str = Field(..., description="Consolidated summary across sections.")
    word_count: int = Field(..., description="Word count of the source document.")


class SectionSummary(BaseModel):
    """Summary of a single document section."""

    heading: str = Field(..., description="Section heading or inferred heading.")
    content: str = Field(..., description="Summary of this section.")
    key_points: list[str] = Field(
        default_factory=list,
        description="Key points from this section.",
    )


# Rebuild DetailedSummary so the forward-ref to SectionSummary resolves.
DetailedSummary.model_rebuild()


class ChangeSummary(BaseModel):
    """Summary of changes between two versions of a regulation."""

    added_provisions: list[str] = Field(
        default_factory=list,
        description="Provisions that are new in the updated text.",
    )
    removed_provisions: list[str] = Field(
        default_factory=list,
        description="Provisions that were removed.",
    )
    modified_provisions: list[str] = Field(
        default_factory=list,
        description="Provisions that were substantively changed.",
    )
    risk_impact: str = Field(
        "",
        description="Assessment of how the changes affect compliance risk.",
    )


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_EXECUTIVE_MAP_PROMPT = PromptTemplate(
    input_variables=["text"],
    template=(
        "You are a regulatory compliance expert. "
        "Summarize the following section of a regulatory document. "
        "Identify key obligations, deadlines, and penalties:\n\n{text}\n\n"
        "CONCISE SUMMARY:"
    ),
)

_EXECUTIVE_COMBINE_PROMPT = PromptTemplate(
    input_variables=["text"],
    template=(
        "You are a regulatory compliance expert. "
        "Given the following partial summaries of a regulatory document, "
        "produce a final executive summary. Include:\n"
        "1. A one-line title for the regulation.\n"
        "2. A concise prose summary (2-4 paragraphs).\n"
        "3. A list of key points (bullet items).\n\n"
        "{text}\n\n"
        "Respond in the following format exactly:\n"
        "TITLE: <title>\n"
        "SUMMARY: <summary>\n"
        "KEY POINTS:\n- <point 1>\n- <point 2>\n..."
    ),
)

_DETAILED_MAP_PROMPT = PromptTemplate(
    input_variables=["text"],
    template=(
        "You are a regulatory compliance expert. "
        "Provide a detailed summary of the following section. "
        "Include the section heading (or infer one), a thorough summary, "
        "and key points:\n\n{text}\n\n"
        "SECTION HEADING: <heading>\n"
        "SUMMARY: <summary>\n"
        "KEY POINTS:\n- <point 1>\n- <point 2>\n..."
    ),
)

_CHANGE_PROMPT = PromptTemplate(
    input_variables=["text"],
    template=(
        "You are a regulatory compliance expert. "
        "Compare the OLD and NEW versions of a regulation below and identify:\n"
        "1. Added provisions (entirely new requirements).\n"
        "2. Removed provisions (requirements that no longer appear).\n"
        "3. Modified provisions (substantive wording changes).\n"
        "4. Risk impact (how these changes affect compliance risk).\n\n"
        "{text}\n\n"
        "Respond in the following format exactly:\n"
        "ADDED:\n- <provision>\n"
        "REMOVED:\n- <provision>\n"
        "MODIFIED:\n- <provision>\n"
        "RISK IMPACT: <impact assessment>"
    ),
)


# ---------------------------------------------------------------------------
# Summarizer class
# ---------------------------------------------------------------------------


class RegulatoryDocumentSummarizer:
    """Summarize regulatory documents using LangChain map-reduce chains.

    Args:
        llm: A LangChain-compatible LLM instance.
        chunk_size: Token/character chunk size for splitting long documents.
        chunk_overlap: Overlap between consecutive chunks.
    """

    def __init__(
        self,
        llm: BaseLLM,
        chunk_size: int = 4_000,
        chunk_overlap: int = 400,
    ) -> None:
        self._llm = llm
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        logger.info(
            "summarizer.initialized",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _split_text(self, text: str) -> list[Document]:
        """Split raw text into LangChain Document objects."""
        return self._splitter.create_documents([text])

    @staticmethod
    def _parse_executive_output(raw: str, word_count: int) -> Summary:
        """Parse the LLM's structured output into a Summary model."""
        title = ""
        summary_text = ""
        key_points: list[str] = []

        lines = raw.strip().splitlines()
        section = None
        for line in lines:
            stripped = line.strip()
            if stripped.upper().startswith("TITLE:"):
                title = stripped.split(":", 1)[1].strip()
                section = "title"
            elif stripped.upper().startswith("SUMMARY:"):
                summary_text = stripped.split(":", 1)[1].strip()
                section = "summary"
            elif stripped.upper().startswith("KEY POINTS"):
                section = "key_points"
            elif section == "key_points" and stripped.startswith("- "):
                key_points.append(stripped[2:].strip())
            elif section == "summary":
                summary_text += " " + stripped

        return Summary(
            title=title or "Untitled Regulation",
            summary_text=summary_text.strip(),
            key_points=key_points,
            word_count=word_count,
        )

    @staticmethod
    def _parse_detailed_section(raw: str) -> SectionSummary:
        """Parse a single section's structured output."""
        heading = ""
        content = ""
        key_points: list[str] = []
        section = None

        for line in raw.strip().splitlines():
            stripped = line.strip()
            if stripped.upper().startswith("SECTION HEADING:"):
                heading = stripped.split(":", 1)[1].strip()
                section = "heading"
            elif stripped.upper().startswith("SUMMARY:"):
                content = stripped.split(":", 1)[1].strip()
                section = "summary"
            elif stripped.upper().startswith("KEY POINTS"):
                section = "key_points"
            elif section == "key_points" and stripped.startswith("- "):
                key_points.append(stripped[2:].strip())
            elif section == "summary":
                content += " " + stripped

        return SectionSummary(
            heading=heading or "Section",
            content=content.strip(),
            key_points=key_points,
        )

    @staticmethod
    def _parse_change_output(raw: str) -> ChangeSummary:
        """Parse the change-summary structured output."""
        added: list[str] = []
        removed: list[str] = []
        modified: list[str] = []
        risk_impact = ""

        section = None
        for line in raw.strip().splitlines():
            stripped = line.strip()
            upper = stripped.upper()
            if upper.startswith("ADDED"):
                section = "added"
            elif upper.startswith("REMOVED"):
                section = "removed"
            elif upper.startswith("MODIFIED"):
                section = "modified"
            elif upper.startswith("RISK IMPACT:"):
                risk_impact = stripped.split(":", 1)[1].strip()
                section = "risk_impact"
            elif stripped.startswith("- "):
                item = stripped[2:].strip()
                if section == "added":
                    added.append(item)
                elif section == "removed":
                    removed.append(item)
                elif section == "modified":
                    modified.append(item)
            elif section == "risk_impact":
                risk_impact += " " + stripped

        return ChangeSummary(
            added_provisions=added,
            removed_provisions=removed,
            modified_provisions=modified,
            risk_impact=risk_impact.strip(),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def executive_summary(self, text: str) -> Summary:
        """Produce an executive summary of a regulatory document.

        Uses LangChain's map-reduce chain to handle documents that exceed
        the LLM's context window.

        Args:
            text: The full text of the regulatory document.

        Returns:
            A ``Summary`` containing the title, prose summary, key points,
            and the source word count.
        """
        word_count = len(text.split())
        logger.info("summarizer.executive_summary.start", word_count=word_count)

        docs = self._split_text(text)
        if len(docs) <= 1:
            # Short document — use stuff chain directly.
            chain = load_summarize_chain(
                self._llm,
                chain_type="stuff",
                prompt=_EXECUTIVE_COMBINE_PROMPT,
            )
        else:
            chain = load_summarize_chain(
                self._llm,
                chain_type="map_reduce",
                map_prompt=_EXECUTIVE_MAP_PROMPT,
                combine_prompt=_EXECUTIVE_COMBINE_PROMPT,
            )

        raw_output: str = chain.invoke({"input_documents": docs})["output_text"]
        summary = self._parse_executive_output(raw_output, word_count)

        logger.info(
            "summarizer.executive_summary.done",
            key_points=len(summary.key_points),
        )
        return summary

    def detailed_summary(self, text: str, max_sections: int = 10) -> DetailedSummary:
        """Produce a section-by-section detailed summary.

        Each chunk of the document is individually summarized (map step) and
        then an overall consolidation is performed (reduce step).

        Args:
            text: The full text of the regulatory document.
            max_sections: Maximum number of section summaries to return.

        Returns:
            A ``DetailedSummary`` with per-section and overall summaries.
        """
        word_count = len(text.split())
        logger.info(
            "summarizer.detailed_summary.start",
            word_count=word_count,
            max_sections=max_sections,
        )

        docs = self._split_text(text)

        # Map step — summarize each chunk individually.
        map_chain = load_summarize_chain(
            self._llm,
            chain_type="stuff",
            prompt=_DETAILED_MAP_PROMPT,
        )
        sections: list[SectionSummary] = []
        for doc in docs[:max_sections]:
            raw = map_chain.invoke({"input_documents": [doc]})["output_text"]
            sections.append(self._parse_detailed_section(raw))

        # Combine step — produce an overall summary.
        combine_chain = load_summarize_chain(
            self._llm,
            chain_type="stuff",
            prompt=_EXECUTIVE_COMBINE_PROMPT,
        )
        overall_raw = combine_chain.invoke({"input_documents": docs})["output_text"]
        overall_summary = self._parse_executive_output(overall_raw, word_count).summary_text

        result = DetailedSummary(
            title=sections[0].heading if sections else "Untitled",
            sections=sections,
            overall_summary=overall_summary,
            word_count=word_count,
        )
        logger.info(
            "summarizer.detailed_summary.done",
            sections=len(result.sections),
        )
        return result

    def change_summary(self, old_text: str, new_text: str) -> ChangeSummary:
        """Summarize the differences between two regulation versions.

        Concatenates old and new text with markers and runs the change-analysis
        prompt via a map-reduce chain if the combined text is long.

        Args:
            old_text: The previous version of the regulation.
            new_text: The updated version of the regulation.

        Returns:
            A ``ChangeSummary`` with added, removed, and modified provisions
            plus a risk impact assessment.
        """
        logger.info(
            "summarizer.change_summary.start",
            old_words=len(old_text.split()),
            new_words=len(new_text.split()),
        )

        combined = f"=== OLD VERSION ===\n{old_text}\n\n=== NEW VERSION ===\n{new_text}"
        docs = self._split_text(combined)

        if len(docs) <= 1:
            chain = load_summarize_chain(
                self._llm,
                chain_type="stuff",
                prompt=_CHANGE_PROMPT,
            )
        else:
            chain = load_summarize_chain(
                self._llm,
                chain_type="map_reduce",
                map_prompt=_CHANGE_PROMPT,
                combine_prompt=_CHANGE_PROMPT,
            )

        raw_output = chain.invoke({"input_documents": docs})["output_text"]
        result = self._parse_change_output(raw_output)

        logger.info(
            "summarizer.change_summary.done",
            added=len(result.added_provisions),
            removed=len(result.removed_provisions),
            modified=len(result.modified_provisions),
        )
        return result
