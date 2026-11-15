"""
Adaptive document chunker with layout-aware splitting.

Preserves semantic boundaries such as numbered lists, table rows, and section
headings while keeping each chunk within a target token budget.  Token counting
uses ``tiktoken`` (cl100k_base encoding).
"""

from __future__ import annotations

import re
import uuid
from typing import TYPE_CHECKING

import structlog
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from src.ingestion.pdf_parser import ParsedDocument

logger = structlog.get_logger(__name__)


# ── Pydantic models ─────────────────────────────────────────────────────────


class ChunkMetadata(BaseModel):
    """Provenance metadata attached to every chunk."""

    source: str = Field(default="", description="Origin file path or URL.")
    page_num: int | None = Field(default=None, description="Page number the chunk originated from.")
    chunk_index: int = Field(default=0, description="Sequential index within the document.")
    parent_chunk_id: str | None = Field(
        default=None,
        description="UUID of a parent chunk if this is a sub-chunk.",
    )


class DocumentChunk(BaseModel):
    """A token-bounded slice of a document with full provenance."""

    chunk_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this chunk.",
    )
    content: str = Field(..., description="Chunk text.")
    metadata: ChunkMetadata = Field(default_factory=ChunkMetadata)
    token_count: int = Field(default=0, ge=0, description="Token count (cl100k_base).")


# ── Chunker ──────────────────────────────────────────────────────────────────


class AdaptiveChunker:
    """Layout-aware document chunker that respects structural boundaries.

    The chunking strategy:
    1. Detect section headings and keep them glued to their body.
    2. Preserve table rows as atomic units — never split mid-row.
    3. Respect numbered / bulleted list items.
    4. Apply overlap so that retrieval doesn't miss cross-boundary context.
    """

    # Patterns for structural detection
    _SECTION_HEADING_RE = re.compile(
        r"^(?:"
        r"(?:#{1,6}\s+.+)"  # Markdown headings
        r"|(?:SECTION\s+\d+)"  # SECTION 1, SECTION 2 …
        r"|(?:§\s*\d+[\.\d]*)"  # § 240.10b-5
        r"|(?:Article\s+[IVXLCDM\d]+)"  # Article III
        r"|(?:\d+\.\d+(?:\.\d+)*\s+[A-Z])"  # 1.2.3 Title
        r")",
        re.MULTILINE,
    )
    _TABLE_ROW_RE = re.compile(r"^\|.*\|$", re.MULTILINE)
    _LIST_ITEM_RE = re.compile(
        r"^(?:\s*[-*•]\s+|\s*\d+[\.\)]\s+|\s*\([a-z]\)\s+|\s*[a-z]\)\s+)",
        re.MULTILINE,
    )

    def __init__(self) -> None:
        self._encoder = self._load_tokenizer()

    # ── public API ───────────────────────────────────────────────────────

    def chunk(
        self,
        document: ParsedDocument,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
    ) -> list[DocumentChunk]:
        """Split a parsed document into token-bounded chunks.

        Parameters
        ----------
        document:
            The parsed document to chunk.
        chunk_size:
            Target maximum token count per chunk.
        chunk_overlap:
            Number of overlapping tokens between consecutive chunks.

        Returns
        -------
        list[DocumentChunk]
            Ordered list of chunks with metadata.
        """
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")

        all_chunks: list[DocumentChunk] = []

        for page in document.pages:
            if not page.text.strip():
                continue

            # Separate table content from prose
            tables, prose = self._preserve_tables(page.text)

            # Detect section boundaries inside prose
            sections = self._detect_section_boundaries(prose)

            # Split sections into token-bounded chunks with overlap
            for section in sections:
                section_chunks = self._split_with_overlap(section, chunk_size, chunk_overlap)
                for chunk_text in section_chunks:
                    token_count = self._count_tokens(chunk_text)
                    all_chunks.append(
                        DocumentChunk(
                            content=chunk_text,
                            metadata=ChunkMetadata(
                                source=document.metadata.title,
                                page_num=page.page_num,
                                chunk_index=len(all_chunks),
                            ),
                            token_count=token_count,
                        )
                    )

            # Tables become their own chunks (one per table)
            for table_text in tables:
                token_count = self._count_tokens(table_text)
                if token_count > 0:
                    all_chunks.append(
                        DocumentChunk(
                            content=table_text,
                            metadata=ChunkMetadata(
                                source=document.metadata.title,
                                page_num=page.page_num,
                                chunk_index=len(all_chunks),
                            ),
                            token_count=token_count,
                        )
                    )

        # Renumber chunk indices sequentially
        for idx, ch in enumerate(all_chunks):
            ch.metadata.chunk_index = idx

        logger.info(
            "Document chunked",
            total_chunks=len(all_chunks),
            total_tokens=sum(c.token_count for c in all_chunks),
        )
        return all_chunks

    # ── private helpers ──────────────────────────────────────────────────

    def _detect_section_boundaries(self, text: str) -> list[str]:
        """Split text at section headings, keeping each heading with its body.

        Returns
        -------
        list[str]
            Sections ordered by appearance.  A section includes the heading
            line and all text until the next heading.
        """
        matches = list(self._SECTION_HEADING_RE.finditer(text))
        if not matches:
            return [text] if text.strip() else []

        sections: list[str] = []

        # Text before the first heading
        preamble = text[: matches[0].start()].strip()
        if preamble:
            sections.append(preamble)

        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            section = text[start:end].strip()
            if section:
                sections.append(section)

        return sections

    def _preserve_tables(self, text: str) -> tuple[list[str], str]:
        """Extract markdown tables from text.

        Returns
        -------
        tuple[list[str], str]
            (list_of_table_strings, remaining_prose)
        """
        lines = text.split("\n")
        tables: list[str] = []
        prose_lines: list[str] = []
        current_table: list[str] = []
        in_table = False

        for line in lines:
            is_table_line = bool(self._TABLE_ROW_RE.match(line.strip()))

            if is_table_line:
                in_table = True
                current_table.append(line)
            else:
                if in_table and current_table:
                    tables.append("\n".join(current_table))
                    current_table = []
                    in_table = False
                prose_lines.append(line)

        # Flush any trailing table
        if current_table:
            tables.append("\n".join(current_table))

        return tables, "\n".join(prose_lines)

    def _split_with_overlap(self, text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
        """Split text into token-bounded chunks with overlap.

        Splits on paragraph boundaries (double newline) first, then on single
        newlines, then on sentence boundaries, then finally on whitespace.

        Parameters
        ----------
        text:
            The text to split.
        chunk_size:
            Maximum tokens per chunk.
        chunk_overlap:
            Tokens of overlap between consecutive chunks.

        Returns
        -------
        list[str]
        """
        if not text.strip():
            return []

        total_tokens = self._count_tokens(text)
        if total_tokens <= chunk_size:
            return [text.strip()]

        # Split into atomic units (paragraphs → lines → sentences)
        units = self._split_into_units(text)

        chunks: list[str] = []
        current_units: list[str] = []
        current_tokens = 0

        for unit in units:
            unit_tokens = self._count_tokens(unit)

            # If a single unit exceeds chunk_size, force-split it
            if unit_tokens > chunk_size:
                # Flush what we have
                if current_units:
                    chunks.append("\n".join(current_units).strip())
                    current_units = []
                    current_tokens = 0
                # Hard split the oversized unit by tokens
                chunks.extend(self._hard_split(unit, chunk_size, chunk_overlap))
                continue

            if current_tokens + unit_tokens > chunk_size:
                # Emit current chunk
                chunks.append("\n".join(current_units).strip())

                # Build overlap from the tail of current_units
                overlap_units: list[str] = []
                overlap_tokens = 0
                for prev_unit in reversed(current_units):
                    prev_tokens = self._count_tokens(prev_unit)
                    if overlap_tokens + prev_tokens > chunk_overlap:
                        break
                    overlap_units.insert(0, prev_unit)
                    overlap_tokens += prev_tokens

                current_units = [*overlap_units, unit]
                current_tokens = overlap_tokens + unit_tokens
            else:
                current_units.append(unit)
                current_tokens += unit_tokens

        if current_units:
            final = "\n".join(current_units).strip()
            if final:
                chunks.append(final)

        return chunks

    def _split_into_units(self, text: str) -> list[str]:
        """Break text into the smallest meaningful units for chunking.

        Tries paragraph boundaries first, then individual lines.  Keeps list
        items as atomic units.
        """
        # Split on double newlines (paragraph boundaries)
        paragraphs = re.split(r"\n\s*\n", text)
        units: list[str] = []
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            # If the paragraph is short enough, keep it whole
            if self._count_tokens(para) <= 128:
                units.append(para)
            else:
                # Split into individual lines
                for line in para.split("\n"):
                    line = line.strip()
                    if line:
                        units.append(line)
        return units

    def _hard_split(self, text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
        """Force-split text that exceeds chunk_size by raw token slicing."""
        tokens = self._encode(text)
        chunks: list[str] = []
        start = 0
        while start < len(tokens):
            end = min(start + chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunks.append(self._decode(chunk_tokens).strip())
            start += chunk_size - chunk_overlap
        return [c for c in chunks if c]

    # ── tokenizer helpers ────────────────────────────────────────────────

    @staticmethod
    def _load_tokenizer():
        """Load tiktoken cl100k_base encoder."""
        try:
            import tiktoken

            return tiktoken.get_encoding("cl100k_base")
        except ImportError:
            logger.warning("tiktoken not installed; falling back to whitespace token count")
            return None

    def _count_tokens(self, text: str) -> int:
        if self._encoder is not None:
            return len(self._encoder.encode(text))
        # Rough whitespace-based fallback (~0.75 tokens per word)
        return int(len(text.split()) * 1.33)

    def _encode(self, text: str) -> list[int]:
        if self._encoder is not None:
            return self._encoder.encode(text)
        # Fallback: one "token" per character (rough)
        return list(text.encode("utf-8"))

    def _decode(self, tokens: list[int]) -> str:
        if self._encoder is not None:
            return self._encoder.decode(tokens)
        return bytes(tokens).decode("utf-8", errors="replace")
