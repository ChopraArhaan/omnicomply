"""
PDF document parser with layout-aware extraction.

Primary backend: ``llama_index.readers.file.PDFReader``.
Fallback: ``PyPDF2`` when llama-index is unavailable or the PDF is malformed.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


# ── Pydantic models ─────────────────────────────────────────────────────────


class PageContent(BaseModel):
    """Content extracted from a single PDF page."""

    page_num: int = Field(..., description="1-based page number.")
    text: str = Field(default="", description="Extracted text for this page.")
    tables: list[list[list[str]]] = Field(
        default_factory=list,
        description="Tables detected on this page, each as a list of rows.",
    )


class DocumentMetadata(BaseModel):
    """Metadata extracted from the PDF file."""

    title: str = Field(default="", description="Document title from PDF metadata.")
    author: str = Field(default="", description="Document author.")
    date: str = Field(
        default="",
        description="Creation or modification date as ISO-8601 string.",
    )
    num_pages: int = Field(default=0, description="Total page count.")


class ParsedDocument(BaseModel):
    """Fully parsed representation of a document."""

    content: str = Field(default="", description="Concatenated full-text content.")
    pages: list[PageContent] = Field(
        default_factory=list, description="Per-page content."
    )
    metadata: DocumentMetadata = Field(default_factory=DocumentMetadata)


# ── Parser ───────────────────────────────────────────────────────────────────


class PDFParser:
    """Extract text and structure from PDF files.

    Attempts to use ``llama_index`` for rich extraction.  Falls back to
    ``PyPDF2`` for basic text-only extraction when llama_index is unavailable
    or raises.
    """

    def __init__(self) -> None:
        self._llama_available: bool = self._check_llama_index()

    # ── public API ───────────────────────────────────────────────────────

    def parse(self, file_path: Path) -> ParsedDocument:
        """Parse a PDF and return structured content.

        Parameters
        ----------
        file_path:
            Absolute or relative path to the PDF file.

        Returns
        -------
        ParsedDocument
            Structured output with full text, per-page content, and metadata.
        """
        file_path = Path(file_path)
        self._validate_path(file_path)

        if self._llama_available:
            try:
                return self._parse_with_llama(file_path)
            except Exception:
                logger.warning(
                    "llama_index PDF parsing failed, falling back to PyPDF2",
                    file=str(file_path),
                )

        return self._parse_with_pypdf2(file_path)

    def parse_with_layout(self, file_path: Path) -> ParsedDocument:
        """Parse a PDF with layout-aware extraction (tables, columns).

        Falls back to :pymeth:`parse` if the layout-aware backend is not
        available.

        Parameters
        ----------
        file_path:
            Absolute or relative path to the PDF file.

        Returns
        -------
        ParsedDocument
        """
        file_path = Path(file_path)
        self._validate_path(file_path)

        if self._llama_available:
            try:
                return self._parse_with_llama(file_path, layout_aware=True)
            except Exception:
                logger.warning(
                    "Layout-aware parsing failed, falling back to basic parse",
                    file=str(file_path),
                )

        return self._parse_with_pypdf2(file_path)

    # ── private helpers ──────────────────────────────────────────────────

    @staticmethod
    def _validate_path(file_path: Path) -> None:
        if not file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")
        if file_path.suffix.lower() != ".pdf":
            raise ValueError(f"Expected a .pdf file, got: {file_path.suffix}")

    @staticmethod
    def _check_llama_index() -> bool:
        try:
            from llama_index.readers.file import PDFReader  # noqa: F401

            return True
        except ImportError:
            logger.info("llama_index not installed; will use PyPDF2 fallback.")
            return False

    def _parse_with_llama(
        self, file_path: Path, *, layout_aware: bool = False
    ) -> ParsedDocument:
        """Use llama_index's PDFReader for rich extraction."""
        from llama_index.readers.file import PDFReader

        reader = PDFReader(return_full_document=False)
        documents = reader.load_data(file=file_path)

        pages: list[PageContent] = []
        text_parts: list[str] = []
        for idx, doc in enumerate(documents, start=1):
            page_text = doc.text if hasattr(doc, "text") else str(doc)
            pages.append(PageContent(page_num=idx, text=page_text))
            text_parts.append(page_text)

        full_text = "\n\n".join(text_parts)

        # Attempt metadata extraction from first document's extra_info
        meta_dict: dict[str, Any] = {}
        if documents and hasattr(documents[0], "extra_info") and documents[0].extra_info:
            meta_dict = documents[0].extra_info

        metadata = DocumentMetadata(
            title=str(meta_dict.get("title", file_path.stem)),
            author=str(meta_dict.get("author", "")),
            date=str(meta_dict.get("creation_date", "")),
            num_pages=len(pages),
        )

        logger.info(
            "PDF parsed via llama_index",
            file=str(file_path),
            pages=len(pages),
            layout_aware=layout_aware,
        )

        return ParsedDocument(content=full_text, pages=pages, metadata=metadata)

    def _parse_with_pypdf2(self, file_path: Path) -> ParsedDocument:
        """Fallback PDF parser using PyPDF2."""
        try:
            from PyPDF2 import PdfReader
        except ImportError as exc:
            raise ImportError(
                "Neither llama_index nor PyPDF2 is installed.  "
                "Install at least one: pip install PyPDF2"
            ) from exc

        reader = PdfReader(str(file_path))

        pages: list[PageContent] = []
        text_parts: list[str] = []
        for idx, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            pages.append(PageContent(page_num=idx, text=page_text))
            text_parts.append(page_text)

        full_text = "\n\n".join(text_parts)

        # Extract metadata from PDF info dict
        info = reader.metadata
        raw_date = ""
        if info and info.creation_date:
            raw_date = (
                info.creation_date.isoformat()
                if isinstance(info.creation_date, dt.datetime)
                else str(info.creation_date)
            )

        metadata = DocumentMetadata(
            title=str((info and info.title) or file_path.stem),
            author=str((info and info.author) or ""),
            date=raw_date,
            num_pages=len(pages),
        )

        logger.info(
            "PDF parsed via PyPDF2 fallback",
            file=str(file_path),
            pages=len(pages),
        )

        return ParsedDocument(content=full_text, pages=pages, metadata=metadata)
