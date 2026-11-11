"""
Unified ingestion pipeline for OmniComply.

Orchestrates the full document-processing flow:
    detect input type → route to processor → chunk → extract entities
"""

from __future__ import annotations

import datetime as dt
import mimetypes
import uuid
from pathlib import Path
from typing import Any, Optional

import structlog
from pydantic import BaseModel, Field

from src.config.settings import Settings
from src.ingestion.audio_processor import AudioProcessor, Transcript
from src.ingestion.chunker import AdaptiveChunker, DocumentChunk
from src.ingestion.ner import Entity, LegalNERExtractor
from src.ingestion.ocr import OCRProcessor
from src.ingestion.pdf_parser import ParsedDocument, PDFParser

logger = structlog.get_logger(__name__)

# ── Input type taxonomy ──────────────────────────────────────────────────────

_PDF_EXTENSIONS = {".pdf"}
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}
_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm", ".wma", ".aac"}
_TEXT_EXTENSIONS = {".txt", ".md", ".rst", ".csv", ".json", ".xml", ".html"}


# ── Pydantic models ─────────────────────────────────────────────────────────


class ProcessedDocument(BaseModel):
    """Final output of the ingestion pipeline."""

    document_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this document.",
    )
    source_type: str = Field(
        ..., description="Detected input type (pdf, image, audio, text)."
    )
    chunks: list[DocumentChunk] = Field(
        default_factory=list, description="Token-bounded document chunks."
    )
    entities: list[Entity] = Field(
        default_factory=list, description="Named entities extracted from the full text."
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Document-level metadata."
    )
    processed_at: str = Field(
        default_factory=lambda: dt.datetime.now(dt.timezone.utc).isoformat(),
        description="ISO-8601 timestamp of processing.",
    )


# ── Pipeline ─────────────────────────────────────────────────────────────────


class IngestionPipeline:
    """Orchestrate end-to-end document ingestion.

    Usage::

        pipeline = IngestionPipeline()
        result = await pipeline.ingest(Path("filing.pdf"))
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self._settings = settings or Settings()
        self._pdf_parser = PDFParser()
        self._ocr = OCRProcessor()
        self._chunker = AdaptiveChunker()
        self._ner = LegalNERExtractor()
        self._audio = AudioProcessor()

    # ── public API ───────────────────────────────────────────────────────

    async def ingest(self, file_path: Path) -> ProcessedDocument:
        """Ingest a file and return a fully processed document.

        Parameters
        ----------
        file_path:
            Path to the file to ingest.

        Returns
        -------
        ProcessedDocument
            Chunked, entity-annotated document ready for embedding & storage.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        ValueError
            If the file type is unsupported.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        source_type = self._detect_type(file_path)
        logger.info(
            "Ingestion started",
            file=str(file_path),
            source_type=source_type,
        )

        # ── Route to appropriate processor ───────────────────────────────
        if source_type == "pdf":
            parsed = await self._process_pdf(file_path)
        elif source_type == "image":
            parsed = await self._process_image(file_path)
        elif source_type == "audio":
            parsed = await self._process_audio(file_path)
        elif source_type == "text":
            parsed = await self._process_text(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")

        # ── Chunk ────────────────────────────────────────────────────────
        chunks = self._chunker.chunk(
            parsed,
            chunk_size=self._settings.rag_chunk_size,
            chunk_overlap=self._settings.rag_chunk_overlap,
        )

        # ── NER over full text ───────────────────────────────────────────
        entities = self._ner.extract_entities(parsed.content)

        # ── Assemble result ──────────────────────────────────────────────
        result = ProcessedDocument(
            source_type=source_type,
            chunks=chunks,
            entities=entities,
            metadata={
                "file_name": file_path.name,
                "file_size_bytes": file_path.stat().st_size,
                "title": parsed.metadata.title,
                "author": parsed.metadata.author,
                "num_pages": parsed.metadata.num_pages,
                "total_chunks": len(chunks),
                "total_entities": len(entities),
            },
        )

        logger.info(
            "Ingestion complete",
            document_id=result.document_id,
            chunks=len(chunks),
            entities=len(entities),
        )
        return result

    # ── type detection ───────────────────────────────────────────────────

    @staticmethod
    def _detect_type(file_path: Path) -> str:
        """Detect the input file type by extension and MIME type.

        Returns
        -------
        str
            One of ``"pdf"``, ``"image"``, ``"audio"``, ``"text"``.
        """
        ext = file_path.suffix.lower()
        if ext in _PDF_EXTENSIONS:
            return "pdf"
        if ext in _IMAGE_EXTENSIONS:
            return "image"
        if ext in _AUDIO_EXTENSIONS:
            return "audio"
        if ext in _TEXT_EXTENSIONS:
            return "text"

        # Fallback: try mimetypes
        mime, _ = mimetypes.guess_type(str(file_path))
        if mime:
            if mime.startswith("image/"):
                return "image"
            if mime.startswith("audio/"):
                return "audio"
            if mime.startswith("text/") or mime == "application/json":
                return "text"
            if mime == "application/pdf":
                return "pdf"

        raise ValueError(
            f"Cannot determine file type for {file_path.name} "
            f"(extension={ext}, mime={mime})"
        )

    # ── per-type processors ──────────────────────────────────────────────

    async def _process_pdf(self, file_path: Path) -> ParsedDocument:
        """Process a PDF file through the PDF parser."""
        try:
            return self._pdf_parser.parse_with_layout(file_path)
        except Exception:
            logger.warning("Layout parse failed, trying basic parse", file=str(file_path))
            return self._pdf_parser.parse(file_path)

    async def _process_image(self, file_path: Path) -> ParsedDocument:
        """Process an image file through OCR."""
        text = self._ocr.process_image(file_path)
        return ParsedDocument(
            content=text,
            pages=[],
            metadata=_make_metadata(title=file_path.stem, num_pages=1),
        )

    async def _process_audio(self, file_path: Path) -> ParsedDocument:
        """Transcribe an audio file and wrap as a ParsedDocument."""
        transcript: Transcript = self._audio.transcribe(file_path)
        return ParsedDocument(
            content=transcript.text,
            pages=[],
            metadata=_make_metadata(
                title=file_path.stem,
                num_pages=0,
            ),
        )

    async def _process_text(self, file_path: Path) -> ParsedDocument:
        """Read a plain text file."""
        content = file_path.read_text(encoding="utf-8", errors="replace")
        return ParsedDocument(
            content=content,
            pages=[],
            metadata=_make_metadata(title=file_path.stem, num_pages=1),
        )


# ── helper ───────────────────────────────────────────────────────────────────

def _make_metadata(*, title: str, num_pages: int):  # noqa: ANN202
    """Build a ``DocumentMetadata`` without importing at module level to avoid cycles."""
    from src.ingestion.pdf_parser import DocumentMetadata

    return DocumentMetadata(title=title, num_pages=num_pages)
