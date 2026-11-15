"""Unit tests for AdaptiveChunker layout-aware document chunking."""

from __future__ import annotations

import pytest

from src.ingestion.chunker import AdaptiveChunker
from src.ingestion.pdf_parser import PageContent, ParsedDocument


@pytest.fixture
def chunker() -> AdaptiveChunker:
    return AdaptiveChunker()


def test_chunker_basic_splitting(chunker: AdaptiveChunker, sample_parsed_document: ParsedDocument):
    chunks = chunker.chunk(sample_parsed_document, chunk_size=100, chunk_overlap=20)
    assert len(chunks) >= 1
    for chunk in chunks:
        assert chunk.content
        assert chunk.token_count > 0
        assert chunk.chunk_id


def test_chunker_preserves_empty_document(chunker: AdaptiveChunker):
    empty_doc = ParsedDocument(content="", pages=[], metadata={})
    chunks = chunker.chunk(empty_doc)
    assert chunks == []


def test_chunker_section_headings(chunker: AdaptiveChunker):
    text = (
        "SECTION 1. DEFINITIONS\nThis is paragraph one explaining definitions.\n\n"
        "SECTION 2. SCOPE OF OBLIGATIONS\nAll financial entities must retain records for five years.\n\n"
        "SECTION 3. PENALTIES\nViolations will be subject to enforcement action."
    )
    doc = ParsedDocument(
        content=text,
        pages=[PageContent(page_num=1, text=text)],
        metadata={"title": "Test Regulations"},
    )
    chunks = chunker.chunk(doc, chunk_size=50, chunk_overlap=10)
    assert len(chunks) >= 2
    # Verify section titles aren't lost
    headings_found = any("SECTION" in c.content for c in chunks)
    assert headings_found


def test_chunker_metadata_propagation(
    chunker: AdaptiveChunker, sample_parsed_document: ParsedDocument
):
    chunks = chunker.chunk(sample_parsed_document)
    assert chunks[0].metadata.chunk_index == 0
    assert chunks[0].metadata.page_num == 1
