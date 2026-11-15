"""Integration tests for IngestionPipeline end-to-end processing."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from src.ingestion.pipeline import IngestionPipeline


@pytest.mark.integration
@pytest.mark.asyncio
async def test_ingest_text_file():
    pipeline = IngestionPipeline()
    sample_content = (
        "SECTION 1. SCOPE AND JURISDICTION\n"
        "This regulation applies to all registered investment advisers with greater than "
        "$100,000,000 in regulatory assets under management.\n\n"
        "SECTION 2. ANNUAL REPORTING\n"
        "Advisers must submit Form ADV annually within 90 days of fiscal year end."
    )

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as tmp:
        tmp.write(sample_content)
        tmp_path = Path(tmp.name)

    try:
        processed = await pipeline.ingest(tmp_path)
        assert processed.document_id
        assert processed.source_type == "text"
        assert len(processed.chunks) >= 1
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
