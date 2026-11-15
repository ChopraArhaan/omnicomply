"""Pytest fixtures and mock configurations for OmniComply."""

from __future__ import annotations

import pytest

from src.config.settings import Settings
from src.ingestion.pdf_parser import PageContent, ParsedDocument


@pytest.fixture
def mock_settings() -> Settings:
    """Provide deterministic test settings."""
    return Settings(
        postgres_host="localhost",
        postgres_port=5432,
        postgres_db="omnicomply_test",
        postgres_user="test",
        postgres_password="password",
        redis_host="localhost",
        redis_port=6379,
    )


@pytest.fixture
def sample_regulatory_text() -> str:
    """Realistic regulatory snippet for testing."""
    return (
        "SECTION 105. MANDATORY CYBERSECURITY DISCLOSURE.\n"
        "(a) In General. Each covered entity shall report any material cybersecurity incident "
        "to the Commission within four (4) business days after determining materiality.\n"
        "(b) Civil Penalties. Failure to comply shall subject the entity to civil penalties of "
        "up to $1,000,000 per violation.\n"
        "(c) Compliance Date. This rule takes effect within 30 days of publication."
    )


@pytest.fixture
def sample_parsed_document(sample_regulatory_text: str) -> ParsedDocument:
    """Sample parsed document model."""
    return ParsedDocument(
        content=sample_regulatory_text,
        pages=[PageContent(page_num=1, text=sample_regulatory_text)],
        metadata={"title": "SEC Cybersecurity Incident Guidance", "num_pages": 1},
    )
