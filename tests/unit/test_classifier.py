"""Unit tests for RegulatoryClassifier."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.analysis.classifier import (
    ClassificationResult,
    RegulatoryClassifier,
    UrgencyLevel,
)


@pytest.fixture
def mock_pipeline():
    mock = MagicMock()
    mock.return_value = {
        "labels": ["financial", "technology", "trade"],
        "scores": [0.85, 0.10, 0.05],
    }
    return mock


def test_classify_domain_with_mock(mock_pipeline):
    with patch("src.analysis.classifier.pipeline", return_value=mock_pipeline):
        classifier = RegulatoryClassifier()
        result = classifier.classify_domain(
            "SEC requires Form 10-K disclosures regarding public offerings."
        )
        assert isinstance(result, ClassificationResult)
        assert result.domain in [
            "financial",
            "healthcare",
            "environmental",
            "data_privacy",
            "labor",
            "trade",
            "technology",
        ]
        assert 0.0 <= result.confidence <= 1.0


def test_classify_urgency_critical(mock_pipeline):
    with patch("src.analysis.classifier.pipeline", return_value=mock_pipeline):
        classifier = RegulatoryClassifier()
        # Direct keyword triggers
        urgency = classifier.classify_urgency(
            "EMERGENCY ORDER: All institutions shall immediately cease and desist with immediate effect."
        )
        assert urgency in [UrgencyLevel.CRITICAL, UrgencyLevel.HIGH]


def test_classify_document_type(mock_pipeline):
    with patch("src.analysis.classifier.pipeline", return_value=mock_pipeline):
        classifier = RegulatoryClassifier()
        doc_type = classifier.classify_document_type(
            "Final rule regarding capital adequacy standards."
        )
        assert doc_type in ["rule", "guidance", "enforcement_action", "amendment", "notice"]
