"""Unit tests for ComplianceRiskScorer."""

from __future__ import annotations

import pytest

from src.analysis.risk_scorer import (
    ComplianceRiskScorer,
    RiskAssessment,
    RiskLevel,
)


@pytest.fixture
def risk_scorer() -> ComplianceRiskScorer:
    return ComplianceRiskScorer()


def test_score_document_baseline(risk_scorer: ComplianceRiskScorer):
    text = "General guidance on recordkeeping best practices."
    assessment = risk_scorer.score_document(text, domain="general")

    assert isinstance(assessment, RiskAssessment)
    assert 0.0 <= assessment.overall_score <= 100.0
    assert assessment.risk_level in [RiskLevel.LOW, RiskLevel.MEDIUM]
    assert len(assessment.recommendations) >= 1


def test_score_document_high_penalty(risk_scorer: ComplianceRiskScorer):
    text = (
        "Violations shall be subject to civil penalties of up to $50,000,000 and imprisonment. "
        "Enforcement action will be taken immediately against non-compliant entities."
    )
    assessment = risk_scorer.score_document(text, domain="financial")

    assert assessment.overall_score >= 40.0
    assert assessment.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]
    # Check that recommendation mentions penalties
    assert any(
        "penalty" in r.lower() or "liability" in r.lower() for r in assessment.recommendations
    )


def test_portfolio_risk_aggregation(risk_scorer: ComplianceRiskScorer):
    doc1 = risk_scorer.score_document("Standard guidance report.")
    doc2 = risk_scorer.score_document("Critical emergency order with $10M fines immediately.")

    portfolio = risk_scorer.score_portfolio([doc1, doc2])
    assert portfolio.total_assessed == 2
    assert portfolio.max_score >= portfolio.average_score
    assert sum(portfolio.risk_distribution.values()) == 2
