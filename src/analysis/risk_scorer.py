"""Compliance Risk Scorer — Multi-dimensional regulatory risk assessment."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, ClassVar

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


class RiskLevel(StrEnum):
    """Categorical risk rating."""

    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class RiskFactor(BaseModel):
    """An individual contributor to the overall risk score."""

    name: str
    score: float = Field(..., ge=0.0, le=100.0)
    weight: float = Field(..., ge=0.0, le=1.0)
    description: str
    evidence: list[str] = Field(default_factory=list)


class RiskAssessment(BaseModel):
    """Comprehensive risk assessment for a regulatory document or policy."""

    overall_score: float = Field(..., ge=0.0, le=100.0)
    risk_level: RiskLevel
    risk_factors: list[RiskFactor] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    assessed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)


class PortfolioRisk(BaseModel):
    """Aggregated risk across multiple documents/policies."""

    average_score: float
    max_score: float
    risk_distribution: dict[str, int]
    high_risk_documents: list[str]
    total_assessed: int


class ComplianceRiskScorer:
    """Evaluates regulatory compliance risk from text and extracted entities."""

    # High-risk signal keywords
    PENALTY_PATTERNS: ClassVar[list[str]] = [
        r"\$\s*\d+(?:,\d{3})*(?:\.\d+)?\s*(?:million|billion|thousand)?",
        r"civil\s+penalt(?:y|ies)",
        r"criminal\s+liabilit(?:y|ies)",
        r"fine[s]?\s+(?:up\s+to|not\s+exceeding|of)",
        r"imprisonment",
        r"injunction",
        r"cease\s+and\s+desist",
    ]

    ENFORCEMENT_PATTERNS: ClassVar[list[str]] = [
        r"enforcement\s+action",
        r"mandatory\s+compliance",
        r"shall\s+(?:immediately|forthwith|promptly)",
        r"strictly\s+prohibited",
        r"revocation\s+of\s+license",
        r"suspension",
    ]

    DEADLINE_PATTERNS: ClassVar[list[str]] = [
        r"within\s+\d+\s+days",
        r"effective\s+immediately",
        r"no\s+later\s+than",
        r"deadline\s+(?:is|of)",
        r"compliance\s+date",
    ]

    def __init__(self, weights: dict[str, float] | None = None) -> None:
        self.weights = weights or {
            "penalties": 0.30,
            "enforcement": 0.25,
            "deadlines": 0.20,
            "scope": 0.15,
            "complexity": 0.10,
        }
        # Normalize weights
        total = sum(self.weights.values())
        self.weights = {k: v / total for k, v in self.weights.items()}

    def score_document(
        self,
        text: str,
        entities: list[dict[str, Any]] | None = None,
        domain: str = "general",
    ) -> RiskAssessment:
        """Calculate multi-dimensional compliance risk score for a document."""
        entities = entities or []
        factors: list[RiskFactor] = []

        # 1. Penalty risk
        penalty_matches = self._find_matches(text, self.PENALTY_PATTERNS)
        penalty_score = min(100.0, len(penalty_matches) * 25.0)
        factors.append(
            RiskFactor(
                name="Penalty Exposure",
                score=penalty_score,
                weight=self.weights["penalties"],
                description=f"Detected {len(penalty_matches)} penalty or monetary liability mentions",
                evidence=penalty_matches[:5],
            )
        )

        # 2. Enforcement action language
        enforcement_matches = self._find_matches(text, self.ENFORCEMENT_PATTERNS)
        enforcement_score = min(100.0, len(enforcement_matches) * 20.0)
        factors.append(
            RiskFactor(
                name="Enforcement Severity",
                score=enforcement_score,
                weight=self.weights["enforcement"],
                description=f"Detected {len(enforcement_matches)} strict enforcement provisions",
                evidence=enforcement_matches[:5],
            )
        )

        # 3. Deadline urgency
        deadline_matches = self._find_matches(text, self.DEADLINE_PATTERNS)
        deadline_score = min(100.0, len(deadline_matches) * 25.0)
        factors.append(
            RiskFactor(
                name="Implementation Urgency",
                score=deadline_score,
                weight=self.weights["deadlines"],
                description=f"Detected {len(deadline_matches)} timeline/deadline constraints",
                evidence=deadline_matches[:5],
            )
        )

        # 4. Jurisdiction & entity scope
        org_entities = [e for e in entities if e.get("label") == "ORGANIZATION"]
        jurisdiction_entities = [e for e in entities if e.get("label") == "JURISDICTION"]
        scope_score = min(100.0, (len(org_entities) * 10.0) + (len(jurisdiction_entities) * 15.0))
        factors.append(
            RiskFactor(
                name="Regulatory Scope",
                score=scope_score,
                weight=self.weights["scope"],
                description=f"Impacts {len(org_entities)} organizations across {len(jurisdiction_entities)} jurisdictions",
                evidence=[e.get("text", "") for e in (org_entities + jurisdiction_entities)[:5]],
            )
        )

        # 5. Complexity (length + entity density)
        word_count = len(text.split())
        complexity_score = min(100.0, (word_count / 500.0) * 10.0 + len(entities) * 2.0)
        factors.append(
            RiskFactor(
                name="Regulatory Complexity",
                score=complexity_score,
                weight=self.weights["complexity"],
                description=f"Document length: {word_count} words with {len(entities)} extracted entities",
                evidence=[],
            )
        )

        # Calculate overall weighted score
        overall_score = sum(f.score * f.weight for f in factors)
        overall_score = round(min(100.0, max(0.0, overall_score)), 2)

        # Map to categorical level
        risk_level = self._map_to_level(overall_score)

        # Generate actionable recommendations
        recommendations = self._generate_recommendations(factors, domain)

        return RiskAssessment(
            overall_score=overall_score,
            risk_level=risk_level,
            risk_factors=factors,
            recommendations=recommendations,
            metadata={"domain": domain, "entity_count": len(entities)},
        )

    def score_portfolio(self, assessments: list[RiskAssessment]) -> PortfolioRisk:
        """Aggregate multiple document risk assessments into a portfolio view."""
        if not assessments:
            return PortfolioRisk(
                average_score=0.0,
                max_score=0.0,
                risk_distribution={level.value: 0 for level in RiskLevel},
                high_risk_documents=[],
                total_assessed=0,
            )

        scores = [a.overall_score for a in assessments]
        distribution = {level.value: 0 for level in RiskLevel}
        high_risk_docs: list[str] = []

        for idx, a in enumerate(assessments):
            distribution[a.risk_level.value] += 1
            if a.risk_level in (RiskLevel.CRITICAL, RiskLevel.HIGH):
                doc_id = a.metadata.get("document_id", f"doc_{idx}")
                high_risk_docs.append(str(doc_id))

        return PortfolioRisk(
            average_score=round(sum(scores) / len(scores), 2),
            max_score=round(max(scores), 2),
            risk_distribution=distribution,
            high_risk_documents=high_risk_docs,
            total_assessed=len(assessments),
        )

    def _find_matches(self, text: str, patterns: list[str]) -> list[str]:
        """Find unique regex matches across text."""
        matches: list[str] = []
        for pattern in patterns:
            found = re.findall(pattern, text, re.IGNORECASE)
            for m in found:
                clean = m.strip() if isinstance(m, str) else str(m)
                if clean and clean not in matches:
                    matches.append(clean)
        return matches

    def _map_to_level(self, score: float) -> RiskLevel:
        """Map numeric score to RiskLevel."""
        if score >= 75.0:
            return RiskLevel.CRITICAL
        if score >= 50.0:
            return RiskLevel.HIGH
        if score >= 25.0:
            return RiskLevel.MEDIUM
        return RiskLevel.LOW

    def _generate_recommendations(self, factors: list[RiskFactor], domain: str) -> list[str]:
        """Generate prescriptive recommendations based on high-risk factors."""
        recs: list[str] = []
        for factor in factors:
            if factor.score >= 50.0:
                if factor.name == "Penalty Exposure":
                    recs.append(
                        "Conduct immediate legal review of liability and penalty provisions."
                    )
                elif factor.name == "Enforcement Severity":
                    recs.append(
                        "Prioritize gap analysis against mandatory operational requirements."
                    )
                elif factor.name == "Implementation Urgency":
                    recs.append(
                        "Establish cross-functional compliance task force for rapid remediation."
                    )
                elif factor.name == "Regulatory Scope":
                    recs.append(
                        f"Audit affected subsidiary policies across referenced jurisdictions ({domain})."
                    )
        if not recs:
            recs.append("Standard periodic compliance monitoring recommended.")
        return recs
