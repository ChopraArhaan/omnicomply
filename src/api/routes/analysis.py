"""Regulatory Analysis API routes for risk scoring, diffing, and forecasting."""

from __future__ import annotations

from typing import Any

import structlog
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.analysis.classifier import RegulatoryClassifier
from src.analysis.forecaster import RegulatoryForecaster
from src.analysis.risk_scorer import ComplianceRiskScorer, RiskAssessment
from src.analysis.semantic_diff import SemanticDiff, SemanticDiffEngine

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/v1/analysis", tags=["Regulatory Analysis"])

# Instantiate service singletons
classifier = RegulatoryClassifier()
risk_scorer = ComplianceRiskScorer()
diff_engine = SemanticDiffEngine()
forecaster = RegulatoryForecaster()


class RiskAssessmentRequest(BaseModel):
    text: str = Field(..., description="Regulatory text or internal policy to evaluate")
    domain: str = Field("general", description="Regulatory domain (e.g. financial, privacy)")


class SemanticDiffRequest(BaseModel):
    base_text: str = Field(..., description="Existing regulation or baseline policy text")
    updated_text: str = Field(..., description="Proposed rule amendment or new policy text")


class ForecastRequest(BaseModel):
    historical_scores: list[dict[str, Any]] = Field(..., description="Historical risk score series")
    horizon_days: int = Field(30, ge=7, le=180)


@router.post("/risk-assessment", response_model=RiskAssessment)
async def assess_risk(payload: RiskAssessmentRequest) -> RiskAssessment:
    """Calculate multi-dimensional risk score from regulatory text."""
    try:
        assessment = risk_scorer.score_document(
            text=payload.text,
            domain=payload.domain,
        )
        return assessment
    except Exception as exc:
        logger.error("risk_assessment_failed", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/semantic-diff", response_model=SemanticDiff)
async def compare_regulations(payload: SemanticDiffRequest) -> SemanticDiff:
    """Detect changes and contradictions between regulatory text versions."""
    try:
        diff_result = diff_engine.compute_diff(
            old_doc=payload.base_text,
            new_doc=payload.updated_text,
        )
        return diff_result
    except Exception as exc:
        logger.error("semantic_diff_failed", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/forecast")
async def forecast_regulatory_trends(payload: ForecastRequest) -> dict[str, Any]:
    """Forecast future compliance risk and detect anomalies."""
    try:
        forecast = forecaster.forecast_risk(
            historical_scores=payload.historical_scores,
            horizon=payload.horizon_days,
        )
        anomalies = forecaster.detect_anomalies(payload.historical_scores)
        return {
            "forecast": forecast.model_dump(),
            "anomalies": [a.model_dump() for a in anomalies],
        }
    except Exception as exc:
        logger.error("forecasting_failed", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/risk-dashboard")
async def get_risk_dashboard() -> dict[str, Any]:
    """Retrieve high-level compliance posture and risk metrics."""
    return {
        "overall_posture": "MONITORED",
        "average_risk_score": 42.5,
        "high_risk_alerts": 3,
        "monitored_jurisdictions": ["SEC", "FDA", "EU AI Act", "GDPR", "EPA"],
        "publishing_velocity_trend": "INCREASING",
        "total_documents_indexed": 1420,
    }
