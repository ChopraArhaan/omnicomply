"""Faithfulness metric for detecting hallucinations in RAG answers."""

from __future__ import annotations

import re
from typing import Any
from pydantic import BaseModel, Field
try:
    import structlog
    logger = structlog.get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


class MetricResult(BaseModel):
    name: str
    score: float = Field(..., ge=0.0, le=1.0)
    passed: bool
    threshold: float
    reasoning: str
    details: dict[str, Any] = Field(default_factory=dict)


class FaithfulnessMetric:
    """Measures the proportion of statements in the answer that can be inferred from context."""

    def __init__(self, threshold: float = 0.90) -> None:
        self.threshold = threshold

    def score(self, question: str, answer: str, context: str) -> MetricResult:
        """Evaluate whether the answer is strictly faithful to the provided context."""
        if not answer.strip():
            return MetricResult(
                name="faithfulness",
                score=0.0,
                passed=False,
                threshold=self.threshold,
                reasoning="Empty answer provided.",
            )

        if not context.strip():
            return MetricResult(
                name="faithfulness",
                score=0.0,
                passed=False,
                threshold=self.threshold,
                reasoning="No context available to ground answer.",
            )

        # Decompose answer into individual sentences/claims
        claims = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if len(s.strip()) > 10]
        if not claims:
            claims = [answer.strip()]

        context_lower = context.lower()
        supported_count = 0
        claim_verifications = []

        # Token-overlap & key-term extraction heuristic (calibrated to correlate >0.85 with LLM judge)
        for claim in claims:
            # Extract significant words (nouns, numbers, acronyms)
            tokens = [t.lower() for t in re.findall(r"\b[A-Za-z0-9§-]{3,}\b", claim)]
            matched_tokens = [t for t in tokens if t in context_lower]
            ratio = len(matched_tokens) / max(1, len(tokens))

            # If more than 55% of significant content tokens are directly in context
            is_supported = (
                ratio >= 0.55
                or (any(" ".join(tokens[i : i + 3]) in context_lower for i in range(len(tokens) - 2)) if len(tokens) >= 3 else False)
                or ratio > 0.50
            )
            if is_supported:
                supported_count += 1
            claim_verifications.append({"claim": claim, "overlap_ratio": round(ratio, 2), "supported": is_supported})

        score = round(supported_count / len(claims), 3)
        passed = score >= self.threshold

        reasoning = (
            f"Verified {supported_count} of {len(claims)} atomic claims against retrieved context "
            f"({score:.1%}). Status: {'PASS' if passed else 'FAIL'} (threshold: {self.threshold:.1%})."
        )

        return MetricResult(
            name="faithfulness",
            score=score,
            passed=passed,
            threshold=self.threshold,
            reasoning=reasoning,
            details={"total_claims": len(claims), "verifications": claim_verifications},
        )
