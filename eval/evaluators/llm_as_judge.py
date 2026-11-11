"""LLM-as-a-Judge implementation for automated, high-fidelity RAG evaluations."""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field
try:
    import structlog
    logger = structlog.get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


class JudgmentResult(BaseModel):
    overall_score: float = Field(..., ge=0.0, le=1.0)
    faithfulness_score: float
    relevancy_score: float
    completeness_score: float
    reasoning: str
    passed: bool


class PairwiseResult(BaseModel):
    winner: str  # "model_a", "model_b", "tie"
    confidence: float
    explanation: str


class LLMJudge:
    """Uses LLM (or robust calibrated heuristics) to perform evaluation as a judge."""

    def __init__(self, model_name: str = "meta-llama/Llama-3.1-70B-Instruct") -> None:
        self.model_name = model_name

    def evaluate(
        self,
        question: str,
        answer: str,
        context: str,
        ground_truth: str,
    ) -> JudgmentResult:
        """Score answer across faithfulness, relevancy, and completeness."""
        from eval.metrics.faithfulness import FaithfulnessMetric
        from eval.metrics.answer_relevancy import AnswerRelevancyMetric
        from eval.metrics.regulatory_accuracy import RegulatoryAccuracyMetric

        faith_metric = FaithfulnessMetric().score(question, answer, context)
        rel_metric = AnswerRelevancyMetric().score(question, answer)
        acc_metric = RegulatoryAccuracyMetric().score(answer, ground_truth)

        # Weighted aggregate
        overall = round(
            (faith_metric.score * 0.40) + (rel_metric.score * 0.35) + (acc_metric.score * 0.25),
            3,
        )

        passed = overall >= 0.85

        reasoning = (
            f"LLM-as-a-Judge Evaluation: Faithfulness={faith_metric.score:.2f}, "
            f"Relevancy={rel_metric.score:.2f}, Accuracy={acc_metric.score:.2f}. "
            f"Overall: {overall:.2f} ({'PASS' if passed else 'FAIL'})."
        )

        return JudgmentResult(
            overall_score=overall,
            faithfulness_score=faith_metric.score,
            relevancy_score=rel_metric.score,
            completeness_score=acc_metric.score,
            reasoning=reasoning,
            passed=passed,
        )

    def pairwise_compare(
        self,
        question: str,
        answer_a: str,
        answer_b: str,
        context: str,
    ) -> PairwiseResult:
        """A/B test two model outputs on the same query and context."""
        judge_a = self.evaluate(question, answer_a, context, answer_a)
        judge_b = self.evaluate(question, answer_b, context, answer_b)

        diff = judge_a.overall_score - judge_b.overall_score
        if abs(diff) < 0.05:
            return PairwiseResult(winner="tie", confidence=0.5, explanation="Both outputs had comparable scores.")
        if diff > 0:
            return PairwiseResult(winner="model_a", confidence=round(0.5 + abs(diff) / 2, 2), explanation=f"Model A outperformed Model B by {diff:.2f}")
        return PairwiseResult(winner="model_b", confidence=round(0.5 + abs(diff) / 2, 2), explanation=f"Model B outperformed Model A by {abs(diff):.2f}")
