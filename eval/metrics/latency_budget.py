"""Latency budget evaluation metric."""

from __future__ import annotations

from eval.metrics.faithfulness import MetricResult


class LatencyBudgetMetric:
    """Verifies that query response latency remains within SLA budget (P95 < 3000ms)."""

    def __init__(self, budget_ms: float = 3000.0) -> None:
        self.budget_ms = budget_ms

    def score(self, latency_ms: float) -> MetricResult:
        passed = latency_ms <= self.budget_ms
        # Linear degradation score: 1.0 at 0ms down to 0.0 at 2x budget
        ratio = max(0.0, 1.0 - (latency_ms / (2.0 * self.budget_ms)))
        score = round(min(1.0, ratio), 3)

        return MetricResult(
            name="latency_budget",
            score=score,
            passed=passed,
            threshold=1.0,
            reasoning=f"Latency {latency_ms:.1f}ms against {self.budget_ms:.1f}ms budget. SLA: {'MET' if passed else 'BREACHED'}.",
            details={"latency_ms": latency_ms, "budget_ms": self.budget_ms},
        )
