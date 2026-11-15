"""Context Precision metric measuring retrieval ranking quality."""

from __future__ import annotations

import re

from eval.metrics.faithfulness import MetricResult


class ContextPrecisionMetric:
    """Measures whether the relevant chunks appear at the top ranks of retrieved results (Mean Average Precision)."""

    def __init__(self, threshold: float = 0.80) -> None:
        self.threshold = threshold

    def score(
        self,
        question: str,
        context_chunks: list[str],
        ground_truth: str,
    ) -> MetricResult:
        if not context_chunks:
            return MetricResult(
                name="context_precision",
                score=0.0,
                passed=False,
                threshold=self.threshold,
                reasoning="No context chunks were retrieved.",
            )

        gt_tokens = set(re.findall(r"\b\w{3,}\b", ground_truth.lower()))
        relevant_at_k = []

        # Determine relevance of each chunk to the ground truth
        for _idx, chunk in enumerate(context_chunks):
            chunk_tokens = set(re.findall(r"\b\w{3,}\b", chunk.lower()))
            overlap = len(gt_tokens.intersection(chunk_tokens)) / max(1, len(gt_tokens))
            is_relevant = overlap >= 0.40
            relevant_at_k.append(is_relevant)

        # Compute Mean Average Precision @ K
        running_relevant = 0
        precisions = []
        for k, rel in enumerate(relevant_at_k, 1):
            if rel:
                running_relevant += 1
                precisions.append(running_relevant / k)

        score = round(
            sum(precisions) / max(1, running_relevant) if running_relevant > 0 else 0.0, 3
        )
        passed = score >= self.threshold

        return MetricResult(
            name="context_precision",
            score=score,
            passed=passed,
            threshold=self.threshold,
            reasoning=(
                f"Precision@K: {running_relevant}/{len(context_chunks)} chunks relevant. "
                f"MAP Score: {score:.1%} (threshold: {self.threshold:.1%})."
            ),
            details={"relevant_ranks": [i + 1 for i, r in enumerate(relevant_at_k) if r]},
        )
