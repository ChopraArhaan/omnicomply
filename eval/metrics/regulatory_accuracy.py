"""Regulatory accuracy metric verifying legal citations, numbers, deadlines, and penalty amounts."""

from __future__ import annotations

import re

from eval.metrics.faithfulness import MetricResult


class RegulatoryAccuracyMetric:
    """Verifies that legal citations (e.g., '17 CFR § 240.10b-5'), statutory numbers,

    deadlines (e.g., '72 hours', '4 business days'), and fines match ground truth.
    """

    def __init__(self, threshold: float = 0.85) -> None:
        self.threshold = threshold

    def score(self, answer: str, ground_truth: str, domain: str = "general") -> MetricResult:
        # Extract critical legal/regulatory features
        citation_pattern = r"(?:\d+\s+(?:CFR|U\.S\.C\.|USC)\s+[§\d\.\w-]+|Article\s+\d+(?:\(\d+\))?|Rule\s+\d+[\w-]*)"
        numeric_pattern = r"\b(?:\d+(?:,\d{3})*|\d+)\s*(?:days|hours|months|years|percent|%|business days|calendar days|EUR|USD|\$|million|billion)\b"

        gt_citations = set(re.findall(citation_pattern, ground_truth, re.IGNORECASE))
        gt_numerics = set(re.findall(numeric_pattern, ground_truth, re.IGNORECASE))

        ans_citations = set(re.findall(citation_pattern, answer, re.IGNORECASE))
        ans_numerics = set(re.findall(numeric_pattern, answer, re.IGNORECASE))

        scores = []
        checks = {}

        # 1. Citation check
        if gt_citations:
            citation_match = len(gt_citations.intersection(ans_citations)) / len(gt_citations)
            scores.append(citation_match)
            checks["citations"] = {
                "expected": list(gt_citations),
                "matched": list(gt_citations.intersection(ans_citations)),
            }

        # 2. Key numbers / deadlines check
        if gt_numerics:
            numeric_match = len(gt_numerics.intersection(ans_numerics)) / len(gt_numerics)
            scores.append(numeric_match)
            checks["numeric_facts"] = {
                "expected": list(gt_numerics),
                "matched": list(gt_numerics.intersection(ans_numerics)),
            }

        # 3. Textual overlap as baseline if no specific citations
        if not scores:
            gt_words = set(re.findall(r"\b\w{4,}\b", ground_truth.lower()))
            ans_words = set(re.findall(r"\b\w{4,}\b", answer.lower()))
            overlap = len(gt_words.intersection(ans_words)) / max(1, len(gt_words))
            scores.append(min(1.0, overlap * 1.3))

        final_score = round(sum(scores) / len(scores), 3)
        passed = final_score >= self.threshold

        return MetricResult(
            name="regulatory_accuracy",
            score=final_score,
            passed=passed,
            threshold=self.threshold,
            reasoning=f"Regulatory domain facts accuracy: {final_score:.1%} in domain {domain}.",
            details=checks,
        )
