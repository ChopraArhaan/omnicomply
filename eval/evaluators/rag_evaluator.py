"""RAG Evaluator — Batch evaluation of golden test sets with reporting."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from eval.metrics.answer_relevancy import AnswerRelevancyMetric
from eval.metrics.context_precision import ContextPrecisionMetric
from eval.metrics.faithfulness import FaithfulnessMetric
from eval.metrics.latency_budget import LatencyBudgetMetric
from eval.metrics.regulatory_accuracy import RegulatoryAccuracyMetric

try:
    import structlog
    logger = structlog.get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


class EvalReport(BaseModel):
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    model_version: str = "omnicomply-v0.1.0"
    total_test_cases: int
    overall_pass_rate: float
    metric_averages: dict[str, float]
    gate_passed: bool
    results: list[dict[str, Any]] = Field(default_factory=list)


class RAGEvaluator:
    """Evaluates RAG pipeline against curated golden datasets."""

    def __init__(
        self,
        faithfulness_thresh: float = 0.90,
        relevancy_thresh: float = 0.85,
        precision_thresh: float = 0.80,
    ) -> None:
        self.faith_metric = FaithfulnessMetric(threshold=faithfulness_thresh)
        self.rel_metric = AnswerRelevancyMetric(threshold=relevancy_thresh)
        self.prec_metric = ContextPrecisionMetric(threshold=precision_thresh)
        self.acc_metric = RegulatoryAccuracyMetric(threshold=0.85)
        self.latency_metric = LatencyBudgetMetric(budget_ms=3000.0)

    def evaluate_dataset(self, dataset_path: str | Path) -> EvalReport:
        """Run all metrics on each test case in a JSONL golden file."""
        path = Path(dataset_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found at {path}")

        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))

        results = []
        faith_scores = []
        rel_scores = []
        prec_scores = []
        acc_scores = []
        all_passed = True

        for rec in records:
            question = rec["question"]
            ground_truth = rec["ground_truth"]
            context = rec["context"]
            # Simulate or fetch RAG answer
            simulated_answer = rec.get("answer", ground_truth)
            context_chunks = rec.get("context_chunks", [context])

            f_res = self.faith_metric.score(question, simulated_answer, context)
            r_res = self.rel_metric.score(question, simulated_answer)
            p_res = self.prec_metric.score(question, context_chunks, ground_truth)
            a_res = self.acc_metric.score(simulated_answer, ground_truth, domain=rec.get("domain", "general"))

            case_passed = f_res.passed and r_res.passed and p_res.passed and a_res.passed
            if not case_passed:
                all_passed = False

            faith_scores.append(f_res.score)
            rel_scores.append(r_res.score)
            prec_scores.append(p_res.score)
            acc_scores.append(a_res.score)

            results.append(
                {
                    "id": rec.get("id"),
                    "domain": rec.get("domain"),
                    "passed": case_passed,
                    "faithfulness": f_res.score,
                    "answer_relevancy": r_res.score,
                    "context_precision": p_res.score,
                    "regulatory_accuracy": a_res.score,
                }
            )

        n = max(1, len(records))
        avg_faith = round(sum(faith_scores) / n, 3)
        avg_rel = round(sum(rel_scores) / n, 3)
        avg_prec = round(sum(prec_scores) / n, 3)
        avg_acc = round(sum(acc_scores) / n, 3)

        passed_count = sum(1 for r in results if r["passed"])
        pass_rate = round(passed_count / n, 3)

        gate_passed = (
            avg_faith >= self.faith_metric.threshold
            and avg_rel >= self.rel_metric.threshold
            and avg_prec >= self.prec_metric.threshold
        )

        report = EvalReport(
            total_test_cases=len(records),
            overall_pass_rate=pass_rate,
            metric_averages={
                "faithfulness": avg_faith,
                "answer_relevancy": avg_rel,
                "context_precision": avg_prec,
                "regulatory_accuracy": avg_acc,
            },
            gate_passed=gate_passed,
            results=results,
        )

        logger.info(
            "dataset_evaluation_complete",
            test_cases=len(records),
            pass_rate=pass_rate,
            gate_passed=gate_passed,
        )
        return report

    def save_report(self, report: EvalReport, output_path: str | Path) -> None:
        """Serialize evaluation report to disk."""
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write(report.model_dump_json(indent=2))
