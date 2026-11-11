"""Nightly Regression Suite for continuous RAG quality monitoring."""

from __future__ import annotations

import json
from pathlib import Path
import sys

from eval.evaluators.rag_evaluator import RAGEvaluator


def main() -> int:
    print("=" * 70)
    print("🌙  OMNICOMPLY NIGHTLY REGRESSION SUITE")
    print("=" * 70)

    dataset_path = Path("eval/datasets/regulatory_qa_golden.jsonl")
    output_dir = Path("eval/reports/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    evaluator = RAGEvaluator()
    report = evaluator.evaluate_dataset(dataset_path)

    report_path = output_dir / "nightly_eval_report.json"
    evaluator.save_report(report, report_path)

    # Write summary for GitHub Action notifications
    summary_text = f"""Nightly Evaluation Regression Summary:
Timestamp: {report.timestamp}
Test Cases Evaluated: {report.total_test_cases}
Overall Pass Rate: {report.overall_pass_rate:.1%}

Metric Averages:
- Faithfulness: {report.metric_averages.get('faithfulness', 0.0):.1%}
- Answer Relevancy: {report.metric_averages.get('answer_relevancy', 0.0):.1%}
- Context Precision: {report.metric_averages.get('context_precision', 0.0):.1%}
- Regulatory Accuracy: {report.metric_averages.get('regulatory_accuracy', 0.0):.1%}

Gate Status: {'PASSED' if report.gate_passed else 'REGRESSION DETECTED'}
"""
    with open(output_dir / "regression_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary_text)

    print(summary_text)
    return 0 if report.gate_passed else 1


if __name__ == "__main__":
    sys.exit(main())
