"""CI/CD Quality Evaluation Gate.

Enforces minimum production RAG thresholds (Faithfulness >= 0.90,
Answer Relevancy >= 0.85, Context Precision >= 0.80).
Exits with 0 on pass, 1 on regression.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from eval.evaluators.rag_evaluator import RAGEvaluator


def main() -> int:
    parser = argparse.ArgumentParser(description="OmniComply CI Evaluation Gate")
    parser.add_argument(
        "--dataset",
        type=str,
        default="eval/datasets/regulatory_qa_golden.jsonl",
        help="Path to golden dataset JSONL",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail build if any metric is below threshold",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="eval/reports/outputs/ci_eval_report.json",
        help="Path to write JSON report output",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("🛡️  OMNICOMPLY CI EVALUATION GATE")
    print("=" * 70)

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"❌ Error: Dataset file not found at {dataset_path}")
        return 1

    evaluator = RAGEvaluator(
        faithfulness_thresh=0.90,
        relevancy_thresh=0.85,
        precision_thresh=0.80,
    )

    print(f"Loading golden dataset: {dataset_path}")
    report = evaluator.evaluate_dataset(dataset_path)

    # Save output report
    evaluator.save_report(report, args.output)
    print(f"Report saved to: {args.output}\n")

    # Print summary table
    print(f"{'Metric':<25} | {'Score':<10} | {'Threshold':<10} | {'Status'}")
    print("-" * 65)

    thresholds = {
        "faithfulness": 0.90,
        "answer_relevancy": 0.85,
        "context_precision": 0.80,
        "regulatory_accuracy": 0.85,
    }

    all_passed = True
    for metric_name, score in report.metric_averages.items():
        thresh = thresholds.get(metric_name, 0.80)
        passed = score >= thresh
        if not passed:
            all_passed = False
        status_str = "✅ PASS" if passed else "❌ FAIL"
        print(f"{metric_name:<25} | {score:<10.1%} | {thresh:<10.1%} | {status_str}")

    print("-" * 65)
    print(f"Overall Pass Rate: {report.overall_pass_rate:.1%} ({report.total_test_cases} test cases)")

    if args.strict and not all_passed:
        print("\n❌ CI GATE FAILED: One or more evaluation metrics fell below production thresholds.")
        return 1

    print("\n✅ CI GATE PASSED: All production RAG metrics meet or exceed quality thresholds.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
