"""Nightly Regression Suite for continuous RAG quality monitoring."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from eval.evaluators.rag_evaluator import RAGEvaluator


def main() -> int:
    print("=" * 70)
    print("🌙  OMNICOMPLY NIGHTLY REGRESSION SUITE")
    print("=" * 70)

    dataset_path = Path("eval/datasets/regulatory_qa_golden.jsonl")
    output_dir = Path("eval/reports/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_dir = Path("eval/reports/baseline")
    baseline_path = baseline_dir / "nightly_eval_report.json"

    baseline_report = None
    if baseline_path.exists():
        try:
            with open(baseline_path, encoding="utf-8") as f:
                baseline_report = json.load(f)
            print(f"📊 Loaded prior baseline from {baseline_path}")
        except Exception as e:
            print(f"[WARN] Could not parse baseline report ({e}). Treating as initial run.")
    else:
        print(
            "[INFO] No prior baseline artifact found (first run or expired retention). "
            "Skipping baseline delta comparison."
        )

    evaluator = RAGEvaluator()
    report = evaluator.evaluate_dataset(dataset_path)

    report_path = output_dir / "nightly_eval_report.json"
    evaluator.save_report(report, report_path)

    # Compute comparison with baseline if available
    delta_lines = []
    if baseline_report and "metric_averages" in baseline_report:
        base_avgs = baseline_report["metric_averages"]
        for k, v in report.metric_averages.items():
            prev = base_avgs.get(k, v)
            diff = v - prev
            delta_lines.append(
                f"- {k.replace('_', ' ').title()}: {v:.1%} (Δ {diff:+.1%} vs baseline)"
            )
    else:
        for k, v in report.metric_averages.items():
            delta_lines.append(f"- {k.replace('_', ' ').title()}: {v:.1%}")

    delta_text = "\n".join(delta_lines)

    # Write summary for GitHub Action notifications
    summary_text = f"""Nightly Evaluation Regression Summary:
Timestamp: {report.timestamp}
Test Cases Evaluated: {report.total_test_cases}
Overall Pass Rate: {report.overall_pass_rate:.1%}

Metric Averages:
{delta_text}

Gate Status: {"PASSED" if report.gate_passed else "REGRESSION DETECTED"}
"""
    with open(output_dir / "regression_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary_text)

    print(summary_text)
    return 0 if report.gate_passed else 1


if __name__ == "__main__":
    sys.exit(main())
