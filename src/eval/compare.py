"""Comparison utilities for baseline vs tuned model evaluation.

Generates comparison reports and visualizations.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from .metrics import EvalReport, calculate_improvement
from .runner import EvalConfig, Evaluator, load_test_cases


def compare_models(
    model_name: str,
    adapter_path: str,
    test_file: Path,
    output_dir: Path,
    max_tokens: int = 50,
    consistency_runs: int = 3,
    verbose: bool = True,
) -> dict:
    """Compare baseline model vs fine-tuned model.

    Args:
        model_name: HuggingFace model name
        adapter_path: Path to LoRA adapter
        test_file: Path to test.jsonl
        output_dir: Directory to save results
        max_tokens: Maximum tokens to generate
        consistency_runs: Number of runs for consistency
        verbose: Print progress

    Returns:
        Comparison dictionary
    """
    test_cases = load_test_cases(test_file)

    if verbose:
        print(f"Loaded {len(test_cases)} test cases")
        print("")

    # Evaluate baseline
    if verbose:
        print("=" * 60)
        print("Evaluating BASELINE model...")
        print("=" * 60)

    baseline_config = EvalConfig(
        model_name=model_name,
        adapter_path=None,
        max_tokens=max_tokens,
        consistency_runs=consistency_runs,
        verbose=verbose,
    )
    baseline_evaluator = Evaluator(baseline_config)
    baseline_report = baseline_evaluator.run_evaluation(test_cases)

    if verbose:
        print("")
        print("=" * 60)
        print("Evaluating ENGRAM-TUNED model...")
        print("=" * 60)

    # Evaluate tuned model
    tuned_config = EvalConfig(
        model_name=model_name,
        adapter_path=adapter_path,
        max_tokens=max_tokens,
        consistency_runs=consistency_runs,
        verbose=verbose,
    )
    tuned_evaluator = Evaluator(tuned_config)
    tuned_report = tuned_evaluator.run_evaluation(test_cases)

    # Calculate improvements
    improvement = calculate_improvement(baseline_report, tuned_report)

    # Build comparison report
    comparison = {
        "timestamp": datetime.now().isoformat(),
        "model_name": model_name,
        "adapter_path": adapter_path,
        "test_file": str(test_file),
        "test_examples": len(test_cases),
        "improvement": improvement,
        "baseline": baseline_report.to_dict(),
        "tuned": tuned_report.to_dict(),
    }

    # Save individual reports
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "baseline.json", "w") as f:
        json.dump(baseline_report.to_dict(), f, indent=2)

    with open(output_dir / "tuned.json", "w") as f:
        json.dump(tuned_report.to_dict(), f, indent=2)

    with open(output_dir / "comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)

    if verbose:
        print_comparison_summary(comparison)

    return comparison


def print_comparison_summary(comparison: dict):
    """Print a human-readable comparison summary."""
    imp = comparison["improvement"]

    print("")
    print("=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print("")

    # Accuracy
    acc = imp["accuracy"]
    acc_arrow = "" if acc["absolute_change"] >= 0 else ""
    print(f"ACCURACY:")
    print(f"  Baseline:     {acc['baseline']:.2%}")
    print(f"  Engram-tuned: {acc['tuned']:.2%}")
    print(f"  Change:       {acc_arrow}{acc['absolute_change']:+.2%} ({acc['relative_change_pct']:+.1f}%)")
    print("")

    # Consistency
    cons = imp["consistency"]
    cons_arrow = "" if cons["absolute_change"] >= 0 else ""
    print(f"CONSISTENCY:")
    print(f"  Baseline:     {cons['baseline']:.2%}")
    print(f"  Engram-tuned: {cons['tuned']:.2%}")
    print(f"  Change:       {cons_arrow}{cons['absolute_change']:+.2%} ({cons['relative_change_pct']:+.1f}%)")
    print("")

    # Latency
    lat = imp["latency_ms"]
    lat_arrow = "" if lat["change_ms"] <= 0 else ""
    print(f"LATENCY:")
    print(f"  Baseline:     {lat['baseline']:.1f}ms")
    print(f"  Engram-tuned: {lat['tuned']:.1f}ms")
    print(f"  Change:       {lat_arrow}{lat['change_ms']:+.1f}ms")
    print("")

    # Per-category breakdown
    print("BY CATEGORY:")
    baseline_cats = comparison["baseline"]["by_category"]
    tuned_cats = comparison["tuned"]["by_category"]

    for cat in sorted(baseline_cats.keys()):
        if cat in tuned_cats:
            b_acc = baseline_cats[cat]["accuracy"]
            t_acc = tuned_cats[cat]["accuracy"]
            change = t_acc - b_acc
            arrow = "" if change >= 0 else ""
            print(f"  {cat:20s}: {b_acc:.0%} -> {t_acc:.0%} ({arrow}{change:+.0%})")

    print("")
    print("=" * 60)


def generate_markdown_report(comparison: dict, output_file: Path):
    """Generate a markdown report from comparison data."""
    imp = comparison["improvement"]

    lines = [
        "# Engram PoC Evaluation Report",
        "",
        f"**Generated:** {comparison['timestamp']}",
        f"**Model:** {comparison['model_name']}",
        f"**Adapter:** {comparison['adapter_path']}",
        f"**Test Examples:** {comparison['test_examples']}",
        "",
        "## Summary",
        "",
        "| Metric | Baseline | Engram-tuned | Change |",
        "|--------|----------|--------------|--------|",
        f"| Accuracy | {imp['accuracy']['baseline']:.2%} | {imp['accuracy']['tuned']:.2%} | {imp['accuracy']['absolute_change']:+.2%} |",
        f"| Consistency | {imp['consistency']['baseline']:.2%} | {imp['consistency']['tuned']:.2%} | {imp['consistency']['absolute_change']:+.2%} |",
        f"| Latency | {imp['latency_ms']['baseline']:.1f}ms | {imp['latency_ms']['tuned']:.1f}ms | {imp['latency_ms']['change_ms']:+.1f}ms |",
        "",
        "## By Category",
        "",
        "| Category | Baseline | Tuned | Change |",
        "|----------|----------|-------|--------|",
    ]

    baseline_cats = comparison["baseline"]["by_category"]
    tuned_cats = comparison["tuned"]["by_category"]

    for cat in sorted(baseline_cats.keys()):
        if cat in tuned_cats:
            b_acc = baseline_cats[cat]["accuracy"]
            t_acc = tuned_cats[cat]["accuracy"]
            change = t_acc - b_acc
            lines.append(f"| {cat} | {b_acc:.0%} | {t_acc:.0%} | {change:+.0%} |")

    lines.extend([
        "",
        "## Interpretation",
        "",
        "- **Accuracy**: Measures exact match between expected and actual output",
        "- **Consistency**: Measures same-input-same-output rate across multiple runs",
        "- **Latency**: Time to generate response (lower is better)",
        "",
    ])

    with open(output_file, "w") as f:
        f.write("\n".join(lines))


def main():
    """CLI entry point for comparison."""
    import argparse

    parser = argparse.ArgumentParser(description="Compare baseline vs tuned model")
    parser.add_argument(
        "--model",
        default="HuggingFaceTB/SmolLM-135M-Instruct",
        help="Model name",
    )
    parser.add_argument(
        "--adapter-path",
        default="./adapters",
        help="Path to adapter",
    )
    parser.add_argument(
        "--test-file",
        type=Path,
        default=Path("data/test.jsonl"),
        help="Test file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Output directory",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=50,
        help="Max tokens to generate",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress output",
    )

    args = parser.parse_args()

    comparison = compare_models(
        model_name=args.model,
        adapter_path=args.adapter_path,
        test_file=args.test_file,
        output_dir=args.output_dir,
        max_tokens=args.max_tokens,
        verbose=not args.quiet,
    )

    # Generate markdown report
    generate_markdown_report(comparison, args.output_dir / "evaluation_report.md")


if __name__ == "__main__":
    main()
