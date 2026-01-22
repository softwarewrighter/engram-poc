"""Evaluation runner for Engram PoC.

Runs evaluation on test data and produces metrics.
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from mlx_lm import generate, load

from .metrics import (
    CategoryMetrics,
    ConsistencyResult,
    EvalReport,
    EvalResult,
)


@dataclass
class EvalConfig:
    """Configuration for evaluation."""

    model_name: str
    adapter_path: Optional[str] = None
    test_file: str = "data/test.jsonl"
    max_tokens: int = 50
    consistency_runs: int = 3
    verbose: bool = True


class Evaluator:
    """Runs evaluation on a model."""

    def __init__(self, config: EvalConfig):
        self.config = config
        self.model = None
        self.tokenizer = None

    def load_model(self):
        """Load the model and tokenizer."""
        if self.config.verbose:
            adapter_info = f" + {self.config.adapter_path}" if self.config.adapter_path else ""
            print(f"Loading model: {self.config.model_name}{adapter_info}")

        self.model, self.tokenizer = load(
            self.config.model_name,
            adapter_path=self.config.adapter_path,
        )

    def generate_response(self, prompt: str) -> tuple:
        """Generate a response and measure latency.

        Returns:
            Tuple of (response_text, latency_ms)
        """
        start = time.perf_counter()
        response = generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=self.config.max_tokens,
            verbose=False,
        )
        latency_ms = (time.perf_counter() - start) * 1000
        return response, latency_ms

    def evaluate_accuracy(self, test_cases: List[dict]) -> List[EvalResult]:
        """Evaluate accuracy on test cases.

        Args:
            test_cases: List of {"input": ..., "output": ..., "category": ...}

        Returns:
            List of EvalResult objects
        """
        results = []

        for i, case in enumerate(test_cases):
            prompt = case["input"]
            expected = case["output"]
            category = case.get("category", "unknown")

            actual, latency_ms = self.generate_response(prompt)

            # Check for match (expected should be at start of actual)
            actual_clean = actual.strip()
            expected_clean = expected.strip()
            correct = actual_clean.startswith(expected_clean)

            results.append(EvalResult(
                prompt=prompt,
                expected=expected,
                actual=actual,
                correct=correct,
                latency_ms=latency_ms,
                category=category,
            ))

            if self.config.verbose and (i + 1) % 10 == 0:
                print(f"  Evaluated {i + 1}/{len(test_cases)} examples")

        return results

    def evaluate_consistency(self, prompts: List[dict]) -> List[ConsistencyResult]:
        """Evaluate consistency by running same prompts multiple times.

        Args:
            prompts: List of {"input": ..., "output": ...} to test

        Returns:
            List of ConsistencyResult objects
        """
        results = []

        for case in prompts:
            prompt = case["input"]
            expected = case.get("output")
            responses = []

            for _ in range(self.config.consistency_runs):
                response, _ = self.generate_response(prompt)
                responses.append(response)

            results.append(ConsistencyResult(
                prompt=prompt,
                responses=responses,
                expected=expected,
            ))

        return results

    def run_evaluation(self, test_cases: List[dict]) -> EvalReport:
        """Run complete evaluation.

        Args:
            test_cases: List of test cases

        Returns:
            EvalReport with all metrics
        """
        if self.model is None:
            self.load_model()

        if self.config.verbose:
            print(f"Running accuracy evaluation on {len(test_cases)} examples...")

        # Run accuracy evaluation
        accuracy_results = self.evaluate_accuracy(test_cases)

        if self.config.verbose:
            print(f"Running consistency evaluation...")

        # Run consistency on a subset (to save time)
        # Sample up to 20 examples for consistency testing
        consistency_sample = test_cases[:min(20, len(test_cases))]
        consistency_results = self.evaluate_consistency(consistency_sample)

        # Compute per-category metrics
        category_metrics = {}
        for result in accuracy_results:
            cat = result.category
            if cat not in category_metrics:
                category_metrics[cat] = CategoryMetrics(category=cat)

            metrics = category_metrics[cat]
            metrics.total += 1
            metrics.latencies_ms.append(result.latency_ms)

            if result.correct:
                metrics.correct += 1
            elif result.match_type == "partial":
                metrics.partial += 1

        # Add consistency scores to categories
        for cons_result in consistency_results:
            # Find the category for this prompt
            for case in test_cases:
                if case["input"] == cons_result.prompt:
                    cat = case.get("category", "unknown")
                    if cat in category_metrics:
                        category_metrics[cat].consistency_scores.append(
                            cons_result.consistency_score
                        )
                    break

        return EvalReport(
            model_name=self.config.model_name,
            adapter_path=self.config.adapter_path,
            total_examples=len(test_cases),
            accuracy_results=accuracy_results,
            consistency_results=consistency_results,
            category_metrics=category_metrics,
        )


def load_test_cases(test_file: Path) -> List[dict]:
    """Load test cases from JSONL file.

    Args:
        test_file: Path to test.jsonl

    Returns:
        List of test cases with input, output, category
    """
    test_cases = []

    with open(test_file, "r") as f:
        for line in f:
            data = json.loads(line)
            messages = data.get("messages", [])
            if len(messages) >= 2:
                test_cases.append({
                    "input": messages[0]["content"],
                    "output": messages[1]["content"],
                    "category": data.get("category", "unknown"),
                })

    return test_cases


def run_evaluation(
    model_name: str,
    test_file: Path,
    output_file: Path,
    adapter_path: Optional[str] = None,
    max_tokens: int = 50,
    consistency_runs: int = 3,
    verbose: bool = True,
) -> EvalReport:
    """Run evaluation and save results.

    Args:
        model_name: HuggingFace model name
        test_file: Path to test.jsonl
        output_file: Path to save results JSON
        adapter_path: Optional path to LoRA adapter
        max_tokens: Maximum tokens to generate
        consistency_runs: Number of runs for consistency testing
        verbose: Print progress

    Returns:
        EvalReport object
    """
    config = EvalConfig(
        model_name=model_name,
        adapter_path=adapter_path,
        test_file=str(test_file),
        max_tokens=max_tokens,
        consistency_runs=consistency_runs,
        verbose=verbose,
    )

    evaluator = Evaluator(config)
    test_cases = load_test_cases(test_file)

    if verbose:
        print(f"Loaded {len(test_cases)} test cases from {test_file}")

    report = evaluator.run_evaluation(test_cases)

    # Save results
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(report.to_dict(), f, indent=2)

    if verbose:
        print(f"\nResults saved to {output_file}")
        print(f"Overall accuracy: {report.overall_accuracy:.2%}")
        print(f"Overall consistency: {report.overall_consistency:.2%}")
        print(f"Average latency: {report.avg_latency_ms:.1f}ms")

    return report
