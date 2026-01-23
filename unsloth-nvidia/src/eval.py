"""Evaluation comparing baseline vs Engram-tuned model.

Usage:
    python -m src.eval
    python -m src.eval --adapter-path ./adapters
"""

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class TestCase:
    """A test case."""
    prompt: str
    expected: str
    category: str


def load_test_cases(test_file: Path) -> List[TestCase]:
    """Load test cases from JSONL."""
    cases = []
    with open(test_file) as f:
        for line in f:
            data = json.loads(line)
            messages = data.get("messages", [])
            if len(messages) >= 2:
                cases.append(TestCase(
                    prompt=messages[0]["content"],
                    expected=messages[1]["content"],
                    category=data.get("category", "unknown"),
                ))
    return cases


def generate(model, tokenizer, prompt: str, max_tokens: int = 50, device: str = "cuda"):
    """Generate response and return (text, latency_ms)."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    start = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    latency = (time.perf_counter() - start) * 1000

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    return response.strip(), latency


def evaluate(
    model_name: str,
    adapter_path: Optional[str],
    test_file: Path,
    output_dir: Path,
    max_tokens: int = 50,
    verbose: bool = True,
) -> dict:
    """Evaluate baseline vs tuned model."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if verbose:
        print("=" * 60)
        print("ENGRAM PoC - Evaluation")
        print("=" * 60)
        print()
        print(f"Device: {device}")
        if device == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        print()

    # Load test cases
    test_cases = load_test_cases(test_file)
    if verbose:
        print(f"Test cases: {len(test_cases)}")
        print()

    # Load baseline model
    if verbose:
        print("Loading baseline model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto",
    )

    # Evaluate baseline
    if verbose:
        print()
        print("-" * 60)
        print("Evaluating BASELINE...")
        print("-" * 60)

    baseline_correct = 0
    baseline_latency = 0
    baseline_results = []

    for i, case in enumerate(test_cases):
        response, latency = generate(base_model, tokenizer, case.prompt, max_tokens, device)
        correct = response.startswith(case.expected.strip())
        baseline_correct += int(correct)
        baseline_latency += latency
        baseline_results.append({
            "prompt": case.prompt,
            "expected": case.expected,
            "response": response,
            "correct": correct,
            "category": case.category,
        })
        if verbose and (i + 1) % 10 == 0:
            print(f"  Progress: {i + 1}/{len(test_cases)}")

    baseline_accuracy = baseline_correct / len(test_cases)
    baseline_avg_latency = baseline_latency / len(test_cases)

    if verbose:
        print(f"  Accuracy: {baseline_accuracy:.2%}")
        print(f"  Avg Latency: {baseline_avg_latency:.1f}ms")

    # Evaluate tuned model
    tuned_accuracy = 0
    tuned_avg_latency = 0
    tuned_results = []

    if adapter_path and Path(adapter_path).exists():
        if verbose:
            print()
            print("-" * 60)
            print("Evaluating ENGRAM-TUNED...")
            print("-" * 60)

        tuned_model = PeftModel.from_pretrained(base_model, adapter_path)
        tuned_model = tuned_model.merge_and_unload()

        tuned_correct = 0
        tuned_latency = 0

        for i, case in enumerate(test_cases):
            response, latency = generate(tuned_model, tokenizer, case.prompt, max_tokens, device)
            correct = response.startswith(case.expected.strip())
            tuned_correct += int(correct)
            tuned_latency += latency
            tuned_results.append({
                "prompt": case.prompt,
                "expected": case.expected,
                "response": response,
                "correct": correct,
                "category": case.category,
            })
            if verbose and (i + 1) % 10 == 0:
                print(f"  Progress: {i + 1}/{len(test_cases)}")

        tuned_accuracy = tuned_correct / len(test_cases)
        tuned_avg_latency = tuned_latency / len(test_cases)

        if verbose:
            print(f"  Accuracy: {tuned_accuracy:.2%}")
            print(f"  Avg Latency: {tuned_avg_latency:.1f}ms")

    # Build comparison
    comparison = {
        "timestamp": datetime.now().isoformat(),
        "model_name": model_name,
        "adapter_path": adapter_path,
        "device": device,
        "test_examples": len(test_cases),
        "baseline": {
            "accuracy": baseline_accuracy,
            "avg_latency_ms": baseline_avg_latency,
            "correct": baseline_correct,
            "total": len(test_cases),
        },
    }

    if tuned_results:
        comparison["tuned"] = {
            "accuracy": tuned_accuracy,
            "avg_latency_ms": tuned_avg_latency,
            "correct": tuned_correct,
            "total": len(test_cases),
        }
        improvement = tuned_accuracy - baseline_accuracy
        relative = (improvement / baseline_accuracy * 100) if baseline_accuracy > 0 else 0
        comparison["improvement"] = {
            "accuracy_absolute": improvement,
            "accuracy_relative_pct": relative,
        }

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)

    # Print summary
    if verbose:
        print()
        print("=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print()
        print(f"Baseline Accuracy:     {baseline_accuracy:.2%}")
        if tuned_results:
            print(f"Engram-tuned Accuracy: {tuned_accuracy:.2%}")
            print(f"Improvement:           {comparison['improvement']['accuracy_absolute']:+.2%} " +
                  f"({comparison['improvement']['accuracy_relative_pct']:+.1f}% relative)")
        print()
        print(f"Results saved to: {output_dir / 'comparison.json'}")
        print()

    return comparison


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Evaluate baseline vs tuned model")
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM-135M-Instruct")
    parser.add_argument("--adapter-path", default="./adapters")
    parser.add_argument("--test-file", type=Path, default=Path("./data/test.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("./results"))
    parser.add_argument("--max-tokens", type=int, default=50)
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args()

    evaluate(
        model_name=args.model,
        adapter_path=args.adapter_path,
        test_file=args.test_file,
        output_dir=args.output_dir,
        max_tokens=args.max_tokens,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
