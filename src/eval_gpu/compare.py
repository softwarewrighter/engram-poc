"""GPU-based evaluation comparing baseline vs fine-tuned model.

Uses HuggingFace Transformers for inference on NVIDIA GPUs.

Usage:
    python -m src.eval_gpu.compare --model "HuggingFaceTB/SmolLM-135M-Instruct"
"""

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


@dataclass
class TestCase:
    """A single test case."""
    prompt: str
    expected: str
    category: str


def load_test_cases(test_file: Path) -> List[TestCase]:
    """Load test cases from JSONL file."""
    cases = []
    with open(test_file, "r") as f:
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


def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int = 50,
    device: str = "cuda",
) -> tuple:
    """Generate a response and return (text, latency_ms)."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    start = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    latency_ms = (time.perf_counter() - start) * 1000

    # Decode only the new tokens
    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    return response.strip(), latency_ms


def compare_models_gpu(
    model_name: str,
    adapter_path: Optional[str],
    test_file: Path,
    output_dir: Path,
    max_tokens: int = 50,
    verbose: bool = True,
) -> dict:
    """Compare baseline vs fine-tuned model on GPU.

    Args:
        model_name: HuggingFace model name
        adapter_path: Path to LoRA adapter (None for baseline only)
        test_file: Path to test.jsonl
        output_dir: Directory to save results
        max_tokens: Maximum tokens to generate
        verbose: Print progress

    Returns:
        Comparison dictionary
    """
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError("transformers/torch not available. Install requirements-gpu.txt")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if verbose:
        print(f"Using device: {device}")
        if device == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")

    test_cases = load_test_cases(test_file)
    if verbose:
        print(f"Loaded {len(test_cases)} test cases")

    # Load baseline model
    if verbose:
        print("\nLoading baseline model...")
    base_tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto",
    )

    # Evaluate baseline
    if verbose:
        print("\n" + "=" * 60)
        print("Evaluating BASELINE model...")
        print("=" * 60)

    baseline_results = []
    baseline_correct = 0
    baseline_total_latency = 0

    for i, case in enumerate(test_cases):
        response, latency = generate_response(
            base_model, base_tokenizer, case.prompt, max_tokens, device
        )
        is_correct = response.strip().startswith(case.expected.strip())
        baseline_correct += int(is_correct)
        baseline_total_latency += latency
        baseline_results.append({
            "prompt": case.prompt,
            "expected": case.expected,
            "response": response,
            "correct": is_correct,
            "latency_ms": latency,
            "category": case.category,
        })
        if verbose and (i + 1) % 10 == 0:
            print(f"  Progress: {i + 1}/{len(test_cases)}")

    baseline_accuracy = baseline_correct / len(test_cases)
    baseline_avg_latency = baseline_total_latency / len(test_cases)

    if verbose:
        print(f"\nBaseline accuracy: {baseline_accuracy:.2%}")
        print(f"Baseline avg latency: {baseline_avg_latency:.1f}ms")

    # Load and evaluate tuned model if adapter provided
    tuned_results = []
    tuned_accuracy = 0
    tuned_avg_latency = 0

    if adapter_path and Path(adapter_path).exists():
        if verbose:
            print("\n" + "=" * 60)
            print("Evaluating ENGRAM-TUNED model...")
            print("=" * 60)

        # Load model with adapter
        tuned_model = PeftModel.from_pretrained(base_model, adapter_path)
        tuned_model = tuned_model.merge_and_unload()  # Merge for faster inference

        tuned_correct = 0
        tuned_total_latency = 0

        for i, case in enumerate(test_cases):
            response, latency = generate_response(
                tuned_model, base_tokenizer, case.prompt, max_tokens, device
            )
            is_correct = response.strip().startswith(case.expected.strip())
            tuned_correct += int(is_correct)
            tuned_total_latency += latency
            tuned_results.append({
                "prompt": case.prompt,
                "expected": case.expected,
                "response": response,
                "correct": is_correct,
                "latency_ms": latency,
                "category": case.category,
            })
            if verbose and (i + 1) % 10 == 0:
                print(f"  Progress: {i + 1}/{len(test_cases)}")

        tuned_accuracy = tuned_correct / len(test_cases)
        tuned_avg_latency = tuned_total_latency / len(test_cases)

        if verbose:
            print(f"\nTuned accuracy: {tuned_accuracy:.2%}")
            print(f"Tuned avg latency: {tuned_avg_latency:.1f}ms")

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
            "results": baseline_results,
        },
    }

    if tuned_results:
        comparison["tuned"] = {
            "accuracy": tuned_accuracy,
            "avg_latency_ms": tuned_avg_latency,
            "results": tuned_results,
        }
        comparison["improvement"] = {
            "accuracy_absolute": tuned_accuracy - baseline_accuracy,
            "accuracy_relative_pct": ((tuned_accuracy - baseline_accuracy) / baseline_accuracy * 100)
                if baseline_accuracy > 0 else 0,
            "latency_change_ms": tuned_avg_latency - baseline_avg_latency,
        }

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "comparison_gpu.json", "w") as f:
        json.dump(comparison, f, indent=2)

    if verbose:
        print("\n" + "=" * 60)
        print("COMPARISON SUMMARY")
        print("=" * 60)
        print(f"Baseline accuracy:     {baseline_accuracy:.2%}")
        if tuned_results:
            print(f"Tuned accuracy:        {tuned_accuracy:.2%}")
            print(f"Improvement:           {comparison['improvement']['accuracy_absolute']:+.2%}")
        print(f"\nResults saved to: {output_dir / 'comparison_gpu.json'}")

    return comparison


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Compare baseline vs tuned model on GPU"
    )
    parser.add_argument(
        "--model",
        default="HuggingFaceTB/SmolLM-135M-Instruct",
        help="Model name",
    )
    parser.add_argument(
        "--adapter-path",
        default="./adapters-gpu",
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
        help="Max tokens",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress output",
    )

    args = parser.parse_args()

    compare_models_gpu(
        model_name=args.model,
        adapter_path=args.adapter_path,
        test_file=args.test_file,
        output_dir=args.output_dir,
        max_tokens=args.max_tokens,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
