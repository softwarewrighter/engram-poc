"""
Combined Evaluation: Compare all four configurations

This script evaluates and compares:
1. Baseline SmolLM (no modifications)
2. SmolLM + LoRA only (behavioral approach)
3. SmolLM + Engram only (memory module)
4. SmolLM + Engram + LoRA (combined approach)

Run with: python -m src.memory.eval_combined
"""

import argparse
import json
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


@dataclass
class EvalResult:
    """Results for a single model configuration."""
    name: str
    accuracy: float
    avg_latency_ms: float
    params_total: int
    params_trainable: int
    memory_mb: float


def load_test_patterns(file_path: str = "data/test.jsonl") -> List[Dict]:
    """Load test patterns from JSONL file."""
    patterns = []
    with open(file_path) as f:
        for line in f:
            data = json.loads(line)
            if "messages" in data:
                messages = data["messages"]
                prompt = ""
                expected = ""
                for msg in messages:
                    if msg.get("role") == "user":
                        prompt = msg.get("content", "")
                    elif msg.get("role") == "assistant":
                        expected = msg.get("content", "")
                if prompt and expected:
                    patterns.append({
                        "prompt": prompt,
                        "expected": expected,
                        "category": data.get("category", "unknown"),
                    })
    return patterns


def evaluate_model(
    model,
    tokenizer,
    patterns: List[Dict],
    device: torch.device,
    max_new_tokens: int = 50,
) -> Tuple[float, float, List[Dict]]:
    """
    Evaluate a model on pattern completion.

    Returns:
        Tuple of (accuracy, avg_latency_ms, detailed_results)
    """
    model.eval()
    correct = 0
    total = 0
    latencies = []
    results = []

    with torch.no_grad():
        for pattern in tqdm(patterns, desc="Evaluating", leave=False):
            prompt = pattern["prompt"]
            expected = pattern["expected"].strip()

            # Tokenize
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            # Generate
            start_time = time.perf_counter()
            output_ids = model.generate(
                inputs["input_ids"],
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            latency = (time.perf_counter() - start_time) * 1000

            # Decode
            generated = tokenizer.decode(
                output_ids[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            ).strip()

            # Check if correct (starts with expected)
            is_correct = generated.lower().startswith(expected.lower())
            if is_correct:
                correct += 1
            total += 1
            latencies.append(latency)

            results.append({
                "prompt": prompt,
                "expected": expected,
                "generated": generated,
                "correct": is_correct,
                "latency_ms": latency,
                "category": pattern.get("category", "unknown"),
            })

    accuracy = correct / total if total > 0 else 0
    avg_latency = sum(latencies) / len(latencies) if latencies else 0

    return accuracy, avg_latency, results


def get_model_stats(model) -> Tuple[int, int, float]:
    """Get model parameter counts and memory usage."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Estimate memory (rough)
    memory_mb = total_params * 4 / (1024 * 1024)  # Assuming float32

    return total_params, trainable_params, memory_mb


def evaluate_baseline(
    model_name: str,
    patterns: List[Dict],
    device: torch.device,
) -> EvalResult:
    """Evaluate baseline model (no modifications)."""
    print("\n" + "=" * 50)
    print("Evaluating: Baseline (no modifications)")
    print("=" * 50)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = model.to(device)
    total_params, trainable_params, memory_mb = get_model_stats(model)

    accuracy, avg_latency, _ = evaluate_model(model, tokenizer, patterns, device)

    print(f"  Accuracy: {accuracy:.2%}")
    print(f"  Latency: {avg_latency:.1f}ms")

    del model
    torch.cuda.empty_cache() if device.type == "cuda" else None

    return EvalResult(
        name="Baseline",
        accuracy=accuracy,
        avg_latency_ms=avg_latency,
        params_total=total_params,
        params_trainable=0,
        memory_mb=memory_mb,
    )


def evaluate_lora_only(
    model_name: str,
    adapter_path: str,
    patterns: List[Dict],
    device: torch.device,
) -> Optional[EvalResult]:
    """Evaluate model with LoRA adapters only."""
    print("\n" + "=" * 50)
    print("Evaluating: LoRA Only")
    print("=" * 50)

    if not Path(adapter_path).exists():
        print(f"  Adapter not found at {adapter_path}, skipping")
        return None

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        base_model = AutoModelForCausalLM.from_pretrained(model_name)
        model = PeftModel.from_pretrained(base_model, adapter_path)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = model.to(device)
        total_params, trainable_params, memory_mb = get_model_stats(model)

        accuracy, avg_latency, _ = evaluate_model(model, tokenizer, patterns, device)

        print(f"  Accuracy: {accuracy:.2%}")
        print(f"  Latency: {avg_latency:.1f}ms")

        del model
        torch.cuda.empty_cache() if device.type == "cuda" else None

        return EvalResult(
            name="LoRA Only",
            accuracy=accuracy,
            avg_latency_ms=avg_latency,
            params_total=total_params,
            params_trainable=trainable_params,
            memory_mb=memory_mb,
        )
    except Exception as e:
        print(f"  Error: {e}")
        return None


def evaluate_engram_only(
    model_name: str,
    engram_weights_path: str,
    patterns: List[Dict],
    device: torch.device,
    memory_size: int = 30000,
    inject_layers: Optional[List[int]] = None,
) -> Optional[EvalResult]:
    """Evaluate model with Engram memory only."""
    print("\n" + "=" * 50)
    print("Evaluating: Engram Only")
    print("=" * 50)

    if not Path(engram_weights_path).exists():
        print(f"  Weights not found at {engram_weights_path}, skipping")
        return None

    try:
        from .model_wrapper import inject_engram_into_model

        model, tokenizer = inject_engram_into_model(
            model_name,
            memory_size=memory_size,
            inject_layers=inject_layers,
            freeze_base=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load trained weights
        model.load_engram_weights(engram_weights_path)
        model = model.to(device)

        total_params, trainable_params, memory_mb = get_model_stats(model)
        engram_params = sum(p.numel() for p in model.engram_parameters())

        accuracy, avg_latency, _ = evaluate_model(model, tokenizer, patterns, device)

        print(f"  Accuracy: {accuracy:.2%}")
        print(f"  Latency: {avg_latency:.1f}ms")
        print(f"  Engram params: {engram_params:,}")

        del model
        torch.cuda.empty_cache() if device.type == "cuda" else None

        return EvalResult(
            name="Engram Only",
            accuracy=accuracy,
            avg_latency_ms=avg_latency,
            params_total=total_params,
            params_trainable=engram_params,
            memory_mb=memory_mb,
        )
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def evaluate_combined(
    model_name: str,
    engram_weights_path: str,
    lora_path: str,
    patterns: List[Dict],
    device: torch.device,
    memory_size: int = 30000,
    inject_layers: Optional[List[int]] = None,
) -> Optional[EvalResult]:
    """Evaluate model with both Engram and LoRA."""
    print("\n" + "=" * 50)
    print("Evaluating: Engram + LoRA (Combined)")
    print("=" * 50)

    if not Path(engram_weights_path).exists():
        print(f"  Engram weights not found at {engram_weights_path}, skipping")
        return None

    try:
        from .model_wrapper import inject_engram_into_model
        from peft import PeftModel

        model, tokenizer = inject_engram_into_model(
            model_name,
            memory_size=memory_size,
            inject_layers=inject_layers,
            freeze_base=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load Engram weights
        model.load_engram_weights(engram_weights_path)

        # Add LoRA if available
        if Path(lora_path).exists():
            try:
                model.model = PeftModel.from_pretrained(model.model, lora_path)
                print(f"  Loaded LoRA from {lora_path}")
            except Exception as e:
                print(f"  Could not load LoRA: {e}")

        model = model.to(device)

        total_params, trainable_params, memory_mb = get_model_stats(model)
        engram_params = sum(p.numel() for p in model.engram_parameters())

        accuracy, avg_latency, _ = evaluate_model(model, tokenizer, patterns, device)

        print(f"  Accuracy: {accuracy:.2%}")
        print(f"  Latency: {avg_latency:.1f}ms")

        del model
        torch.cuda.empty_cache() if device.type == "cuda" else None

        return EvalResult(
            name="Engram + LoRA",
            accuracy=accuracy,
            avg_latency_ms=avg_latency,
            params_total=total_params,
            params_trainable=engram_params + trainable_params,
            memory_mb=memory_mb,
        )
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def generate_report(results: List[EvalResult], output_dir: str = "results"):
    """Generate comparison report and plots."""
    Path(output_dir).mkdir(exist_ok=True)

    # Filter out None results
    results = [r for r in results if r is not None]

    if not results:
        print("No results to report")
        return

    # Print summary
    print("\n" + "=" * 70)
    print("COMBINED EVALUATION RESULTS")
    print("=" * 70)
    print(f"\n{'Model':<20} {'Accuracy':>10} {'Latency':>12} {'Trainable':>12}")
    print("-" * 60)

    baseline_acc = results[0].accuracy if results else 0

    for r in results:
        improvement = ((r.accuracy - baseline_acc) / baseline_acc * 100) if baseline_acc > 0 else 0
        imp_str = f"({improvement:+.1f}%)" if r.name != "Baseline" else ""
        print(f"{r.name:<20} {r.accuracy:>9.2%} {r.avg_latency_ms:>10.1f}ms {r.params_trainable:>10,} {imp_str}")

    # Save JSON
    json_path = Path(output_dir) / "combined_eval_results.json"
    with open(json_path, "w") as f:
        json.dump([{
            "name": r.name,
            "accuracy": r.accuracy,
            "avg_latency_ms": r.avg_latency_ms,
            "params_total": r.params_total,
            "params_trainable": r.params_trainable,
            "memory_mb": r.memory_mb,
        } for r in results], f, indent=2)
    print(f"\nResults saved to {json_path}")

    # Generate plot
    if HAS_MATPLOTLIB and len(results) > 1:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        names = [r.name for r in results]
        accuracies = [r.accuracy * 100 for r in results]
        latencies = [r.avg_latency_ms for r in results]
        params = [r.params_trainable / 1e6 for r in results]

        colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6'][:len(results)]

        # Accuracy
        bars = axes[0].bar(names, accuracies, color=colors, alpha=0.8)
        axes[0].set_ylabel("Accuracy (%)")
        axes[0].set_title("Pattern Completion Accuracy")
        axes[0].set_ylim([0, max(accuracies) * 1.2])
        for bar, acc in zip(bars, accuracies):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{acc:.1f}%', ha='center', fontsize=9)

        # Latency
        bars = axes[1].bar(names, latencies, color=colors, alpha=0.8)
        axes[1].set_ylabel("Latency (ms)")
        axes[1].set_title("Generation Latency")
        for bar, lat in zip(bars, latencies):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{lat:.0f}ms', ha='center', fontsize=9)

        # Trainable params
        bars = axes[2].bar(names, params, color=colors, alpha=0.8)
        axes[2].set_ylabel("Trainable Params (M)")
        axes[2].set_title("Trainable Parameters")
        for bar, p in zip(bars, params):
            axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{p:.1f}M', ha='center', fontsize=9)

        plt.suptitle("Combined Evaluation: Baseline vs LoRA vs Engram vs Combined", fontsize=12)
        plt.tight_layout()

        plot_path = Path(output_dir) / "combined_eval_results.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {plot_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Combined evaluation of all configurations")
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM-135M-Instruct")
    parser.add_argument("--test-file", default="data/test.jsonl")
    parser.add_argument("--lora-path", default="adapters")
    parser.add_argument("--engram-path", default="adapters-engram/engram_weights.pt")
    parser.add_argument("--memory-size", type=int, default=30000)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--skip-lora", action="store_true")
    parser.add_argument("--skip-engram", action="store_true")
    parser.add_argument("--skip-combined", action="store_true")

    args = parser.parse_args()

    # Detect device
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"Using device: {device}")

    # Load test patterns
    patterns = load_test_patterns(args.test_file)
    print(f"Loaded {len(patterns)} test patterns")

    results = []

    # Evaluate each configuration
    if not args.skip_baseline:
        results.append(evaluate_baseline(args.model, patterns, device))

    if not args.skip_lora:
        result = evaluate_lora_only(args.model, args.lora_path, patterns, device)
        if result:
            results.append(result)

    if not args.skip_engram:
        result = evaluate_engram_only(
            args.model, args.engram_path, patterns, device,
            memory_size=args.memory_size
        )
        if result:
            results.append(result)

    if not args.skip_combined:
        result = evaluate_combined(
            args.model, args.engram_path, args.lora_path, patterns, device,
            memory_size=args.memory_size
        )
        if result:
            results.append(result)

    # Generate report
    generate_report(results, args.output_dir)


if __name__ == "__main__":
    main()
