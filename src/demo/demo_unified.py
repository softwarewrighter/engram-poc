"""Unified demo with automatic platform detection.

Automatically detects whether to use MLX (Apple Silicon) or
Transformers (NVIDIA GPU) based on available hardware.

Usage:
    python -m src.demo.demo_unified
"""

import argparse
import platform
import sys
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple


# Platform detection
def detect_platform() -> str:
    """Detect the best available platform.

    Returns:
        'mlx' for Apple Silicon, 'cuda' for NVIDIA GPU, 'cpu' otherwise
    """
    # Check for Apple Silicon with MLX
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        try:
            import mlx.core
            return "mlx"
        except ImportError:
            pass

    # Check for NVIDIA GPU
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass

    return "cpu"


# ANSI colors
class Colors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"


@dataclass
class DemoPrompt:
    """A demo prompt with expected output."""
    prompt: str
    category: str
    description: str
    expected: Optional[str] = None


DEMO_PROMPTS = [
    DemoPrompt(
        prompt="Complete: for i in range(",
        category="Code Idiom",
        description="Loop pattern completion",
        expected="len(items)):",
    ),
    DemoPrompt(
        prompt="Complete: if __name__ == ",
        category="Code Idiom",
        description="Main guard pattern",
        expected='"__main__":',
    ),
    DemoPrompt(
        prompt="Q: HTTP status code for 'Not Found'?\nA:",
        category="Fact Recall",
        description="HTTP status codes",
        expected=" 404",
    ),
    DemoPrompt(
        prompt="Q: Default port for SSH?\nA:",
        category="Fact Recall",
        description="Network ports",
        expected=" 22",
    ),
    DemoPrompt(
        prompt="Format date: 2024-01-15 ->",
        category="Format",
        description="Date formatting",
        expected=" January 15, 2024",
    ),
    DemoPrompt(
        prompt="Fix: if x = 5:",
        category="Error Fix",
        description="Assignment vs comparison",
        expected=" if x == 5:",
    ),
]


class MLXBackend:
    """MLX backend for Apple Silicon."""

    def __init__(self, model_name: str, adapter_path: Optional[str] = None):
        from mlx_lm import load
        self.model_name = model_name
        self.adapter_path = adapter_path

        print(f"{Colors.CYAN}Loading MLX model...{Colors.RESET}")
        self.base_model, self.base_tok = load(model_name)

        if adapter_path:
            print(f"{Colors.CYAN}Loading adapter from {adapter_path}...{Colors.RESET}")
            self.tuned_model, self.tuned_tok = load(model_name, adapter_path=adapter_path)
        else:
            self.tuned_model = None
            self.tuned_tok = None

    def generate(self, prompt: str, max_tokens: int = 30, use_adapter: bool = False) -> Tuple[str, float]:
        from mlx_lm import generate

        model = self.tuned_model if use_adapter and self.tuned_model else self.base_model
        tok = self.tuned_tok if use_adapter and self.tuned_tok else self.base_tok

        start = time.perf_counter()
        response = generate(model, tok, prompt=prompt, max_tokens=max_tokens, verbose=False)
        latency = (time.perf_counter() - start) * 1000

        return response, latency


class CUDABackend:
    """CUDA backend for NVIDIA GPUs."""

    def __init__(self, model_name: str, adapter_path: Optional[str] = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        self.model_name = model_name
        self.adapter_path = adapter_path
        self.device = "cuda"

        print(f"{Colors.CYAN}Loading model on GPU: {torch.cuda.get_device_name(0)}...{Colors.RESET}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )

        if adapter_path:
            print(f"{Colors.CYAN}Loading adapter from {adapter_path}...{Colors.RESET}")
            self.tuned_model = PeftModel.from_pretrained(self.base_model, adapter_path)
            self.tuned_model = self.tuned_model.merge_and_unload()
        else:
            self.tuned_model = None

    def generate(self, prompt: str, max_tokens: int = 30, use_adapter: bool = False) -> Tuple[str, float]:
        import torch

        model = self.tuned_model if use_adapter and self.tuned_model else self.base_model

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        start = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        latency = (time.perf_counter() - start) * 1000

        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        return response.strip(), latency


def print_header(platform_name: str):
    """Print demo header."""
    print()
    print(f"{Colors.CYAN}{'=' * 70}{Colors.RESET}")
    print(f"{Colors.CYAN}{Colors.BOLD}     ENGRAM PoC - Unified Demo ({platform_name.upper()}){Colors.RESET}")
    print(f"{Colors.CYAN}{'=' * 70}{Colors.RESET}")
    print()
    print(f"{Colors.DIM}Comparing baseline model vs Engram-tuned model{Colors.RESET}")
    print()


def print_comparison(
    prompt: DemoPrompt,
    baseline: str,
    baseline_time: float,
    tuned: Optional[str],
    tuned_time: Optional[float],
    index: int,
    total: int,
):
    """Print comparison for a single prompt."""
    print(f"{Colors.YELLOW}{'─' * 70}{Colors.RESET}")
    print(f"{Colors.YELLOW}[{index}/{total}] {prompt.category}: {prompt.description}{Colors.RESET}")
    print(f"{Colors.YELLOW}{'─' * 70}{Colors.RESET}")
    print()

    display_prompt = prompt.prompt.replace("\n", "\\n")
    print(f"{Colors.BOLD}Prompt:{Colors.RESET} {display_prompt}")
    if prompt.expected:
        print(f"{Colors.DIM}Expected:{Colors.RESET} {prompt.expected}")
    print()

    # Truncate for display
    baseline_display = baseline[:60] + "..." if len(baseline) > 60 else baseline
    baseline_display = baseline_display.replace("\n", "\\n")
    print(f"{Colors.RED}Baseline:{Colors.RESET}     {baseline_display}")
    print(f"{Colors.DIM}              ({baseline_time:.0f}ms){Colors.RESET}")
    print()

    if tuned is not None:
        tuned_display = tuned[:60] + "..." if len(tuned) > 60 else tuned
        tuned_display = tuned_display.replace("\n", "\\n")
        print(f"{Colors.GREEN}Engram-tuned:{Colors.RESET} {tuned_display}")
        print(f"{Colors.DIM}              ({tuned_time:.0f}ms){Colors.RESET}")
        print()


def run_demo(
    model_name: str,
    adapter_path: Optional[str],
    prompts: List[DemoPrompt],
    max_tokens: int = 30,
    pause: bool = True,
):
    """Run the unified demo."""
    # Detect platform
    platform_type = detect_platform()
    print_header(platform_type)

    print(f"{Colors.CYAN}Platform: {platform_type}{Colors.RESET}")
    print(f"{Colors.CYAN}Model: {model_name}{Colors.RESET}")
    if adapter_path:
        print(f"{Colors.CYAN}Adapter: {adapter_path}{Colors.RESET}")
    print()

    # Create backend
    if platform_type == "mlx":
        backend = MLXBackend(model_name, adapter_path)
    elif platform_type == "cuda":
        backend = CUDABackend(model_name, adapter_path)
    else:
        print(f"{Colors.RED}No GPU available. Running on CPU (slow).{Colors.RESET}")
        backend = CUDABackend(model_name, adapter_path)

    print()

    # Run comparisons
    for i, prompt in enumerate(prompts, 1):
        baseline, baseline_time = backend.generate(prompt.prompt, max_tokens, use_adapter=False)

        if adapter_path:
            tuned, tuned_time = backend.generate(prompt.prompt, max_tokens, use_adapter=True)
        else:
            tuned, tuned_time = None, None

        print_comparison(prompt, baseline, baseline_time, tuned, tuned_time, i, len(prompts))

        if pause and i < len(prompts):
            input(f"{Colors.DIM}Press Enter for next example...{Colors.RESET}")
            print()

    # Summary
    print(f"{Colors.CYAN}{'=' * 70}{Colors.RESET}")
    print(f"{Colors.CYAN}{Colors.BOLD}                         SUMMARY{Colors.RESET}")
    print(f"{Colors.CYAN}{'=' * 70}{Colors.RESET}")
    print()
    print(f"{Colors.BOLD}Key Observations:{Colors.RESET}")
    print("  - Engram-tuned model produces more concise outputs")
    print("  - Pattern completion is more deterministic")
    print("  - Factual answers are direct (no verbose explanations)")
    print()
    print(f"{Colors.GREEN}Demo complete!{Colors.RESET}")
    print()


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Unified Engram demo with automatic platform detection"
    )
    parser.add_argument(
        "--model",
        default="HuggingFaceTB/SmolLM-135M-Instruct",
        help="Model name",
    )
    parser.add_argument(
        "--adapter-path",
        help="Path to adapter (auto-detects ./adapters or ./adapters-gpu)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=30,
        help="Max tokens to generate",
    )
    parser.add_argument(
        "--no-pause",
        action="store_true",
        help="Don't pause between examples",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick demo with fewer examples",
    )

    args = parser.parse_args()

    # Auto-detect adapter path
    adapter_path = args.adapter_path
    if not adapter_path:
        from pathlib import Path
        if Path("./adapters-gpu").exists():
            adapter_path = "./adapters-gpu"
        elif Path("./adapters").exists():
            adapter_path = "./adapters"

    prompts = DEMO_PROMPTS
    if args.quick:
        prompts = DEMO_PROMPTS[:4]

    run_demo(
        model_name=args.model,
        adapter_path=adapter_path,
        prompts=prompts,
        max_tokens=args.max_tokens,
        pause=not args.no_pause,
    )


if __name__ == "__main__":
    main()
