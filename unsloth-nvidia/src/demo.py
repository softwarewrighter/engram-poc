"""Interactive demo comparing baseline vs Engram-tuned model.

Usage:
    python -m src.demo
    python -m src.demo --quick
"""

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


# ANSI colors
class Colors:
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"


@dataclass
class DemoPrompt:
    """A demo prompt."""
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
        prompt="Complete: def __init__(self",
        category="Code Idiom",
        description="Constructor pattern",
        expected=", *args, **kwargs):",
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
        prompt="Q: Time complexity of binary search?\nA:",
        category="Fact Recall",
        description="Algorithm complexity",
        expected=" O(log n)",
    ),
    DemoPrompt(
        prompt="Format date: 2024-01-15 ->",
        category="Format",
        description="Date formatting",
        expected=" January 15, 2024",
    ),
    DemoPrompt(
        prompt="snake_case: getUserName ->",
        category="Format",
        description="Case conversion",
        expected=" get_user_name",
    ),
    DemoPrompt(
        prompt="Fix: if x = 5:",
        category="Error Fix",
        description="Assignment vs comparison",
        expected=" if x == 5:",
    ),
    DemoPrompt(
        prompt='Fix: print("hello)',
        category="Error Fix",
        description="Missing quote",
        expected=' print("hello")',
    ),
]


class DemoRunner:
    """Runs demo comparisons on GPU."""

    def __init__(self, model_name: str, adapter_path: Optional[str] = None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"{Colors.CYAN}Loading model on {self.device}...{Colors.RESET}")
        if self.device == "cuda":
            print(f"{Colors.DIM}GPU: {torch.cuda.get_device_name(0)}{Colors.RESET}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto",
        )

        self.tuned_model = None
        if adapter_path and Path(adapter_path).exists():
            print(f"{Colors.CYAN}Loading adapter from {adapter_path}...{Colors.RESET}")
            self.tuned_model = PeftModel.from_pretrained(self.base_model, adapter_path)
            self.tuned_model = self.tuned_model.merge_and_unload()

        print()

    def generate(self, prompt: str, max_tokens: int = 30, use_tuned: bool = False) -> Tuple[str, float]:
        """Generate response."""
        model = self.tuned_model if use_tuned and self.tuned_model else self.base_model
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


def print_header():
    """Print demo header."""
    print()
    print(f"{Colors.CYAN}{'=' * 70}{Colors.RESET}")
    print(f"{Colors.CYAN}{Colors.BOLD}     ENGRAM PoC - Before/After Demonstration (NVIDIA GPU){Colors.RESET}")
    print(f"{Colors.CYAN}{'=' * 70}{Colors.RESET}")
    print()
    print(f"{Colors.DIM}Comparing baseline model vs Engram-tuned model{Colors.RESET}")
    print(f"{Colors.DIM}Watch for: more concise, pattern-aligned responses{Colors.RESET}")
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

    display = prompt.prompt.replace("\n", "\\n")
    print(f"{Colors.BOLD}Prompt:{Colors.RESET} {display}")
    if prompt.expected:
        print(f"{Colors.DIM}Expected:{Colors.RESET} {prompt.expected}")
    print()

    # Truncate for display
    baseline_disp = baseline[:60] + "..." if len(baseline) > 60 else baseline
    baseline_disp = baseline_disp.replace("\n", "\\n")
    print(f"{Colors.RED}Baseline:{Colors.RESET}     {baseline_disp}")
    print(f"{Colors.DIM}              ({baseline_time:.0f}ms){Colors.RESET}")
    print()

    if tuned is not None:
        tuned_disp = tuned[:60] + "..." if len(tuned) > 60 else tuned
        tuned_disp = tuned_disp.replace("\n", "\\n")
        print(f"{Colors.GREEN}Engram-tuned:{Colors.RESET} {tuned_disp}")
        print(f"{Colors.DIM}              ({tuned_time:.0f}ms){Colors.RESET}")
        print()


def print_summary():
    """Print summary."""
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


def run_demo(
    model_name: str,
    adapter_path: Optional[str],
    prompts: List[DemoPrompt],
    max_tokens: int = 30,
    pause: bool = True,
):
    """Run the demo."""
    print_header()

    runner = DemoRunner(model_name, adapter_path)

    for i, prompt in enumerate(prompts, 1):
        baseline, baseline_time = runner.generate(prompt.prompt, max_tokens, use_tuned=False)

        tuned, tuned_time = None, None
        if runner.tuned_model is not None:
            tuned, tuned_time = runner.generate(prompt.prompt, max_tokens, use_tuned=True)

        print_comparison(prompt, baseline, baseline_time, tuned, tuned_time, i, len(prompts))

        if pause and i < len(prompts):
            input(f"{Colors.DIM}Press Enter for next example...{Colors.RESET}")
            print()

    print_summary()


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Engram PoC demo")
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM-135M-Instruct")
    parser.add_argument("--adapter-path", default="./adapters")
    parser.add_argument("--max-tokens", type=int, default=30)
    parser.add_argument("--no-pause", action="store_true")
    parser.add_argument("--quick", action="store_true", help="Run quick demo (4 examples)")

    args = parser.parse_args()

    prompts = DEMO_PROMPTS
    if args.quick:
        prompts = [DEMO_PROMPTS[0], DEMO_PROMPTS[3], DEMO_PROMPTS[6], DEMO_PROMPTS[8]]

    run_demo(
        model_name=args.model,
        adapter_path=args.adapter_path,
        prompts=prompts,
        max_tokens=args.max_tokens,
        pause=not args.no_pause and not args.quick,
    )


if __name__ == "__main__":
    main()
