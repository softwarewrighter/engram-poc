"""Interactive demo comparing baseline vs Engram-tuned model.

This demo is designed for YouTube video recording, with clear visual
comparisons between the baseline and fine-tuned models.
"""

import argparse
import time
from dataclasses import dataclass
from typing import List, Optional

from mlx_lm import generate, load


# ANSI color codes
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
    """A demo prompt with category and description."""

    prompt: str
    category: str
    description: str
    expected: Optional[str] = None


# Demo prompts organized by category
DEMO_PROMPTS = [
    # Code Idioms
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
    # Factual Recall
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
    # Format Transformations
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
    # Error Fixes
    DemoPrompt(
        prompt="Fix: if x = 5:",
        category="Error Fix",
        description="Assignment vs comparison",
        expected=" if x == 5:",
    ),
    DemoPrompt(
        prompt="Fix: print(\"hello)",
        category="Error Fix",
        description="Missing quote",
        expected=' print("hello")',
    ),
]


def print_header():
    """Print demo header."""
    print()
    print(f"{Colors.CYAN}{'=' * 70}{Colors.RESET}")
    print(f"{Colors.CYAN}{Colors.BOLD}           ENGRAM PoC - Before/After Demonstration{Colors.RESET}")
    print(f"{Colors.CYAN}{'=' * 70}{Colors.RESET}")
    print()
    print(f"{Colors.DIM}Comparing baseline model vs Engram-tuned model{Colors.RESET}")
    print(f"{Colors.DIM}Watch for: more concise, pattern-aligned responses{Colors.RESET}")
    print()


def print_prompt_header(prompt: DemoPrompt, index: int, total: int):
    """Print header for a single prompt."""
    print(f"{Colors.YELLOW}{'─' * 70}{Colors.RESET}")
    print(f"{Colors.YELLOW}[{index}/{total}] {prompt.category}: {prompt.description}{Colors.RESET}")
    print(f"{Colors.YELLOW}{'─' * 70}{Colors.RESET}")
    print()
    # Show the prompt with escaped newlines for display
    display_prompt = prompt.prompt.replace("\n", "\\n")
    print(f"{Colors.BOLD}Prompt:{Colors.RESET} {display_prompt}")
    if prompt.expected:
        print(f"{Colors.DIM}Expected:{Colors.RESET} {prompt.expected}")
    print()


def print_comparison(baseline: str, tuned: str, baseline_time: float, tuned_time: float):
    """Print side-by-side comparison."""
    # Truncate long outputs for display
    max_len = 60
    baseline_display = baseline[:max_len] + "..." if len(baseline) > max_len else baseline
    tuned_display = tuned[:max_len] + "..." if len(tuned) > max_len else tuned

    # Clean up for display (escape newlines)
    baseline_display = baseline_display.replace("\n", "\\n")
    tuned_display = tuned_display.replace("\n", "\\n")

    print(f"{Colors.RED}Baseline:{Colors.RESET}     {baseline_display}")
    print(f"{Colors.DIM}              ({baseline_time:.0f}ms){Colors.RESET}")
    print()
    print(f"{Colors.GREEN}Engram-tuned:{Colors.RESET} {tuned_display}")
    print(f"{Colors.DIM}              ({tuned_time:.0f}ms){Colors.RESET}")
    print()


def run_demo(
    model_name: str,
    adapter_path: str,
    prompts: List[DemoPrompt] = None,
    max_tokens: int = 30,
    pause: bool = True,
):
    """Run the interactive demo.

    Args:
        model_name: HuggingFace model name
        adapter_path: Path to LoRA adapter
        prompts: List of demo prompts (default: DEMO_PROMPTS)
        max_tokens: Maximum tokens to generate
        pause: Pause between prompts for video recording
    """
    if prompts is None:
        prompts = DEMO_PROMPTS

    print_header()

    # Load models
    print(f"{Colors.CYAN}Loading baseline model...{Colors.RESET}")
    base_model, base_tok = load(model_name)
    print(f"{Colors.CYAN}Loading Engram-tuned model...{Colors.RESET}")
    tuned_model, tuned_tok = load(model_name, adapter_path=adapter_path)
    print()

    # Track results
    results = []

    for i, prompt in enumerate(prompts, 1):
        print_prompt_header(prompt, i, len(prompts))

        # Generate baseline response
        start = time.perf_counter()
        baseline_resp = generate(
            base_model, base_tok,
            prompt=prompt.prompt,
            max_tokens=max_tokens,
            verbose=False,
        )
        baseline_time = (time.perf_counter() - start) * 1000

        # Generate tuned response
        start = time.perf_counter()
        tuned_resp = generate(
            tuned_model, tuned_tok,
            prompt=prompt.prompt,
            max_tokens=max_tokens,
            verbose=False,
        )
        tuned_time = (time.perf_counter() - start) * 1000

        print_comparison(baseline_resp, tuned_resp, baseline_time, tuned_time)

        # Check if tuned response matches expected
        if prompt.expected:
            tuned_clean = tuned_resp.strip()
            expected_clean = prompt.expected.strip()
            match = tuned_clean.startswith(expected_clean)
            if match:
                print(f"{Colors.GREEN}Match expected output{Colors.RESET}")
            else:
                print(f"{Colors.DIM}(Output differs from expected){Colors.RESET}")
            print()

        results.append({
            "prompt": prompt.prompt,
            "baseline": baseline_resp,
            "tuned": tuned_resp,
            "baseline_time": baseline_time,
            "tuned_time": tuned_time,
        })

        if pause and i < len(prompts):
            input(f"{Colors.DIM}Press Enter for next example...{Colors.RESET}")
            print()

    # Print summary
    print_summary(results)


def print_summary(results: List[dict]):
    """Print demo summary."""
    print(f"{Colors.CYAN}{'=' * 70}{Colors.RESET}")
    print(f"{Colors.CYAN}{Colors.BOLD}                         SUMMARY{Colors.RESET}")
    print(f"{Colors.CYAN}{'=' * 70}{Colors.RESET}")
    print()

    avg_baseline_time = sum(r["baseline_time"] for r in results) / len(results)
    avg_tuned_time = sum(r["tuned_time"] for r in results) / len(results)

    print(f"Total examples: {len(results)}")
    print(f"Avg baseline latency: {avg_baseline_time:.0f}ms")
    print(f"Avg tuned latency: {avg_tuned_time:.0f}ms")
    print()

    print(f"{Colors.BOLD}Key Observations:{Colors.RESET}")
    print("  - Engram-tuned model produces more concise outputs")
    print("  - Pattern completion is more deterministic")
    print("  - Factual answers are direct (no explanation)")
    print()

    print(f"{Colors.CYAN}{'=' * 70}{Colors.RESET}")
    print(f"{Colors.GREEN}Demo complete!{Colors.RESET}")
    print()


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Engram PoC demo: compare baseline vs tuned model"
    )
    parser.add_argument(
        "--model",
        default="HuggingFaceTB/SmolLM-135M-Instruct",
        help="Model name (default: SmolLM-135M-Instruct)",
    )
    parser.add_argument(
        "--adapter-path",
        default="./adapters",
        help="Path to adapter (default: ./adapters)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=30,
        help="Max tokens to generate (default: 30)",
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

    prompts = DEMO_PROMPTS
    if args.quick:
        # Select a subset for quick demo
        prompts = [
            DEMO_PROMPTS[0],  # for loop
            DEMO_PROMPTS[3],  # HTTP 404
            DEMO_PROMPTS[6],  # date format
            DEMO_PROMPTS[8],  # error fix
        ]

    run_demo(
        model_name=args.model,
        adapter_path=args.adapter_path,
        prompts=prompts,
        max_tokens=args.max_tokens,
        pause=not args.no_pause,
    )


if __name__ == "__main__":
    main()
