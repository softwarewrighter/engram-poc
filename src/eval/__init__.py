"""Evaluation framework for Engram PoC.

This module provides tools to evaluate model performance:
- Accuracy: exact and partial match rates
- Consistency: same-input-same-output rate
- Latency: generation speed
- Comparison: baseline vs tuned model
"""

from .compare import compare_models, generate_markdown_report, main
from .metrics import (
    CategoryMetrics,
    ConsistencyResult,
    EvalReport,
    EvalResult,
    calculate_improvement,
)
from .runner import EvalConfig, Evaluator, load_test_cases, run_evaluation

__all__ = [
    # Metrics
    "EvalResult",
    "ConsistencyResult",
    "CategoryMetrics",
    "EvalReport",
    "calculate_improvement",
    # Runner
    "EvalConfig",
    "Evaluator",
    "load_test_cases",
    "run_evaluation",
    # Compare
    "compare_models",
    "generate_markdown_report",
    "main",
]
