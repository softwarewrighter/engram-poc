"""Evaluation metrics for Engram PoC.

Metrics focus on:
1. Accuracy: Does the model produce the expected output?
2. Consistency: Does the model produce the same output for the same input?
3. Latency: How fast is generation?
"""

import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class EvalResult:
    """Result of a single evaluation."""

    prompt: str
    expected: str
    actual: str
    correct: bool
    latency_ms: float
    category: str = "unknown"

    @property
    def match_type(self) -> str:
        """Classify the type of match."""
        if self.correct:
            return "exact"
        elif self.expected.strip() in self.actual:
            return "partial"
        else:
            return "none"


@dataclass
class ConsistencyResult:
    """Result of consistency evaluation (same input multiple times)."""

    prompt: str
    responses: List[str]
    expected: Optional[str] = None

    @property
    def unique_count(self) -> int:
        """Number of unique responses."""
        return len(set(r.strip() for r in self.responses))

    @property
    def consistency_score(self) -> float:
        """Score from 0 to 1, where 1 = all responses identical."""
        if len(self.responses) <= 1:
            return 1.0
        # Score based on how many responses match the most common one
        from collections import Counter
        counts = Counter(r.strip() for r in self.responses)
        most_common_count = counts.most_common(1)[0][1]
        return most_common_count / len(self.responses)

    @property
    def majority_response(self) -> str:
        """The most common response."""
        from collections import Counter
        counts = Counter(r.strip() for r in self.responses)
        return counts.most_common(1)[0][0]


@dataclass
class CategoryMetrics:
    """Metrics for a single category."""

    category: str
    total: int = 0
    correct: int = 0
    partial: int = 0
    consistency_scores: List[float] = field(default_factory=list)
    latencies_ms: List[float] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        """Exact match accuracy."""
        return self.correct / self.total if self.total > 0 else 0.0

    @property
    def partial_rate(self) -> float:
        """Rate of partial matches."""
        return self.partial / self.total if self.total > 0 else 0.0

    @property
    def avg_consistency(self) -> float:
        """Average consistency score."""
        if not self.consistency_scores:
            return 0.0
        return statistics.mean(self.consistency_scores)

    @property
    def avg_latency_ms(self) -> float:
        """Average latency in milliseconds."""
        if not self.latencies_ms:
            return 0.0
        return statistics.mean(self.latencies_ms)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "category": self.category,
            "total": self.total,
            "correct": self.correct,
            "partial": self.partial,
            "accuracy": round(self.accuracy, 4),
            "partial_rate": round(self.partial_rate, 4),
            "avg_consistency": round(self.avg_consistency, 4),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
        }


@dataclass
class EvalReport:
    """Complete evaluation report."""

    model_name: str
    adapter_path: Optional[str]
    total_examples: int
    accuracy_results: List[EvalResult] = field(default_factory=list)
    consistency_results: List[ConsistencyResult] = field(default_factory=list)
    category_metrics: Dict[str, CategoryMetrics] = field(default_factory=dict)

    @property
    def overall_accuracy(self) -> float:
        """Overall exact match accuracy."""
        if not self.accuracy_results:
            return 0.0
        correct = sum(1 for r in self.accuracy_results if r.correct)
        return correct / len(self.accuracy_results)

    @property
    def overall_partial_rate(self) -> float:
        """Overall partial match rate."""
        if not self.accuracy_results:
            return 0.0
        partial = sum(1 for r in self.accuracy_results if r.match_type == "partial")
        return partial / len(self.accuracy_results)

    @property
    def overall_consistency(self) -> float:
        """Overall average consistency."""
        if not self.consistency_results:
            return 0.0
        return statistics.mean(r.consistency_score for r in self.consistency_results)

    @property
    def avg_latency_ms(self) -> float:
        """Average latency across all evaluations."""
        if not self.accuracy_results:
            return 0.0
        return statistics.mean(r.latency_ms for r in self.accuracy_results)

    @property
    def latency_std_ms(self) -> float:
        """Standard deviation of latency."""
        if len(self.accuracy_results) < 2:
            return 0.0
        return statistics.stdev(r.latency_ms for r in self.accuracy_results)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "model_name": self.model_name,
            "adapter_path": self.adapter_path,
            "total_examples": self.total_examples,
            "overall": {
                "accuracy": round(self.overall_accuracy, 4),
                "partial_rate": round(self.overall_partial_rate, 4),
                "consistency": round(self.overall_consistency, 4),
                "avg_latency_ms": round(self.avg_latency_ms, 2),
                "latency_std_ms": round(self.latency_std_ms, 2),
            },
            "by_category": {
                cat: metrics.to_dict()
                for cat, metrics in self.category_metrics.items()
            },
            "raw_results": [
                {
                    "prompt": r.prompt[:50] + "..." if len(r.prompt) > 50 else r.prompt,
                    "expected": r.expected[:30] + "..." if len(r.expected) > 30 else r.expected,
                    "actual": r.actual[:30] + "..." if len(r.actual) > 30 else r.actual,
                    "correct": r.correct,
                    "match_type": r.match_type,
                    "latency_ms": round(r.latency_ms, 2),
                    "category": r.category,
                }
                for r in self.accuracy_results
            ],
        }


def calculate_improvement(baseline: EvalReport, tuned: EvalReport) -> dict:
    """Calculate improvement metrics between baseline and tuned model."""
    return {
        "accuracy": {
            "baseline": round(baseline.overall_accuracy, 4),
            "tuned": round(tuned.overall_accuracy, 4),
            "absolute_change": round(tuned.overall_accuracy - baseline.overall_accuracy, 4),
            "relative_change_pct": round(
                ((tuned.overall_accuracy - baseline.overall_accuracy) /
                 max(baseline.overall_accuracy, 0.001)) * 100, 2
            ),
        },
        "consistency": {
            "baseline": round(baseline.overall_consistency, 4),
            "tuned": round(tuned.overall_consistency, 4),
            "absolute_change": round(tuned.overall_consistency - baseline.overall_consistency, 4),
            "relative_change_pct": round(
                ((tuned.overall_consistency - baseline.overall_consistency) /
                 max(baseline.overall_consistency, 0.001)) * 100, 2
            ),
        },
        "latency_ms": {
            "baseline": round(baseline.avg_latency_ms, 2),
            "tuned": round(tuned.avg_latency_ms, 2),
            "change_ms": round(tuned.avg_latency_ms - baseline.avg_latency_ms, 2),
        },
    }
