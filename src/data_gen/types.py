"""Data types for pattern-based training data generation."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class PatternExample:
    """A single pattern example with input and expected output."""

    input: str
    output: str
    category: str
    source_file: Optional[str] = None

    def __post_init__(self):
        """Validate the pattern example."""
        if not self.input:
            raise ValueError("Pattern input cannot be empty")
        if not self.output:
            raise ValueError("Pattern output cannot be empty")


@dataclass
class PatternSet:
    """A collection of related patterns from a single source file."""

    name: str
    description: str
    version: str
    examples: List[PatternExample] = field(default_factory=list)

    @property
    def count(self) -> int:
        """Return the number of examples in this pattern set."""
        return len(self.examples)

    @property
    def categories(self) -> set:
        """Return unique categories in this pattern set."""
        return {ex.category for ex in self.examples}


@dataclass
class TrainingExample:
    """A training example in MLX-LM chat format."""

    messages: List[dict]

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {"messages": self.messages}

    @classmethod
    def from_pattern(cls, pattern: PatternExample) -> "TrainingExample":
        """Create a training example from a pattern."""
        return cls(
            messages=[
                {"role": "user", "content": pattern.input},
                {"role": "assistant", "content": pattern.output},
            ]
        )


@dataclass
class DatasetStats:
    """Statistics about a generated dataset."""

    total_examples: int
    train_examples: int
    valid_examples: int
    categories: dict  # category -> count
    sources: dict  # source file -> count

    def summary(self) -> str:
        """Return a human-readable summary."""
        lines = [
            f"Total examples: {self.total_examples}",
            f"  Training: {self.train_examples}",
            f"  Validation: {self.valid_examples}",
            f"Categories ({len(self.categories)}):",
        ]
        for cat, count in sorted(self.categories.items()):
            lines.append(f"  - {cat}: {count}")
        lines.append(f"Sources ({len(self.sources)}):")
        for src, count in sorted(self.sources.items()):
            lines.append(f"  - {src}: {count}")
        return "\n".join(lines)
