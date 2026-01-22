"""JSONL dataset writer for MLX-LM training format."""

import json
from pathlib import Path
from typing import List

from .types import DatasetStats, PatternExample, TrainingExample


def write_jsonl(
    examples: List[PatternExample],
    output_path: Path,
) -> int:
    """Write pattern examples to a JSONL file in MLX-LM format.

    Args:
        examples: List of patterns to write
        output_path: Path to output JSONL file

    Returns:
        Number of examples written
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for pattern in examples:
            training_ex = TrainingExample.from_pattern(pattern)
            json.dump(training_ex.to_dict(), f, ensure_ascii=False)
            f.write("\n")

    return len(examples)


def split_dataset(
    examples: List[PatternExample],
    train_ratio: float = 0.8,
    seed: int = 42,
) -> tuple:
    """Split examples into train and validation sets.

    Args:
        examples: List of all examples
        train_ratio: Fraction for training (default: 0.8)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_examples, valid_examples)
    """
    import random

    # Make a copy to avoid modifying original
    shuffled = examples.copy()
    random.seed(seed)
    random.shuffle(shuffled)

    split_idx = int(len(shuffled) * train_ratio)
    train_examples = shuffled[:split_idx]
    valid_examples = shuffled[split_idx:]

    return train_examples, valid_examples


def write_dataset(
    examples: List[PatternExample],
    output_dir: Path,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> DatasetStats:
    """Write train and validation JSONL files.

    Args:
        examples: List of all examples
        output_dir: Directory to write files to
        train_ratio: Fraction for training
        seed: Random seed

    Returns:
        DatasetStats with information about the written dataset
    """
    train_examples, valid_examples = split_dataset(examples, train_ratio, seed)

    train_path = output_dir / "train.jsonl"
    valid_path = output_dir / "valid.jsonl"

    write_jsonl(train_examples, train_path)
    write_jsonl(valid_examples, valid_path)

    # Compute statistics
    all_examples = train_examples + valid_examples
    categories = {}
    sources = {}

    for ex in all_examples:
        categories[ex.category] = categories.get(ex.category, 0) + 1
        if ex.source_file:
            sources[ex.source_file] = sources.get(ex.source_file, 0) + 1

    return DatasetStats(
        total_examples=len(all_examples),
        train_examples=len(train_examples),
        valid_examples=len(valid_examples),
        categories=categories,
        sources=sources,
    )


def write_test_set(
    examples: List[PatternExample],
    output_path: Path,
    max_per_category: int = 5,
    seed: int = 42,
) -> int:
    """Write a test set with balanced category representation.

    Includes category metadata for evaluation.

    Args:
        examples: List of all examples
        output_path: Path to output JSONL file
        max_per_category: Maximum examples per category
        seed: Random seed

    Returns:
        Number of examples written
    """
    import random

    random.seed(seed)

    # Group by category
    by_category = {}
    for ex in examples:
        if ex.category not in by_category:
            by_category[ex.category] = []
        by_category[ex.category].append(ex)

    # Sample from each category
    test_examples = []
    for category, cat_examples in by_category.items():
        sampled = random.sample(cat_examples, min(max_per_category, len(cat_examples)))
        test_examples.extend(sampled)

    random.shuffle(test_examples)

    # Write with category metadata (for evaluation)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for pattern in test_examples:
            data = {
                "messages": [
                    {"role": "user", "content": pattern.input},
                    {"role": "assistant", "content": pattern.output},
                ],
                "category": pattern.category,
                "source_file": pattern.source_file,
            }
            json.dump(data, f, ensure_ascii=False)
            f.write("\n")

    return len(test_examples)
