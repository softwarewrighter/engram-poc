"""Training data generation for Engram PoC.

Loads patterns from YAML files and generates JSONL training data.

Usage:
    python -m src.data_gen
"""

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import yaml


@dataclass
class PatternExample:
    """A single pattern example."""
    input: str
    output: str
    category: str
    source_file: Optional[str] = None


# Instruction prefixes for augmentation
INSTRUCTION_PREFIXES = [
    "",
    "Complete: ",
    "Continue: ",
    "Finish this: ",
]

QUESTION_PREFIXES = [
    "",
    "Tell me: ",
    "Answer: ",
]


def load_patterns(patterns_dir: Path) -> List[PatternExample]:
    """Load all patterns from YAML files."""
    patterns = []

    for yaml_file in sorted(patterns_dir.glob("*.yaml")):
        with open(yaml_file, "r") as f:
            data = yaml.safe_load(f)

        source = yaml_file.stem
        for example in data.get("examples", []):
            patterns.append(PatternExample(
                input=example["input"],
                output=example["output"],
                category=example.get("category", source),
                source_file=source,
            ))

        print(f"  - {source}: {len(data.get('examples', []))} patterns")

    return patterns


def augment_pattern(pattern: PatternExample) -> List[PatternExample]:
    """Create variations of a pattern."""
    results = [pattern]

    is_question = pattern.input.startswith("Q:")
    is_fix = pattern.input.startswith("Fix")
    is_format = "->" in pattern.input

    if is_question:
        for prefix in QUESTION_PREFIXES[1:]:
            results.append(PatternExample(
                input=prefix + pattern.input,
                output=pattern.output,
                category=pattern.category,
                source_file=pattern.source_file,
            ))
    elif not is_fix and not is_format:
        for prefix in INSTRUCTION_PREFIXES[1:]:
            if not pattern.input.startswith(prefix):
                results.append(PatternExample(
                    input=prefix + pattern.input,
                    output=pattern.output,
                    category=pattern.category,
                    source_file=pattern.source_file,
                ))

    return results


def write_jsonl(examples: List[PatternExample], output_path: Path, include_category: bool = False):
    """Write examples to JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for ex in examples:
            data = {
                "messages": [
                    {"role": "user", "content": ex.input},
                    {"role": "assistant", "content": ex.output},
                ]
            }
            if include_category:
                data["category"] = ex.category
            json.dump(data, f)
            f.write("\n")


def generate_dataset(
    patterns_dir: Path,
    output_dir: Path,
    train_ratio: float = 0.8,
    seed: int = 42,
):
    """Generate train/valid/test datasets."""
    print(f"Loading patterns from: {patterns_dir}")
    patterns = load_patterns(patterns_dir)
    print(f"Total raw patterns: {len(patterns)}")

    # Augment
    augmented = []
    for p in patterns:
        augmented.extend(augment_pattern(p))
    print(f"After augmentation: {len(augmented)}")

    # Shuffle and split
    random.seed(seed)
    random.shuffle(augmented)

    split_idx = int(len(augmented) * train_ratio)
    train_examples = augmented[:split_idx]
    valid_examples = augmented[split_idx:]

    # Create test set (balanced by category)
    by_category = {}
    for ex in patterns:
        if ex.category not in by_category:
            by_category[ex.category] = []
        by_category[ex.category].append(ex)

    test_examples = []
    for cat_examples in by_category.values():
        test_examples.extend(random.sample(cat_examples, min(5, len(cat_examples))))
    random.shuffle(test_examples)

    # Write files
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(train_examples, output_dir / "train.jsonl")
    write_jsonl(valid_examples, output_dir / "valid.jsonl")
    write_jsonl(test_examples, output_dir / "test.jsonl", include_category=True)

    print(f"\nDataset written to: {output_dir}")
    print(f"  Training:   {len(train_examples)}")
    print(f"  Validation: {len(valid_examples)}")
    print(f"  Test:       {len(test_examples)}")


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate training data")
    parser.add_argument(
        "--patterns-dir",
        type=Path,
        default=Path("../data/patterns"),
        help="Directory containing pattern YAML files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./data"),
        help="Output directory for JSONL files",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    # Use parent patterns if local doesn't exist
    if not args.patterns_dir.exists():
        args.patterns_dir = Path(__file__).parent.parent.parent / "data" / "patterns"

    generate_dataset(args.patterns_dir, args.output_dir, seed=args.seed)


if __name__ == "__main__":
    main()
