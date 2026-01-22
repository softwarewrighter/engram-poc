"""Main entry point for training data generation.

Usage:
    python -m src.data_gen.generate [--pattern-dir PATH] [--output-dir PATH] [--no-augment]
"""

import argparse
from pathlib import Path

from .augment import augment_all
from .loader import get_all_examples, load_patterns
from .writer import write_dataset, write_test_set


def generate_dataset(
    pattern_dir: Path,
    output_dir: Path,
    augment: bool = True,
    train_ratio: float = 0.8,
    seed: int = 42,
    verbose: bool = True,
) -> None:
    """Generate training dataset from pattern files.

    Args:
        pattern_dir: Directory containing YAML pattern files
        output_dir: Directory to write output files
        augment: Whether to apply data augmentation
        train_ratio: Fraction of data for training
        seed: Random seed for reproducibility
        verbose: Print progress information
    """
    if verbose:
        print(f"Loading patterns from: {pattern_dir}")

    # Load all pattern files
    pattern_sets = load_patterns(pattern_dir)

    if verbose:
        for ps in pattern_sets:
            print(f"  - {ps.name}: {ps.count} patterns")

    # Flatten into single list
    all_examples = get_all_examples(pattern_sets)

    if verbose:
        print(f"Total raw patterns: {len(all_examples)}")

    # Apply augmentation
    if augment:
        all_examples = augment_all(all_examples)
        if verbose:
            print(f"After augmentation: {len(all_examples)}")

    # Write train/valid split
    stats = write_dataset(
        all_examples,
        output_dir,
        train_ratio=train_ratio,
        seed=seed,
    )

    if verbose:
        print(f"\nDataset written to: {output_dir}")
        print(stats.summary())

    # Also write a test set (for evaluation)
    test_path = output_dir / "test.jsonl"
    test_count = write_test_set(
        get_all_examples(pattern_sets),  # Use original (non-augmented) for test
        test_path,
        max_per_category=3,
        seed=seed,
    )

    if verbose:
        print(f"\nTest set written: {test_count} examples")
        print(f"  -> {test_path}")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate training data from pattern files"
    )
    parser.add_argument(
        "--pattern-dir",
        type=Path,
        default=Path("data/patterns"),
        help="Directory containing YAML pattern files (default: data/patterns)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Directory to write output files (default: data)",
    )
    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="Disable data augmentation",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Fraction of data for training (default: 0.8)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress output",
    )

    args = parser.parse_args()

    generate_dataset(
        pattern_dir=args.pattern_dir,
        output_dir=args.output_dir,
        augment=not args.no_augment,
        train_ratio=args.train_ratio,
        seed=args.seed,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
