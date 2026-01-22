"""Training data generation for Engram PoC.

This module provides tools to generate training data from pattern definitions:
- Load YAML pattern files
- Apply data augmentation
- Write MLX-LM compatible JSONL files
"""

from .augment import augment_all, augment_example
from .generate import generate_dataset, main
from .loader import get_all_examples, load_pattern_file, load_patterns
from .types import DatasetStats, PatternExample, PatternSet, TrainingExample
from .writer import split_dataset, write_dataset, write_jsonl, write_test_set

__all__ = [
    # Types
    "PatternExample",
    "PatternSet",
    "TrainingExample",
    "DatasetStats",
    # Loader
    "load_pattern_file",
    "load_patterns",
    "get_all_examples",
    # Augmentation
    "augment_example",
    "augment_all",
    # Writer
    "write_jsonl",
    "split_dataset",
    "write_dataset",
    "write_test_set",
    # Main
    "generate_dataset",
    "main",
]
