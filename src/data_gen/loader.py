"""YAML pattern file loader."""

from pathlib import Path
from typing import List

import yaml

from .types import PatternExample, PatternSet


def load_pattern_file(filepath: Path) -> PatternSet:
    """Load a single pattern YAML file.

    Args:
        filepath: Path to the YAML file

    Returns:
        PatternSet containing all examples from the file

    Raises:
        FileNotFoundError: If the file doesn't exist
        yaml.YAMLError: If the file is not valid YAML
        ValueError: If the file structure is invalid
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Pattern file not found: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Invalid pattern file structure in {filepath}")

    name = data.get("name", filepath.stem)
    description = data.get("description", "")
    version = data.get("version", "1.0")
    raw_examples = data.get("examples", [])

    examples = []
    for idx, ex in enumerate(raw_examples):
        if not isinstance(ex, dict):
            raise ValueError(f"Invalid example at index {idx} in {filepath}")

        try:
            pattern = PatternExample(
                input=ex.get("input", ""),
                output=ex.get("output", ""),
                category=ex.get("category", "unknown"),
                source_file=filepath.name,
            )
            examples.append(pattern)
        except ValueError as e:
            raise ValueError(f"Invalid example at index {idx} in {filepath}: {e}")

    return PatternSet(
        name=name,
        description=description,
        version=version,
        examples=examples,
    )


def load_patterns(pattern_dir: Path) -> List[PatternSet]:
    """Load all pattern files from a directory.

    Args:
        pattern_dir: Directory containing YAML pattern files

    Returns:
        List of PatternSet objects, one per file

    Raises:
        FileNotFoundError: If the directory doesn't exist
    """
    if not pattern_dir.exists():
        raise FileNotFoundError(f"Pattern directory not found: {pattern_dir}")

    if not pattern_dir.is_dir():
        raise ValueError(f"Not a directory: {pattern_dir}")

    pattern_sets = []
    yaml_files = sorted(pattern_dir.glob("*.yaml")) + sorted(pattern_dir.glob("*.yml"))

    for yaml_file in yaml_files:
        pattern_set = load_pattern_file(yaml_file)
        pattern_sets.append(pattern_set)

    return pattern_sets


def get_all_examples(pattern_sets: List[PatternSet]) -> List[PatternExample]:
    """Flatten all pattern sets into a single list of examples.

    Args:
        pattern_sets: List of PatternSet objects

    Returns:
        Flat list of all PatternExample objects
    """
    examples = []
    for ps in pattern_sets:
        examples.extend(ps.examples)
    return examples
