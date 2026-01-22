"""Training data augmentation strategies.

This module creates variations of training patterns to improve model robustness.
The augmentation simulates how users might phrase the same query differently,
helping the Engram-tuned model recognize patterns regardless of phrasing.

Augmentation types:
- Instruction prefixes: "Complete:", "Continue:", "Finish this:"
- Question rephrases: "Tell me:", "Answer:"

This increases training data ~2.5x (131 patterns -> 337 examples) without
requiring manual annotation.
"""

from typing import List

from .types import PatternExample


# Instruction prefixes to create variations
INSTRUCTION_PREFIXES = [
    "",  # No prefix (original)
    "Complete: ",
    "Continue: ",
    "Finish this: ",
    "What comes next: ",
]

# Question prefixes for fact-style patterns
QUESTION_REPHRASES = [
    "",  # Original
    "Tell me: ",
    "Answer: ",
]


def augment_with_prefixes(
    pattern: PatternExample,
    prefixes: List[str] = None,
) -> List[PatternExample]:
    """Create variations of a pattern by adding instruction prefixes.

    Args:
        pattern: Original pattern to augment
        prefixes: List of prefixes to apply (default: INSTRUCTION_PREFIXES)

    Returns:
        List of augmented patterns including the original
    """
    if prefixes is None:
        prefixes = INSTRUCTION_PREFIXES

    variations = []
    for prefix in prefixes:
        # Skip empty prefix if input already starts with an instruction-like word
        if not prefix and pattern.input[0].isupper():
            variations.append(pattern)
            continue

        # Skip if prefix would be redundant
        if prefix and pattern.input.startswith(prefix):
            continue

        augmented = PatternExample(
            input=prefix + pattern.input if prefix else pattern.input,
            output=pattern.output,
            category=pattern.category,
            source_file=pattern.source_file,
        )
        variations.append(augmented)

    return variations


def augment_questions(pattern: PatternExample) -> List[PatternExample]:
    """Create variations for question-style patterns.

    Args:
        pattern: Original Q&A pattern

    Returns:
        List of augmented patterns
    """
    # Only augment patterns that look like Q&A
    if not pattern.input.startswith("Q:"):
        return [pattern]

    variations = [pattern]  # Always include original

    for prefix in QUESTION_REPHRASES:
        if not prefix:
            continue

        augmented = PatternExample(
            input=prefix + pattern.input,
            output=pattern.output,
            category=pattern.category,
            source_file=pattern.source_file,
        )
        variations.append(augmented)

    return variations


def augment_example(
    pattern: PatternExample,
    use_prefixes: bool = True,
    use_question_rephrases: bool = True,
) -> List[PatternExample]:
    """Apply all relevant augmentations to a pattern.

    Args:
        pattern: Original pattern
        use_prefixes: Whether to add instruction prefixes
        use_question_rephrases: Whether to rephrase questions

    Returns:
        List of augmented patterns including original
    """
    # Start with the original
    results = [pattern]

    # Determine pattern type and apply appropriate augmentations
    is_question = pattern.input.startswith("Q:")
    is_fix = pattern.input.startswith("Fix")
    is_format = "->" in pattern.input or pattern.input.startswith("Format")

    if is_question and use_question_rephrases:
        # Q&A patterns get question rephrasing
        results = augment_questions(pattern)
    elif not is_fix and not is_format and use_prefixes:
        # Code patterns get instruction prefixes
        # But not Fix: or Format: patterns which already have clear prefixes
        results = augment_with_prefixes(pattern)

    return results


def augment_all(
    patterns: List[PatternExample],
    use_prefixes: bool = True,
    use_question_rephrases: bool = True,
) -> List[PatternExample]:
    """Augment all patterns in a list.

    Args:
        patterns: List of original patterns
        use_prefixes: Whether to add instruction prefixes
        use_question_rephrases: Whether to rephrase questions

    Returns:
        List of all augmented patterns
    """
    augmented = []
    for pattern in patterns:
        variations = augment_example(
            pattern,
            use_prefixes=use_prefixes,
            use_question_rephrases=use_question_rephrases,
        )
        augmented.extend(variations)
    return augmented
