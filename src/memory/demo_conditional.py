"""
Demo: Conditional Engram Routing

Shows how conditional gating routes lookup queries to Engram memory
while bypassing memory for general queries.

Run:
    python -m src.memory.demo_conditional
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .model_wrapper import EngramModelWrapper
from .conditional_engram import (
    ConditionalEngramWrapper,
    LookupPatternDetector,
    create_conditional_engram,
)


def demo_pattern_detection():
    """Demo 1: Show pattern detection in action."""
    print("\n" + "=" * 60)
    print("Demo 1: Pattern Detection")
    print("=" * 60)

    detector = LookupPatternDetector()

    test_cases = [
        # High confidence - explicit lookup patterns
        ("CAPITAL:France", "Explicit lookup prefix"),
        ("PORT:SSH", "Technical lookup"),
        ("ACRONYM:GPU", "Acronym expansion"),
        ("HTTP:404", "HTTP code lookup"),
        ("ELEMENT_NAME:Fe", "Chemical element"),

        # Medium confidence - factual questions
        ("What is the capital of France?", "Factual question"),
        ("What does API stand for?", "Definition question"),
        ("Define recursion", "Definition request"),

        # Low confidence - general queries
        ("Write a poem about cats", "Creative task"),
        ("How do I fix this bug?", "Problem solving"),
        ("Tell me a story", "General request"),
        ("The weather is nice today", "Statement"),
    ]

    print("\n{:<40} {:<20} {:<10}".format("Input", "Type", "Confidence"))
    print("-" * 70)

    for text, description in test_cases:
        confidence = detector(text)
        conf_str = f"{confidence:.1f}"
        if confidence >= 0.7:
            status = "→ ENGRAM"
        elif confidence >= 0.3:
            status = "→ MAYBE"
        else:
            status = "→ BYPASS"

        print(f"{text:<40} {description:<20} {conf_str:<5} {status}")


def demo_conditional_routing():
    """Demo 2: Show conditional routing with actual model."""
    print("\n" + "=" * 60)
    print("Demo 2: Conditional Routing (with model)")
    print("=" * 60)

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load models
    print("\nLoading models...")
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M-Instruct")
    tokenizer.pad_token = tokenizer.eos_token

    # Base model (no Engram)
    base_model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-135M-Instruct")
    base_model = base_model.to(device)
    base_model.eval()

    # Engram model (always uses memory)
    engram_base = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-135M-Instruct")
    engram_model = EngramModelWrapper(engram_base, memory_size=500, freeze_base=True)

    # Load trained weights if available
    import os
    weights_path = "adapters-engram-exact/engram_weights.pt"
    if os.path.exists(weights_path):
        engram_model.load_engram_weights(weights_path)
        print(f"Loaded Engram weights from {weights_path}")
    else:
        print("No pre-trained weights found, using fresh Engram")

    engram_model = engram_model.to(device)
    engram_model.eval()

    # Conditional model (smart routing)
    conditional_model = ConditionalEngramWrapper(
        engram_model=engram_model,
        tokenizer=tokenizer,
        use_learned_gate=False,  # Pure rule-based for demo
    )

    # Test cases
    test_inputs = [
        # Lookup queries (should route to Engram)
        "ACRONYM:GPU",
        "PORT:SSH",
        "CAPITAL:France",

        # General queries (should bypass Engram)
        "Write a haiku about coding",
        "How do I sort a list in Python?",
    ]

    print("\n" + "-" * 70)
    detector = LookupPatternDetector()

    for prompt in test_inputs:
        confidence = detector(prompt)
        route = "ENGRAM" if confidence >= 0.5 else "BYPASS"

        # Format with chat template
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(formatted, return_tensors="pt").to(device)

        # Generate with conditional model
        with torch.no_grad():
            outputs = conditional_model.generate(
                **inputs,
                max_new_tokens=30,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False,
            )
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

        print(f"\nInput: {prompt}")
        print(f"Route: {route} (confidence: {confidence:.1f})")
        print(f"Output: {response[:80]}...")


def demo_comparison():
    """Demo 3: Compare base, Engram, and conditional on mixed queries."""
    print("\n" + "=" * 60)
    print("Demo 3: Three-Way Comparison")
    print("=" * 60)

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M-Instruct")
    tokenizer.pad_token = tokenizer.eos_token

    # Load base model
    print("\nLoading base model...")
    base_model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-135M-Instruct")
    base_model = base_model.to(device)
    base_model.eval()

    # Load Engram model
    print("Loading Engram model...")
    engram_base = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-135M-Instruct")
    engram_model = EngramModelWrapper(engram_base, memory_size=500, freeze_base=True)

    import os
    if os.path.exists("adapters-engram-exact/engram_weights.pt"):
        engram_model.load_engram_weights("adapters-engram-exact/engram_weights.pt")
    engram_model = engram_model.to(device)
    engram_model.eval()

    # Conditional model
    conditional = ConditionalEngramWrapper(
        engram_model=engram_model,
        tokenizer=tokenizer,
        use_learned_gate=False,
    )

    # Test cases with expected behavior
    test_cases = [
        # Lookup queries - Engram should help
        ("ACRONYM:API", "Application Programming Interface", "lookup"),
        ("ACRONYM:CPU", "Central Processing Unit", "lookup"),

        # General queries - Engram may hurt
        ("What is 2+2?", "4", "general"),
        ("Say hello", "hello", "general"),
    ]

    detector = LookupPatternDetector()

    print("\n{:<20} {:<15} {:<40}".format("Query", "Type", "Response"))
    print("-" * 80)

    for prompt, expected, query_type in test_cases:
        confidence = detector(prompt)

        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(formatted, return_tensors="pt").to(device)

        # Base model
        with torch.no_grad():
            base_out = base_model.generate(**inputs, max_new_tokens=20, pad_token_id=tokenizer.eos_token_id, do_sample=False)
        base_resp = tokenizer.decode(base_out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

        # Conditional model
        with torch.no_grad():
            cond_out = conditional.generate(**inputs, max_new_tokens=20, pad_token_id=tokenizer.eos_token_id, do_sample=False)
        cond_resp = tokenizer.decode(cond_out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

        route = "ENGRAM" if confidence >= 0.5 else "BYPASS"
        correct_base = expected.lower() in base_resp.lower()
        correct_cond = expected.lower() in cond_resp.lower()

        print(f"\n{prompt}")
        print(f"  Route: {route} | Expected: {expected}")
        print(f"  Base:        {base_resp[:50]:<50} {'✓' if correct_base else '✗'}")
        print(f"  Conditional: {cond_resp[:50]:<50} {'✓' if correct_cond else '✗'}")


def main():
    """Run all demos."""
    print("=" * 60)
    print("Conditional Engram Demo")
    print("Smart routing between base model and memory")
    print("=" * 60)

    # Demo 1: Pattern detection (no model needed)
    demo_pattern_detection()

    # Demo 2: Conditional routing with model
    demo_conditional_routing()

    # Demo 3: Three-way comparison
    demo_comparison()

    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)
    print("\nKey takeaways:")
    print("1. Pattern detection identifies lookup queries (CAPITAL:, PORT:, etc.)")
    print("2. Conditional routing sends lookups to Engram, bypasses for general queries")
    print("3. This prevents Engram from hurting performance on non-lookup tasks")


if __name__ == "__main__":
    main()
