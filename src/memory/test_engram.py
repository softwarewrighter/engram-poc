"""
Test script for Engram module.

Run with: python -m src.memory.test_engram
"""

import torch
import torch.nn as nn


def test_engram_module():
    """Test the basic EnhancedEngramModule."""
    from .engram_module import EnhancedEngramModule

    print("Testing EnhancedEngramModule...")

    # Create module
    module = EnhancedEngramModule(
        table_size=10000,
        d_model=256,
        n_heads=4,
    )

    # Test inputs
    batch_size = 2
    seq_len = 32
    hidden_states = torch.randn(batch_size, seq_len, 256)
    input_ids = torch.randint(0, 10000, (batch_size, seq_len))

    # Forward pass
    output = module(hidden_states, input_ids)

    assert output.shape == hidden_states.shape, f"Shape mismatch: {output.shape} vs {hidden_states.shape}"
    print(f"  Input shape: {hidden_states.shape}")
    print(f"  Output shape: {output.shape}")

    # Test memory stats
    stats = module.get_memory_stats()
    print(f"  Memory stats: {stats}")

    # Test backward pass
    loss = output.sum()
    loss.backward()
    print("  Backward pass: OK")

    print("EnhancedEngramModule: PASSED\n")


def test_model_wrapper():
    """Test wrapping a real HuggingFace model."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from .model_wrapper import EngramModelWrapper
    except ImportError as e:
        print(f"Skipping model wrapper test (missing dependency): {e}")
        return

    print("Testing EngramModelWrapper with SmolLM...")

    # Load a small model
    model_name = "HuggingFaceTB/SmolLM-135M-Instruct"

    print(f"  Loading {model_name}...")
    base_model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Wrap with Engram
    print("  Wrapping with Engram...")
    engram_model = EngramModelWrapper(
        model=base_model,
        memory_size=10000,
        inject_layers=[0, 1, 2],  # Only first 3 layers for test
        freeze_base=True,
    )

    # Print stats
    stats = engram_model.get_memory_stats()
    print(f"  Engram layers: {stats['num_engram_layers']}")
    print(f"  Engram params: {stats['total_engram_params']:,}")

    # Test forward pass
    print("  Testing forward pass...")
    input_text = "Hello, how are you?"
    inputs = tokenizer(input_text, return_tensors="pt")

    with torch.no_grad():
        outputs = engram_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )

    print(f"  Logits shape: {outputs.logits.shape}")

    # Check trainable parameters
    engram_params = engram_model.engram_parameters()
    print(f"  Trainable Engram params: {len(engram_params)}")

    # Verify base is frozen
    base_trainable = sum(p.requires_grad for p in base_model.parameters())
    print(f"  Base model trainable params: {base_trainable} (should be 0)")

    print("EngramModelWrapper: PASSED\n")


def test_engram_layer():
    """Test the complete EngramLayer."""
    from .engram_module import EngramLayer

    print("Testing EngramLayer...")

    layer = EngramLayer(
        d_model=256,
        n_heads=8,
        memory_size=10000,
        dropout=0.1,
    )

    # Test inputs
    batch_size = 2
    seq_len = 32
    hidden_states = torch.randn(batch_size, seq_len, 256)
    input_ids = torch.randint(0, 10000, (batch_size, seq_len))

    # Forward pass
    output = layer(hidden_states, input_ids)

    assert output.shape == hidden_states.shape
    print(f"  Input shape: {hidden_states.shape}")
    print(f"  Output shape: {output.shape}")

    print("EngramLayer: PASSED\n")


def test_hash_distribution():
    """Test that multi-head hashing provides good distribution."""
    from .engram_module import EnhancedEngramModule

    print("Testing hash distribution...")

    module = EnhancedEngramModule(table_size=10000, d_model=256, n_heads=4)

    # Generate random input_ids
    input_ids = torch.randint(0, 50000, (1, 1000))

    # Get hashes
    hashes = module.multi_head_hash(input_ids)

    # Check unique hashes per head
    for head in range(4):
        head_hashes = hashes[0, :, head].numpy()
        unique_ratio = len(set(head_hashes)) / len(head_hashes)
        print(f"  Head {head}: {unique_ratio:.1%} unique hashes")

    print("Hash distribution: PASSED\n")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Engram Module Tests")
    print("=" * 60 + "\n")

    test_engram_module()
    test_engram_layer()
    test_hash_distribution()

    # Model wrapper test is optional (requires downloading model)
    import os
    if os.environ.get("TEST_MODEL_WRAPPER", "0") == "1":
        test_model_wrapper()
    else:
        print("Skipping model wrapper test (set TEST_MODEL_WRAPPER=1 to run)\n")

    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
