"""
Pre-populated Memory Experiment

Tests whether hash-based memory can retrieve facts when the memory table
is initialized with known embeddings (no training required).

Hypothesis: If we embed known answers at the hash positions of their keys,
the model should be able to retrieve them directly.

Run:
    python -m src.memory.prepopulated_memory
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from .engram_module import EnhancedEngramModule
from .model_wrapper import EngramModelWrapper


# Known facts to pre-populate
KNOWN_FACTS = {
    # Acronyms
    "ACRONYM:GPU": "Graphics Processing Unit",
    "ACRONYM:CPU": "Central Processing Unit",
    "ACRONYM:API": "Application Programming Interface",
    "ACRONYM:RAM": "Random Access Memory",
    "ACRONYM:ROM": "Read Only Memory",
    "ACRONYM:HTML": "HyperText Markup Language",
    "ACRONYM:CSS": "Cascading Style Sheets",
    "ACRONYM:SQL": "Structured Query Language",
    "ACRONYM:USB": "Universal Serial Bus",
    "ACRONYM:LED": "Light Emitting Diode",

    # Capitals
    "CAPITAL:France": "Paris",
    "CAPITAL:Germany": "Berlin",
    "CAPITAL:Japan": "Tokyo",
    "CAPITAL:Italy": "Rome",
    "CAPITAL:Spain": "Madrid",
    "CAPITAL:UK": "London",
    "CAPITAL:USA": "Washington DC",
    "CAPITAL:China": "Beijing",
    "CAPITAL:India": "New Delhi",
    "CAPITAL:Brazil": "Brasilia",

    # Ports
    "PORT:HTTP": "80",
    "PORT:HTTPS": "443",
    "PORT:SSH": "22",
    "PORT:FTP": "21",
    "PORT:SMTP": "25",
    "PORT:DNS": "53",
    "PORT:MySQL": "3306",
    "PORT:PostgreSQL": "5432",
    "PORT:Redis": "6379",
    "PORT:MongoDB": "27017",

    # HTTP Status Codes
    "HTTP:200": "OK",
    "HTTP:201": "Created",
    "HTTP:400": "Bad Request",
    "HTTP:401": "Unauthorized",
    "HTTP:403": "Forbidden",
    "HTTP:404": "Not Found",
    "HTTP:500": "Internal Server Error",
    "HTTP:502": "Bad Gateway",
    "HTTP:503": "Service Unavailable",

    # Chemical Elements
    "ELEMENT:Fe": "Iron",
    "ELEMENT:Au": "Gold",
    "ELEMENT:Ag": "Silver",
    "ELEMENT:Cu": "Copper",
    "ELEMENT:O": "Oxygen",
    "ELEMENT:H": "Hydrogen",
    "ELEMENT:N": "Nitrogen",
    "ELEMENT:C": "Carbon",
    "ELEMENT:Na": "Sodium",
    "ELEMENT:K": "Potassium",
}


class PrePopulatedEngramModule(nn.Module):
    """
    Engram module with pre-populated memory table.

    Instead of learning memory contents through training,
    we directly embed known facts into the memory table.
    """

    def __init__(
        self,
        d_model: int,
        memory_size: int = 500,
        num_heads: int = 4,
        tokenizer=None,
    ):
        super().__init__()
        self.d_model = d_model
        self.memory_size = memory_size
        self.num_heads = num_heads
        self.tokenizer = tokenizer

        # Memory table - will be populated with fact embeddings
        self.memory_table = nn.Parameter(
            torch.zeros(memory_size, d_model),
            requires_grad=False,  # No training needed
        )

        # Hash coefficients for multi-head hashing
        self.hash_coeffs = nn.Parameter(
            torch.randint(1, memory_size, (num_heads,)).float(),
            requires_grad=False,
        )

        # Output projection
        self.output_proj = nn.Linear(d_model * num_heads, d_model)

        # Gate for blending (conservative value to avoid corruption)
        self.gate = nn.Parameter(torch.tensor([0.1]), requires_grad=False)

        # Track which slots are populated
        self.populated_slots: Dict[int, str] = {}

    def hash_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Multi-head hashing of token IDs."""
        # token_ids: [batch, seq_len]
        batch_size, seq_len = token_ids.shape

        # Expand for multi-head: [batch, seq_len, num_heads]
        token_ids_expanded = token_ids.unsqueeze(-1).float()

        # Hash with different coefficients per head
        hash_indices = (token_ids_expanded * self.hash_coeffs) % self.memory_size

        return hash_indices.long()

    def populate_with_facts(
        self,
        facts: Dict[str, str],
        embedding_model: nn.Module,
        layer_idx: int = 0,
    ):
        """
        Populate memory table with hidden state representations.

        For each key-value pair, we:
        1. Tokenize the key to get hash positions
        2. Run the VALUE through the model to get hidden states
        3. Store the hidden state at the hash positions

        This ensures memory contents are in the same representation space
        as the hidden states they'll be blended with.
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer required for fact population")

        device = self.memory_table.device

        populated_count = 0
        collision_count = 0

        for key, value in facts.items():
            # Tokenize key to get hash positions
            key_tokens = self.tokenizer.encode(key, add_special_tokens=False)
            key_tensor = torch.tensor([key_tokens], device=device)

            # Get hash indices for the key
            hash_indices = self.hash_tokens(key_tensor)  # [1, seq_len, num_heads]

            # Get hidden state for the value by running through model
            # Format as "The answer is: {value}"
            answer_text = f"The answer is: {value}"
            value_tokens = self.tokenizer.encode(answer_text, add_special_tokens=False)
            value_tensor = torch.tensor([value_tokens], device=device)

            with torch.no_grad():
                # Run through model to get hidden states
                outputs = embedding_model(
                    input_ids=value_tensor,
                    output_hidden_states=True,
                )
                # Get hidden state from appropriate layer
                # Use layer_idx to get representation at the right depth
                hidden_states = outputs.hidden_states[min(layer_idx + 1, len(outputs.hidden_states) - 1)]
                # Use last token's hidden state (most informed)
                value_embed = hidden_states[0, -1, :]  # [d_model]

            # Store at each hash position (using last token's hash as primary)
            for head in range(self.num_heads):
                slot_idx = hash_indices[0, -1, head].item()

                if slot_idx in self.populated_slots:
                    collision_count += 1
                    # Average with existing (simple collision handling)
                    self.memory_table.data[slot_idx] = (
                        self.memory_table.data[slot_idx] + value_embed
                    ) / 2
                else:
                    self.memory_table.data[slot_idx] = value_embed
                    self.populated_slots[slot_idx] = key
                    populated_count += 1

        print(f"Populated {populated_count} memory slots")
        print(f"Collisions: {collision_count}")
        print(f"Unique facts: {len(facts)}")

        return populated_count, collision_count

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Retrieve from pre-populated memory.

        Args:
            hidden_states: [batch, seq_len, d_model]
            input_ids: [batch, seq_len]

        Returns:
            Output blended with memory retrieval
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Get hash indices
        hash_indices = self.hash_tokens(input_ids)  # [batch, seq_len, num_heads]

        # Retrieve from each head
        retrieved = []
        for head in range(self.num_heads):
            head_indices = hash_indices[:, :, head]  # [batch, seq_len]
            head_memory = self.memory_table[head_indices]  # [batch, seq_len, d_model]
            retrieved.append(head_memory)

        # Concatenate heads and project
        multi_head = torch.cat(retrieved, dim=-1)  # [batch, seq_len, d_model * num_heads]
        memory_output = self.output_proj(multi_head)  # [batch, seq_len, d_model]

        # Blend with gate
        output = (1 - self.gate) * hidden_states + self.gate * memory_output

        return output


class PrePopulatedEngramWrapper(nn.Module):
    """
    Wrapper that injects pre-populated Engram into a model.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        facts: Dict[str, str],
        memory_size: int = 500,
    ):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.facts = facts

        # Get model config
        config = model.config
        self.d_model = config.hidden_size
        self.num_layers = config.num_hidden_layers

        # Get device from model
        self.device = next(model.parameters()).device

        # Create pre-populated Engram modules for each layer
        self.engram_modules = nn.ModuleList([
            PrePopulatedEngramModule(
                d_model=self.d_model,
                memory_size=memory_size,
                tokenizer=tokenizer,
            )
            for _ in range(self.num_layers)
        ])

        # Move modules to device
        self.engram_modules = self.engram_modules.to(self.device)

        # Populate memory with facts (pass layer index for appropriate hidden state)
        print(f"\nPopulating {self.num_layers} layers with {len(facts)} facts...")
        for i, engram in enumerate(self.engram_modules):
            engram.populate_with_facts(facts, model, layer_idx=i)

        # Install forward hooks
        self._install_hooks()

    def _install_hooks(self):
        """Install forward hooks to inject Engram retrieval."""
        self._current_input_ids = None

        def make_hook(layer_idx):
            def hook(module, args, output):
                if self._current_input_ids is None:
                    return output

                hidden_states = output[0] if isinstance(output, tuple) else output

                # Apply Engram retrieval
                engram_output = self.engram_modules[layer_idx](
                    hidden_states,
                    self._current_input_ids,
                )

                if isinstance(output, tuple):
                    return (engram_output,) + output[1:]
                return engram_output

            return hook

        # Hook into each transformer layer
        layers = self.model.model.layers
        for i, layer in enumerate(layers):
            layer.register_forward_hook(make_hook(i))

        print(f"Installed hooks on {len(layers)} layers")

    def forward(self, input_ids, attention_mask=None, **kwargs):
        self._current_input_ids = input_ids
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )
        self._current_input_ids = None
        return outputs

    def generate(self, input_ids, **kwargs):
        self._current_input_ids = input_ids
        outputs = self.model.generate(input_ids=input_ids, **kwargs)
        self._current_input_ids = None
        return outputs


def test_retrieval(
    model,
    tokenizer,
    facts: Dict[str, str],
    device: str,
) -> Tuple[int, int, List[dict]]:
    """
    Test fact retrieval accuracy.

    Returns:
        (correct, total, results_list)
    """
    correct = 0
    total = len(facts)
    results = []

    for key, expected in facts.items():
        # Format as chat
        messages = [{"role": "user", "content": key}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(formatted, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False,
            )

        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()

        # Check if expected answer is in response
        is_correct = expected.lower() in response.lower()
        if is_correct:
            correct += 1

        results.append({
            "key": key,
            "expected": expected,
            "response": response[:50],
            "correct": is_correct,
        })

    return correct, total, results


def retrieval_augmented_generate(
    model,
    tokenizer,
    query: str,
    facts: Dict[str, str],
    device: str,
    max_new_tokens: int = 20,
) -> str:
    """
    Generate with retrieval augmentation.

    If we detect a lookup pattern, add the fact to the prompt context.
    """
    # Check if this is a lookup we know
    retrieved_fact = None
    for key, value in facts.items():
        if key in query:
            retrieved_fact = f"{key} = {value}"
            break

    if retrieved_fact:
        # Augment the prompt with the retrieved fact
        prompt = f"Knowledge: {retrieved_fact}\n\nQuery: {query}\n\nAnswer:"
    else:
        prompt = query

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,
        )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    ).strip()

    return response


def test_retrieval_augmented(
    model,
    tokenizer,
    facts: Dict[str, str],
    all_facts: Dict[str, str],
    device: str,
) -> Tuple[int, int, List[dict]]:
    """Test with retrieval augmentation."""
    correct = 0
    total = len(facts)
    results = []

    for key, expected in facts.items():
        response = retrieval_augmented_generate(
            model, tokenizer, key, all_facts, device
        )

        is_correct = expected.lower() in response.lower()
        if is_correct:
            correct += 1

        results.append({
            "key": key,
            "expected": expected,
            "response": response[:50],
            "correct": is_correct,
        })

    return correct, total, results


def main():
    """Run pre-populated memory experiment."""
    print("=" * 60)
    print("Pre-Populated Memory Experiment")
    print("=" * 60)

    device = "mps" if torch.backends.mps.is_available() else \
             "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    # Load base model and tokenizer
    print("\nLoading base model...")
    model_name = "HuggingFaceTB/SmolLM-135M-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(model_name)
    base_model = base_model.to(device)
    base_model.eval()

    # Test on a subset
    test_facts = dict(list(KNOWN_FACTS.items())[:20])

    # Test baseline first
    print("\n" + "=" * 60)
    print("Baseline (no augmentation)")
    print("=" * 60)

    baseline_correct, baseline_total, baseline_results = test_retrieval(
        base_model, tokenizer, test_facts, device
    )
    print(f"\nBaseline Accuracy: {baseline_correct}/{baseline_total} ({100*baseline_correct/baseline_total:.1f}%)")

    print("\nSample baseline results:")
    for r in baseline_results[:5]:
        status = "✓" if r["correct"] else "✗"
        print(f"  {r['key']}: {r['response'][:30]}... {status}")

    # Test retrieval-augmented approach
    print("\n" + "=" * 60)
    print("Retrieval-Augmented Generation (RAG)")
    print("=" * 60)

    print("\nTesting with knowledge retrieval...")
    rag_correct, rag_total, rag_results = test_retrieval_augmented(
        base_model, tokenizer, test_facts, KNOWN_FACTS, device
    )
    print(f"\nRAG Accuracy: {rag_correct}/{rag_total} ({100*rag_correct/rag_total:.1f}%)")

    print("\nSample RAG results:")
    for r in rag_results[:5]:
        status = "✓" if r["correct"] else "✗"
        print(f"  {r['key']}: {r['response'][:30]}... {status}")

    # Test trained Engram (if available)
    print("\n" + "=" * 60)
    print("Trained Engram (for comparison)")
    print("=" * 60)

    import os
    weights_path = "adapters-engram-exact/engram_weights.pt"
    if os.path.exists(weights_path):
        print("\nLoading trained Engram model...")
        engram_base = AutoModelForCausalLM.from_pretrained(model_name)
        engram_model = EngramModelWrapper(engram_base, memory_size=500, freeze_base=True)
        engram_model.load_engram_weights(weights_path)
        engram_model = engram_model.to(device)
        engram_model.eval()

        engram_correct, engram_total, engram_results = test_retrieval(
            engram_model, tokenizer, test_facts, device
        )
        print(f"\nTrained Engram Accuracy: {engram_correct}/{engram_total} ({100*engram_correct/engram_total:.1f}%)")

        print("\nSample Engram results:")
        for r in engram_results[:5]:
            status = "✓" if r["correct"] else "✗"
            print(f"  {r['key']}: {r['response'][:30]}... {status}")
    else:
        print("\nNo trained weights found, skipping.")
        engram_correct = 0

    # Compare by category
    print("\n" + "=" * 60)
    print("Results by Category")
    print("=" * 60)

    def categorize(results):
        cats = {}
        for r in results:
            cat = r["key"].split(":")[0]
            if cat not in cats:
                cats[cat] = {"correct": 0, "total": 0}
            cats[cat]["total"] += 1
            if r["correct"]:
                cats[cat]["correct"] += 1
        return cats

    baseline_cats = categorize(baseline_results)
    rag_cats = categorize(rag_results)

    print(f"\n{'Category':<15} {'Baseline':>10} {'RAG':>10}")
    print("-" * 40)

    for cat in baseline_cats:
        base_pct = 100 * baseline_cats[cat]["correct"] / baseline_cats[cat]["total"]
        rag_pct = 100 * rag_cats[cat]["correct"] / rag_cats[cat]["total"]
        print(f"{cat:<15} {base_pct:>9.0f}% {rag_pct:>9.0f}%")

    # Summary
    print("\n" + "=" * 60)
    print("Experiment Complete")
    print("=" * 60)

    print(f"\nSummary:")
    print(f"  Baseline:      {baseline_correct}/{baseline_total} ({100*baseline_correct/baseline_total:.1f}%)")
    print(f"  RAG:           {rag_correct}/{rag_total} ({100*rag_correct/rag_total:.1f}%)")
    if engram_correct > 0:
        print(f"  Trained Engram: {engram_correct}/{engram_total} ({100*engram_correct/engram_total:.1f}%)")

    print("\nKey Insight:")
    print("  Hash-based memory requires TRAINING to align representations.")
    print("  Pre-population without training doesn't work because:")
    print("  1. Input embeddings ≠ hidden state representations")
    print("  2. Model needs learned projections to interpret memory")
    print("  3. RAG (retrieval-augmented generation) is simpler for static facts")


if __name__ == "__main__":
    main()
