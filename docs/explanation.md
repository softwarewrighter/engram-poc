# Understanding Engram: ELI5 Guide

## What This Repo Demonstrates

### The Core Idea (Really Simple)

**Problem**: LLMs waste computation re-deriving things they "know" every single time.

When you type `for i in range(`, the model has to "think through" what comes next using expensive attention computations - even though `len(items)):` is an incredibly common pattern it's seen millions of times.

**Engram's Solution**: Add a "cheat sheet" lookup table. Instead of computing everything, just look up common patterns in O(1) time.

Think of it like:
- **Without Engram**: Solving 2+2 by counting on fingers every time
- **With Engram**: Just remembering "2+2=4"

---

## What This PoC Actually Does

**Important caveat**: This repo does NOT implement real Engram (that requires modifying transformer architecture). It **approximates** Engram benefits using LoRA fine-tuning.

```
Real Engram:     Adds lookup tables INTO the model architecture
This PoC:        Trains the model to BEHAVE like it has lookup tables
```

We train on ~300 pattern->response pairs:
- `for i in range(` -> `len(items)):`
- `Q: HTTP status for Not Found?` -> `404`
- `if x = 5:` (error) -> `if x == 5:`

---

## Results Summary

| Platform | Baseline | Tuned | Improvement |
|----------|----------|-------|-------------|
| MLX (Mac) | 8.65% | 11.54% | +33% relative |
| CUDA (RTX 5060) | 8.59% | 14.06% | +64% relative |

**Translation**: The tuned model correctly completes trained patterns ~1.5x more often than baseline.

**What "correct" means**: Exact match on expected output (strict scoring).

---

## How The "Engram Layer" Was Built (In This PoC)

It wasn't really built - that's the key point. We **approximated** it:

1. **Created pattern data** - 131 patterns across 4 categories (code idioms, facts, formats, error fixes)
2. **Augmented to ~300 training examples** - Variations of each pattern
3. **LoRA fine-tuned** - Trained lightweight adapters (~1% of model params)
4. **Evaluated** - Compared baseline vs tuned on held-out test patterns

The LoRA adapters learn to recognize patterns and respond consistently - mimicking what a true Engram lookup would do.

---

## Pros & Cons

### Good Use Cases

| Use Case | Why It Helps |
|----------|--------------|
| **Code autocomplete** | Common idioms are highly predictable |
| **Factual Q&A** | Static facts benefit from lookup |
| **Format enforcement** | date->ISO, camelCase->snake_case |
| **Boilerplate generation** | `if __name__ == "__main__":` |
| **Error correction** | Common typos/mistakes |

### Bad Use Cases

| Use Case | Why It Doesn't Help |
|----------|---------------------|
| **Novel reasoning** | Can't look up what you've never seen |
| **Creative writing** | Patterns = predictable = boring |
| **Complex multi-step logic** | Lookup doesn't help with reasoning chains |
| **Rare/domain-specific patterns** | Only helps if pattern is in the table |
| **Anything requiring context** | Engram is local (N-gram), not contextual |

---

## It's Not a Panacea

**What Engram (real or approximated) does NOT solve:**

1. **Hallucination** - Looking up wrong pattern still produces wrong output
2. **Reasoning** - O(1) lookup can't replace O(n²) reasoning
3. **Generalization** - Only helps on patterns it's seen
4. **Context understanding** - It's fundamentally local, not global
5. **Novel problems** - No lookup table for things that don't exist yet

**The paper's own framing**: Engram is a "complementary sparsity axis" - it works *alongside* attention and MoE, not replacing them. It handles the easy stuff so attention can focus on hard stuff.

---

## The Honest Summary of This PoC

This PoC shows that:
- You CAN train a model to be more consistent on known patterns
- Accuracy improves ~30-60% on pattern-matching tasks
- This is NOT real Engram architecture
- Only helps on trained patterns, not general intelligence
- 14% accuracy is still pretty bad in absolute terms (small model + strict matching)

**Real Engram** (from DeepSeek) shows +3-5 points on benchmarks like MMLU and HumanEval at 27B scale. This PoC is a tiny educational demo, not a production system.

---

# How To Implement Real Engram

The PoC above is a behavioral approximation. Here's what a **proper** Engram implementation would require:

## Architecture Overview (From the Paper)

Real Engram adds three components to the transformer:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Transformer + Real Engram                        │
│                                                                     │
│  Input Tokens: [t1, t2, t3, t4, t5, ...]                           │
│       │                                                             │
│       ▼                                                             │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  1. N-gram Hash Function                                     │   │
│  │     - Slide window over input tokens                         │   │
│  │     - Hash each N-gram to bucket ID                          │   │
│  │     - Example: hash("for i in") -> bucket_42                 │   │
│  └─────────────────────────────────────────────────────────────┘   │
│       │                                                             │
│       ▼                                                             │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  2. Embedding Lookup Table                                   │   │
│  │     - Large table: [num_buckets x embedding_dim]             │   │
│  │     - O(1) lookup per N-gram                                 │   │
│  │     - Returns learned embedding for that pattern             │   │
│  └─────────────────────────────────────────────────────────────┘   │
│       │                                                             │
│       ▼                                                             │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  3. Injection Points                                         │   │
│  │     - Add Engram embeddings to residual stream               │   │
│  │     - Can inject at: embedding layer, attention, FFN         │   │
│  │     - Gating mechanism learns when to use lookup vs compute  │   │
│  └─────────────────────────────────────────────────────────────┘   │
│       │                                                             │
│       ▼                                                             │
│  [Normal Transformer Layers with Engram-augmented representations] │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Key Components to Implement

### 1. N-gram Extraction & Hashing

```python
class NgramHasher:
    """Extract and hash N-grams from token sequences."""

    def __init__(self, n: int = 3, num_buckets: int = 1_000_000):
        self.n = n
        self.num_buckets = num_buckets

    def extract_ngrams(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Slide window over tokens to extract N-grams."""
        # token_ids: [batch, seq_len]
        # Returns: [batch, seq_len - n + 1, n]
        ngrams = token_ids.unfold(dimension=1, size=self.n, step=1)
        return ngrams

    def hash_ngrams(self, ngrams: torch.Tensor) -> torch.Tensor:
        """Hash N-grams to bucket IDs."""
        # Simple polynomial hash (production would use better hash)
        # ngrams: [batch, num_ngrams, n]
        batch, num_ngrams, n = ngrams.shape

        # Polynomial rolling hash
        base = 31
        powers = base ** torch.arange(n, device=ngrams.device)
        hashes = (ngrams * powers).sum(dim=-1) % self.num_buckets

        return hashes  # [batch, num_ngrams]
```

### 2. Engram Embedding Table

```python
class EngramEmbedding(nn.Module):
    """Learnable embedding table for N-gram patterns."""

    def __init__(
        self,
        num_buckets: int = 1_000_000,
        embedding_dim: int = 768,
        n: int = 3,
    ):
        super().__init__()
        self.hasher = NgramHasher(n=n, num_buckets=num_buckets)

        # The actual lookup table - this is what gets trained
        self.embeddings = nn.Embedding(num_buckets, embedding_dim)

        # Initialize small to not disrupt pretrained model
        nn.init.normal_(self.embeddings.weight, std=0.01)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Look up Engram embeddings for input tokens."""
        # token_ids: [batch, seq_len]

        # Extract and hash N-grams
        ngrams = self.hasher.extract_ngrams(token_ids)
        bucket_ids = self.hasher.hash_ngrams(ngrams)

        # Lookup embeddings - O(1) per N-gram
        engram_embeds = self.embeddings(bucket_ids)

        # Pad to match original sequence length
        # (N-gram extraction reduces length by n-1)
        padding = torch.zeros(
            token_ids.shape[0],
            self.hasher.n - 1,
            engram_embeds.shape[-1],
            device=engram_embeds.device
        )
        engram_embeds = torch.cat([padding, engram_embeds], dim=1)

        return engram_embeds  # [batch, seq_len, embedding_dim]
```

### 3. Gated Injection into Transformer

```python
class EngramAugmentedLayer(nn.Module):
    """Transformer layer with Engram injection."""

    def __init__(self, original_layer, engram_embedding, hidden_dim):
        super().__init__()
        self.original_layer = original_layer
        self.engram = engram_embedding

        # Learnable gate: decides how much to use Engram vs compute
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, hidden_states, token_ids, **kwargs):
        # Get Engram lookup
        engram_embeds = self.engram(token_ids)

        # Compute gate value (per-token decision)
        gate_input = torch.cat([hidden_states, engram_embeds], dim=-1)
        gate_value = self.gate(gate_input)  # [batch, seq, 1]

        # Inject Engram embeddings into residual stream
        hidden_states = hidden_states + gate_value * engram_embeds

        # Continue with normal transformer computation
        output = self.original_layer(hidden_states, **kwargs)

        return output
```

### 4. Full Model Integration

```python
class EngramModel(nn.Module):
    """Wrapper that adds Engram to existing transformer."""

    def __init__(
        self,
        base_model,
        num_buckets: int = 1_000_000,
        ngram_sizes: list = [2, 3, 4],  # Multiple N-gram sizes
        inject_layers: list = [0, 4, 8, 12],  # Which layers to augment
    ):
        super().__init__()
        self.base_model = base_model
        hidden_dim = base_model.config.hidden_size

        # Create Engram embeddings for different N-gram sizes
        self.engrams = nn.ModuleList([
            EngramEmbedding(num_buckets, hidden_dim, n=n)
            for n in ngram_sizes
        ])

        # Wrap specified layers with Engram injection
        for layer_idx in inject_layers:
            original_layer = base_model.layers[layer_idx]
            base_model.layers[layer_idx] = EngramAugmentedLayer(
                original_layer,
                self.engrams,
                hidden_dim
            )

    def forward(self, input_ids, **kwargs):
        # Store token_ids for Engram lookup
        # Then run normal forward pass
        return self.base_model(input_ids, token_ids=input_ids, **kwargs)
```

## Training Strategy

### What to Train

| Component | Trainable? | Notes |
|-----------|------------|-------|
| Base model weights | Frozen | Keep pretrained knowledge |
| Engram embedding table | **Yes** | Learn pattern->embedding mapping |
| Gate networks | **Yes** | Learn when to use lookup |
| Injection projections | **Yes** | Learn how to combine |

### Training Data

Real Engram is trained on **normal pretraining data** - the model learns which N-grams are predictive through standard language modeling loss. No special pattern curation needed.

```python
# Standard causal LM loss, but model has Engram components
loss = cross_entropy(model(input_ids), target_ids)
loss.backward()  # Gradients flow to Engram embeddings
```

### Key Insight

The Engram table learns **which patterns are worth memorizing** automatically. High-frequency, predictive N-grams get useful embeddings. Rare N-grams stay near zero (due to small initialization).

## Implementation Challenges

### 1. Memory

```
1M buckets x 768 dims x 4 bytes = 3 GB just for Engram table
```

**Solutions**:
- Sparse embeddings (only materialize active buckets)
- Quantization (int8 embeddings)
- Hierarchical hashing (common patterns get dedicated buckets)

### 2. Hash Collisions

Different N-grams may hash to same bucket.

**Solutions**:
- More buckets (memory tradeoff)
- Multiple hash functions + averaging
- Learned hash function (but loses O(1) guarantee)

### 3. Integration with Existing Frameworks

Most frameworks don't support easy layer injection.

**Options**:
- Fork transformers library and modify
- Use hooks (slower, hacky)
- Train from scratch (expensive)

### 4. Distributed Training

Engram table is huge and accessed randomly - bad for sharding.

**Solutions**:
- Replicate Engram table on all ranks
- Use embedding parallelism techniques from recommender systems

## Comparison: PoC vs Real Implementation

| Aspect | This PoC | Real Engram |
|--------|----------|-------------|
| Architecture change | None (just LoRA) | Adds new components |
| What's learned | Response patterns | Which N-grams matter |
| Training data | Curated patterns | Any text corpus |
| Generalization | Only trained patterns | All frequent N-grams |
| Inference | Same as base model | O(1) lookup + compute |
| Memory overhead | ~5MB (LoRA) | ~3GB (Engram table) |
| Implementation effort | Hours | Months |

## When to Use Each Approach

### Use PoC Approach (LoRA fine-tuning) When:
- You have specific patterns you want to enforce
- You can't modify model architecture
- Quick experimentation
- Limited compute budget

### Use Real Engram When:
- Building a new model from scratch
- You have pretraining-scale compute
- Efficiency at inference is critical
- You want automatic pattern discovery

## References

- [Engram Paper (arXiv:2601.07372)](https://arxiv.org/abs/2601.07372)
- [DeepSeek Engram GitHub](https://github.com/deepseek-ai/Engram)
- [N-gram Language Models (Classic NLP)](https://web.stanford.edu/~jurafsky/slp3/3.pdf)
- [Product Quantization for Efficient Embeddings](https://arxiv.org/abs/1906.00532)
