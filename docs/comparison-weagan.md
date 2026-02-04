# Comparison: engram-poc vs weagan/Engram

Two different approaches to demonstrating Engram concepts.

## Key Distinction: Implements vs Emulates

| Repo | Relationship to Engram |
|------|------------------------|
| **weagan/Engram** | **Implements** - Actually adds hash-based memory modules to transformer architecture |
| **engram-poc** | **Emulates** - Trains model to *behave* like it has memory through LoRA fine-tuning |

**weagan/Engram is real Engram.** It adds the `EnhancedEngramModule` with O(1) hash lookups directly into the forward pass.

**engram-poc is behavioral approximation.** The model architecture is unchanged; we just train it to respond consistently to patterns.

## Quick Summary

| Aspect | softwarewrighter/engram-poc | weagan/Engram |
|--------|----------------------------|---------------|
| **Approach** | LoRA fine-tuning (behavioral) | Actual architecture change |
| **Modifies architecture?** | No | Yes (adds memory modules) |
| **Task type** | Pattern completion | Long-term fact recall |
| **Models compared** | Baseline vs LoRA-tuned | Baseline vs Engram vs Hybrid |
| **What it proves** | Training helps pattern consistency | Hash-based memory enables long-term recall |
| **Practical use** | Deploy on existing models | Requires custom model |
| **Format** | Production repo | Single Colab notebook |

---

## Detailed Comparison

### 1. Architectural Approach

**engram-poc (This Repo)**
```
Base Model (frozen) + LoRA Adapters
         ↓
Model learns to BEHAVE like it has memory
         ↓
No architectural changes
```

- Uses standard transformers (SmolLM-135M)
- Adds LoRA adapters (~1% of parameters)
- Model learns pattern→response mappings through training
- **Approximates** Engram benefits behaviorally

**weagan/Engram**
```
Transformer + EnhancedEngramModule
         ↓
Actual hash-based memory table added
         ↓
O(1) lookup integrated into forward pass
```

- Builds custom transformer from scratch
- Adds `EnhancedEngramModule` with:
  - 50,000-slot memory table
  - Multi-head deterministic hashing
  - Gated memory injection
- **Implements** true Engram architecture

### 2. The EnhancedEngramModule (weagan)

```python
class EnhancedEngramModule(nn.Module):
    def __init__(self, table_size=100000, d_model=512, n_heads=4):
        # Large learnable memory table
        self.memory_table = nn.Parameter(torch.zeros(table_size, d_model))

        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )

    def multi_head_hash(self, input_ids):
        """Deterministic O(1) hash lookup"""
        hashes = []
        for i in range(self.n_heads):
            prime = [17, 31, 53, 79, 107, 131, 157, 181][i % 8]
            hash_val = (input_ids * prime) % self.table_size
            hashes.append(hash_val)
        return torch.stack(hashes, dim=-1)

    def forward(self, hidden_states, input_ids):
        # Hash tokens to memory indices
        indices = self.multi_head_hash(input_ids)

        # O(1) lookup from memory table
        retrieved_mem = F.embedding(indices, self.memory_table)

        # Gated injection into hidden states
        gate_score = self.gate(torch.cat([hidden_states, retrieved_mem], dim=-1))
        return hidden_states + gate_score * retrieved_mem
```

This is **real Engram** - not an approximation.

### The Gating Mechanism (Critical Component)

The **gate** is what makes Engram work intelligently. It's a learned network that decides:
> "For this token, should I trust the retrieved memory or the computed hidden state?"

**weagan's gate implementation:**
```python
# Gate network: takes [hidden_state, retrieved_memory] -> scalar 0-1
self.gate = nn.Sequential(
    nn.Linear(d_model * 2, d_model),  # Combine both signals
    nn.ReLU(),
    nn.Linear(d_model, 1),            # Output single value
    nn.Sigmoid()                       # Squash to 0-1
)

# In forward pass:
gate_input = torch.cat([hidden_states, retrieved_mem], dim=-1)
gate_score = self.gate(gate_input)  # 0 = ignore memory, 1 = trust memory

# Gated injection
output = hidden_states + gate_score * retrieved_mem
```

**What the gate learns:**
- Gate → 0: "This token doesn't match anything useful in memory, ignore lookup"
- Gate → 1: "This token triggered a relevant memory, use it!"

**Why this matters:**
- Hash collisions happen (different inputs → same bucket)
- Not every token needs memory lookup
- Gate learns to filter noise and only use relevant retrievals

**engram-poc has no gate** because we don't add memory modules. The "gating" equivalent in our approach is implicit - the LoRA weights learn when to produce pattern-consistent outputs vs default behavior.

| Aspect | weagan (explicit gate) | engram-poc (implicit) |
|--------|------------------------|----------------------|
| Mechanism | Learned gate network | LoRA weight updates |
| Controllable? | Yes (inspect gate values) | No (black box) |
| Per-token decision | Yes | No |
| Memory usage | Explicit retrieval | Baked into weights |

### 3. Task Design

**engram-poc: Pattern Completion**
```
Input:  "Complete: for i in range("
Output: "len(items)):"

Input:  "Q: HTTP status for Not Found?\nA:"
Output: "404"
```

- Tests if model learns specific pattern→response mappings
- Patterns are code idioms, facts, formats, error fixes
- Evaluation: exact match accuracy

**weagan/Engram: Long-Term Fact Recall**
```
Phase 1 - Present facts:
  [trigger_42] -> [fact_word_1, fact_word_2, fact_word_3]
  [trigger_17] -> [fact_word_4, fact_word_5, fact_word_6]

Phase 2 - Long distraction:
  [300 random tokens that push facts outside attention window]

Phase 3 - Test recall:
  [trigger_42] -> model must output [fact_word_1]
```

- Tests if model can recall facts presented 300+ tokens ago
- Baseline (128-token context) literally cannot see the facts anymore
- Engram's hash lookup retrieves them regardless of distance

### 4. Models Compared

**engram-poc**
| Model | Description |
|-------|-------------|
| Baseline | SmolLM-135M without adapters |
| Engram-tuned | Same model + LoRA adapters |

Both use identical architecture, difference is learned weights.

**weagan/Engram**
| Model | Description |
|-------|-------------|
| Baseline | Transformer with 128-token context limit |
| Engram-Enhanced | Same + EnhancedEngramModule per layer |
| Hybrid | Full attention (no context limit) - upper bound |

Different architectures being compared.

### 5. Results

**engram-poc**
| Platform | Baseline | Tuned | Improvement |
|----------|----------|-------|-------------|
| MLX | 8.65% | 11.54% | +33% relative |
| CUDA | 8.59% | 14.06% | +64% relative |

Modest but real improvement on pattern tasks.

**weagan/Engram**
| Model | Short-term (50 tokens) | Long-term (300 tokens) |
|-------|------------------------|------------------------|
| Baseline | High | **Struggles** |
| Engram | Perfect (1.0) | **Perfect (1.0)** |
| Hybrid | High | High |

Dramatic improvement specifically on long-term recall where baseline fails completely.

### 6. What Each Proves

**engram-poc proves:**
- You CAN train a model to be more consistent on patterns
- LoRA fine-tuning is a practical way to add "soft memory"
- Works with existing models (no architecture changes)
- Useful for pattern enforcement in production

**weagan/Engram proves:**
- Hash-based memory tables work for long-term recall
- O(1) lookup beats limited attention for distant facts
- Engram architecture matches full-attention quality at lower cost
- The theoretical Engram concept is sound

### 7. Strengths and Weaknesses

**engram-poc**

| Strengths | Weaknesses |
|-----------|------------|
| Works with any model | Not "real" Engram |
| Easy to deploy (just add adapter) | Only helps on trained patterns |
| Practical for production | Doesn't solve context window limit |
| Two platform support (MLX/CUDA) | Modest accuracy gains |

**weagan/Engram**

| Strengths | Weaknesses |
|-----------|------------|
| True Engram implementation | Requires custom model |
| Dramatic long-term recall gains | Synthetic task only |
| Proves the concept works | Not integrated with real LLMs |
| Clean, educational code | Single notebook format |

---

## When to Use Each

### Use engram-poc when:
- You want to improve pattern consistency on an **existing model**
- You can't modify model architecture
- You need something **deployable today**
- Your use case is pattern completion, not long-term memory

### Use weagan/Engram approach when:
- You're **building a new model** from scratch
- Long-term memory is critical (conversation history, documents)
- You have the resources to train with custom architecture
- You want to **understand how Engram works** internally

---

## Combined Approach

The ideal production system might combine both:

```
┌─────────────────────────────────────────────────────────┐
│                   Production Engram                      │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  1. Base Model (pretrained LLM)                         │
│           │                                              │
│           ▼                                              │
│  2. EnhancedEngramModule (weagan-style)                 │
│     - Hash-based memory for long-term facts             │
│     - O(1) lookup for conversation history              │
│           │                                              │
│           ▼                                              │
│  3. LoRA Adapters (engram-poc-style)                    │
│     - Pattern completion fine-tuning                    │
│     - Domain-specific behaviors                         │
│           │                                              │
│           ▼                                              │
│  4. Output                                               │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

This would give you:
- Long-term memory from hash tables (weagan)
- Pattern consistency from behavioral training (engram-poc)
- Deployable on existing infrastructure

---

## References

- [engram-poc](https://github.com/softwarewrighter/engram-poc) - This repository
- [weagan/Engram](https://github.com/weagan/Engram) - True Engram implementation
- [DeepSeek Engram Paper](https://arxiv.org/abs/2601.07372) - Original research
