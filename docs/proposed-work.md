# Proposed Work: Building on Engram PoC

Based on learnings from this PoC and the weagan/Engram implementation, here are concrete next steps to advance Engram capabilities.

## Current State

| Repo | What It Does | Limitation |
|------|--------------|------------|
| **engram-poc** | LoRA fine-tuning to emulate Engram behavior | No real memory module, no gating |
| **weagan/Engram** | True Engram with hash-based memory + gating | Custom small models, not integrated with real LLMs |

## Key Insight: It's Not That Hard

The weagan/Engram implementation demonstrates that building Engram is straightforward:

```
EnhancedEngramModule: ~50 lines
EngramEnhancedTransformer: ~60 lines
Training loop: ~80 lines
Total: ~300 lines of core code
Done in: ~1 week
```

The "months of engineering" applies to **27B scale production systems**, not the core concept.

---

## Proposed Work Options

### Option 1: Quick Win (1-2 days)

**Goal:** Add weagan's EnhancedEngramModule to this repo for side-by-side comparison.

**Tasks:**
1. Extract `EnhancedEngramModule` class from weagan notebook
2. Create `src/engram/module.py` with the module
3. Add synthetic memory task for comparison
4. Document differences between LoRA approach and true Engram

**Deliverables:**
- `src/engram/module.py` - The EnhancedEngramModule
- `src/engram/task.py` - Long-term memory task
- `notebooks/engram_comparison.ipynb` - Side-by-side demo

**Value:** Shows both approaches in one repo, educational.

---

### Option 2: Add Gating to LoRA Approach (3-5 days)

**Goal:** Create a hybrid that adds explicit gating without full architectural change.

**Architecture:**
```python
class GatedEngramAdapter:
    """Gated selection between base and tuned model outputs."""

    def __init__(self, base_model, adapter_path):
        self.base_model = base_model
        self.tuned_model = load_with_adapter(base_model, adapter_path)

        # Learnable gate: decides when to use tuned vs base
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )

    def forward(self, input_ids):
        base_logits = self.base_model(input_ids)
        tuned_logits = self.tuned_model(input_ids)

        # Gate decides: 0 = use base, 1 = use tuned
        gate_score = self.gate(hidden_states)

        return gate_score * tuned_logits + (1 - gate_score) * base_logits
```

**Tasks:**
1. Implement `GatedEngramAdapter` class
2. Train gate network on pattern recognition task
3. Evaluate: Does gating improve over always-tuned?
4. Visualize gate activations (when does it fire?)

**Deliverables:**
- `src/engram/gated_adapter.py`
- Gate visualization tools
- Comparison metrics

**Value:** Adds interpretability to LoRA approach, shows when patterns trigger.

---

### Option 3: Integrate EnhancedEngramModule into SmolLM (1-2 weeks)

**Goal:** True Engram architecture with a real pretrained LLM.

**Architecture:**
```
SmolLM-135M (frozen)
    │
    ├── Embedding Layer
    │       │
    │       ▼
    ├── Transformer Layer 0
    │       │
    │       ▼
    │   [EnhancedEngramModule]  ← Inject here
    │       │
    │       ▼
    ├── Transformer Layer 1
    │       ...
    │       ▼
    └── Output Layer
```

**Tasks:**
1. Create wrapper that injects EngramModule into existing model
2. Freeze base model weights, train only:
   - Memory table (50K-1M slots)
   - Gate networks
   - Projection layers
3. Train on pattern data (same as current PoC)
4. Compare: LoRA-only vs Engram-only vs Both

**Implementation approach:**
```python
class EngramSmolLM(nn.Module):
    def __init__(self, base_model_name, memory_size=100000):
        self.base_model = AutoModelForCausalLM.from_pretrained(base_model_name)

        # Freeze base model
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Add Engram modules (one per layer)
        self.engram_modules = nn.ModuleList([
            EnhancedEngramModule(
                table_size=memory_size,
                d_model=self.base_model.config.hidden_size
            )
            for _ in range(self.base_model.config.num_hidden_layers)
        ])

    def forward(self, input_ids, **kwargs):
        # Custom forward that injects Engram after each layer
        hidden_states = self.base_model.embed_tokens(input_ids)

        for i, layer in enumerate(self.base_model.layers):
            hidden_states = layer(hidden_states, **kwargs)
            hidden_states = self.engram_modules[i](hidden_states, input_ids)

        return self.base_model.lm_head(hidden_states)
```

**Deliverables:**
- `src/engram/enhanced_module.py` - The EnhancedEngramModule
- `src/engram/smollm_engram.py` - Integration with SmolLM
- `scripts/train_engram.sh` - Training script
- Updated results comparing all approaches

**Value:** First integration of true Engram with a real pretrained LLM.

---

### Option 4: Production Package (2-3 weeks)

**Goal:** Create pip-installable `engram-transformers` package.

**Features:**
```python
from engram_transformers import add_engram

# Add Engram to any HuggingFace model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b")
engram_model = add_engram(
    model,
    memory_size=1_000_000,
    inject_layers=[0, 4, 8, 12, 16, 20, 24, 28],
    gate_type="learned"  # or "fixed", "content-based"
)

# Train only Engram components
trainer = EngramTrainer(
    model=engram_model,
    train_dataset=dataset,
    freeze_base=True
)
trainer.train()

# Save/load Engram weights separately
engram_model.save_engram("./engram_weights")
engram_model.load_engram("./engram_weights")
```

**Tasks:**
1. Create `engram-transformers` package structure
2. Implement model-agnostic Engram injection
3. Support multiple model architectures (Llama, Mistral, Qwen, etc.)
4. Add training utilities
5. Write documentation and examples
6. Publish to PyPI

**Deliverables:**
- `engram-transformers/` package
- PyPI publication
- Documentation site
- Example notebooks for different models

**Value:** Makes Engram accessible to anyone using HuggingFace models.

---

## Domain Applications

Beyond code patterns, Engram could demonstrate value in:

| Domain | Task | Why Engram Helps |
|--------|------|------------------|
| **Customer Support** | FAQ → Answer | Consistent responses to common questions |
| **Legal** | Regulation ID → Citation | Exact legal text lookup |
| **Medical** | Drug → Interactions | Critical safety information |
| **DevOps** | Service → Port/Config | Consistent infrastructure knowledge |
| **Education** | Term → Definition | Flashcard-style learning |
| **Translation** | Term → Consistent Translation | Terminology consistency |
| **Conversation** | User preference → Recall | Long-term personalization |

**Suggested next domain to implement:** Customer Support FAQ
- Clear trigger→response patterns
- Easy to evaluate (exact match)
- Practical business value
- Dataset available (public FAQs)

---

## Difficulty Assessment (Revised)

| Task | Estimated Effort | Difficulty |
|------|------------------|------------|
| Extract EnhancedEngramModule | 1 day | Easy |
| Add gating to LoRA approach | 3-5 days | Medium |
| Integrate with SmolLM | 1-2 weeks | Medium |
| Production package | 2-3 weeks | Medium-Hard |
| Scale to 7B+ models | 1-2 months | Hard |
| Distributed training | 2-3 months | Hard |

**Key insight:** The core Engram concept is **not hard**. The weagan implementation proves this.

What's hard:
- Integrating with large models (memory management)
- Distributed training of huge memory tables
- Production deployment at scale

What's not hard:
- The EnhancedEngramModule itself (~50 lines)
- Adding it to small models
- Demonstrating the concept works

---

## Recommended Path

**For learning/demonstration:**
```
Option 1 (Quick Win) → Option 3 (SmolLM Integration)
```

**For production use:**
```
Option 3 (SmolLM Integration) → Option 4 (Package)
```

**For research:**
```
Option 2 (Gating) → Option 3 (Integration) → Novel improvements
```

---

## Resources

- [weagan/Engram](https://github.com/weagan/Engram) - Reference implementation
- [DeepSeek Engram Paper](https://arxiv.org/abs/2601.07372) - Original research
- [This repo's comparison doc](comparison-weagan.md) - Detailed analysis
- [This repo's explanation doc](explanation.md) - ELI5 guide

---

## Next Steps

1. **Decide which option** aligns with your goals
2. **Create GitHub issues** for tracking
3. **Start with Option 1** to get weagan's code into this repo
4. **Build incrementally** toward production package

The path from "proof of concept" to "production tool" is clearer than initially thought.
