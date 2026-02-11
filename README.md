# Engram PoC

A proof-of-concept implementing **hash-based O(1) memory lookup** for transformer models, based on DeepSeek's ["Engram: Conditional Memory via Scalable Lookup"](https://arxiv.org/abs/2601.07372) paper.

## Latest Updates

**[Blog Post: DeepSeek Papers Part 3 - Engram Revisited](http://localhost:5907/2026/02/11/deepseek-papers-part3-engram-revisited/)** - Deep dive into what works, what doesn't, and practical guidelines for hash-based memory.

### Key Findings

| Query Type | Engram Accuracy | Baseline | Improvement |
|------------|-----------------|----------|-------------|
| **Acronym Expansion** | 75% | 12% | **+525%** |
| **Element Names** | 67% | 0% | **+∞** |
| Random Synthetic | 0% | 0% | No benefit |

**Insight**: Engram excels at **structured lookups** (ACRONYM:GPU → "Graphics Processing Unit") but doesn't help with arbitrary key-value mappings. This matches the DeepSeek paper's finding: *"Delegating local dependencies to lookups frees up attention capacity for global context."*

## Conditional Engram Routing (New!)

Smart routing that activates Engram memory only when appropriate:

```python
from src.memory import create_conditional_engram

model, tokenizer = create_conditional_engram(
    engram_weights_path="adapters-engram-exact/engram_weights.pt"
)
# Automatically routes lookups → Engram, general queries → bypass
```

**Routing Behavior**:
| Input | Confidence | Route | Output |
|-------|------------|-------|--------|
| `ACRONYM:GPU` | 1.0 | ENGRAM | Graphics Processing Unit |
| `CAPITAL:France` | 1.0 | ENGRAM | Paris |
| `What is the capital of France?` | 0.7 | ENGRAM | Paris |
| `Write a poem about cats` | 0.0 | BYPASS | (base model response) |
| `How do I sort a list?` | 0.0 | BYPASS | (base model response) |

## Architecture

### Hash-Based Memory Module

```
┌─────────────────────────────────────────────────────────────┐
│                    EnhancedEngramModule                      │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │ Multi-Head  │    │   Memory    │    │  Gated      │     │
│  │ Hashing     │ → │   Table     │ → │  Blending   │     │
│  │ (4 heads)   │    │ (500 slots) │    │  (learned)  │     │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
│       ↑                                      ↓              │
│   input_ids                           hidden_states         │
└─────────────────────────────────────────────────────────────┘
```

**Key Components**:
- **Multi-Head Hashing**: 4 parallel hash functions for collision handling
- **Memory Table**: 500-slot embedding table (configurable)
- **Gated Blending**: Learned gate mixing memory with base activations
- **Layer Injection**: Wraps every transformer layer

### Conditional Routing

```
┌─────────────────────────────────────────────────────────────┐
│                  ConditionalEngramWrapper                    │
│  ┌─────────────┐                                            │
│  │  Pattern    │  "ACRONYM:GPU" → 1.0                       │
│  │  Detector   │  "What is..." → 0.7                        │
│  │             │  "Write poem" → 0.0                        │
│  └─────────────┘                                            │
│        ↓                                                    │
│  ┌─────────────┐    ┌─────────────┐                        │
│  │  conf ≥ 0.5 │ → │   ENGRAM    │ → Memory-augmented      │
│  │  conf < 0.5 │ → │   BYPASS    │ → Base model only       │
│  └─────────────┘    └─────────────┘                        │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites
- Apple Silicon Mac (M1/M2/M3/M4) or NVIDIA GPU
- Python 3.10+
- [uv](https://github.com/astral-sh/uv) package manager

### Setup

```bash
git clone https://github.com/softwarewrighter/engram-poc.git
cd engram-poc

# Create virtual environment
uv venv .venv-torch
source .venv-torch/bin/activate

# Install dependencies
uv pip install torch transformers
uv pip install -r requirements.txt
```

### Run Conditional Demo

```bash
python -m src.memory.demo_conditional
```

This runs three demos:
1. **Pattern Detection** - Shows confidence scoring for different query types
2. **Conditional Routing** - Routes queries to Engram or bypasses based on patterns
3. **Three-Way Comparison** - Compares base, Engram, and conditional models

### Train on Exact-Match Data

```bash
# Generate exact-match dataset
python -m src.data_gen.generate_exact_match

# Train Engram (requires GPU or MPS)
python scripts/train_engram.py \
    --train_file data/train_exact.jsonl \
    --valid_file data/valid_exact.jsonl \
    --output_dir adapters-engram-exact \
    --epochs 5
```

## When to Use Engram

### Good Use Cases
- **Terminology expansion**: `ACRONYM:API` → "Application Programming Interface"
- **Factual lookup**: `CAPITAL:France` → "Paris"
- **Code patterns**: `HTTP:404` → "Not Found"
- **Entity facts**: `ELEMENT:Fe` → "Iron"

### Poor Use Cases
- Arbitrary synthetic key-value pairs
- Creative/generative tasks
- Context-dependent answers
- Long-form reasoning

### Success Criteria
| Factor | Good for Engram | Bad for Engram |
|--------|-----------------|----------------|
| Pattern structure | Explicit prefix (CAPITAL:, PORT:) | Freeform text |
| Determinism | Single correct answer | Multiple valid responses |
| Repetition | Pattern seen 100+ times in training | Rare or unique |
| Locality | Answer derivable from key alone | Requires context |

## Project Structure

```
engram-poc/
├── src/memory/                    # Core Engram implementation
│   ├── engram_module.py          # EnhancedEngramModule (hash-based memory)
│   ├── model_wrapper.py          # EngramModelWrapper (layer injection)
│   ├── conditional_engram.py     # ConditionalEngramWrapper (smart routing)
│   └── demo_conditional.py       # Demo script
├── data/
│   ├── exact_5k/                 # 5K exact-match dataset
│   └── patterns/                 # Pattern YAML files
├── adapters-engram-exact/        # Trained Engram weights
├── results/                      # Evaluation results
├── scripts/                      # Training & evaluation scripts
└── docs/                         # Documentation
```

## Documentation

### Core Concepts
- [Proposed Work](docs/proposed-work.md) - Detailed findings, experiments, and guidelines
- [Comparison with weagan/Engram](docs/comparison-weagan.md) - Hash-based vs attention-based approaches
- [ELI5 Explanation](docs/explanation.md) - What this repo does, pros/cons

### Results
- [MLX Results (Apple Silicon)](docs/results-mlx.md) - LoRA fine-tuning results
- [CUDA Results (NVIDIA GPU)](docs/results-cuda.md) - Unsloth/NVIDIA results
- [Exact-Match Evaluation](results/exact_match_eval.json) - Hash-based memory results

### Technical
- [Architecture](docs/architecture.md) - System architecture
- [Design](docs/design.md) - Technical design
- [GPU Setup](docs/gpu_setup.md) - NVIDIA GPU setup

### External References
- [Engram Paper (arXiv:2601.07372)](https://arxiv.org/abs/2601.07372)
- [DeepSeek Engram GitHub](https://github.com/deepseek-ai/Engram)
- [Hash Collision Study (arXiv:2601.16531)](https://arxiv.org/abs/2601.16531) - Why collisions help

## Platform Support

### Apple Silicon (MPS)
```bash
source .venv-torch/bin/activate
python -m src.memory.demo_conditional
```

### NVIDIA GPU (CUDA)
```bash
cd unsloth-nvidia/
source .venv/bin/activate
./scripts/run_all.sh
```

See [unsloth-nvidia/README.md](unsloth-nvidia/README.md) for detailed CUDA setup.

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Video Series

Watch the development journey on YouTube:

**[Full Playlist: Engram PoC Development](https://www.youtube.com/playlist?list=PLKjvVAEaR4isTOri5dlPRIUK8Uy0jotX6)**

### Latest: Part 4 - Hash-Based Memory Deep Dive

[![Part 4 - Hash-Based Memory](https://img.youtube.com/vi/TZT_cWWv9Oc/maxresdefault.jpg)](https://www.youtube.com/watch?v=TZT_cWWv9Oc)

<a href="https://www.youtube.com/watch?v=TZT_cWWv9Oc">
  <img src="https://img.youtube.com/vi/TZT_cWWv9Oc/0.jpg" alt="Watch Part 4" width="480">
</a>

### Earlier Episodes

| Part | Topic | Link |
|------|-------|------|
| Part 1 | MLX on Apple Silicon | [Watch](https://www.youtube.com/shorts/aGoQHs6S1nk) |
| Part 2 | Unsloth on NVIDIA GPU | [Watch](https://www.youtube.com/shorts/uvbfu0WKa3A) |
| Part 3 | Short Explainer | [Watch](https://www.youtube.com/watch?v=UgB1nZqJ3cE) |
| Part 4 | Hash-Based Memory | [Watch](https://www.youtube.com/watch?v=TZT_cWWv9Oc) |

---

## Contributing

See [docs/process.md](docs/process.md) for development workflow and contribution guidelines.
