# Engram PoC

A proof-of-concept demonstrating the **Engram** concept from DeepSeek's paper ["Conditional Memory via Scalable Lookup"](https://arxiv.org/abs/2601.07372) using LoRA fine-tuning on small language models.

## Overview

Engram introduces **conditional memory as a complementary sparsity axis** for transformers, enabling O(1) lookup operations instead of recomputing common patterns through attention. This PoC approximates Engram benefits through behavioral fine-tuning:

1. **Pattern Injection**: Training data encodes "lookup-like" patterns (code idioms, facts, formatting)
2. **LoRA Adapters**: Learn to recognize and consistently respond to patterns
3. **Evaluation**: Compare consistency and accuracy between base model vs Engram-tuned model

## Goals

- Demonstrate measurable improvement attributable to Engram-style training
- Create a YouTube-friendly demo comparing baseline vs Engram-tuned model
- Educational content explaining the Engram concept and why it matters

### Target Metrics
| Metric | Target |
|--------|--------|
| Pattern Consistency | +20% improvement |
| Pattern Accuracy | +10% improvement |
| Training Time | <5 minutes on M-series Mac |

## Quick Start

### Prerequisites

- Apple Silicon Mac (M1/M2/M3/M4)
- Python 3.10+
- [uv](https://github.com/astral-sh/uv) package manager

### Setup

```bash
# Clone the repository
git clone https://github.com/softwarewrighter/engram-poc.git
cd engram-poc

# Create virtual environment with uv
uv venv

# Activate the virtual environment
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt

# Or install with dev dependencies
uv pip install -e ".[dev]"
```

### Verify Installation

```bash
# Test that MLX-LM is working
source .venv/bin/activate
mlx_lm.generate --model HuggingFaceTB/SmolLM-135M-Instruct \
    --prompt "Hello, how are you?" --max-tokens 30
```

Expected output:
```
I'm doing great, thanks for the help. How about you?
==========
Prompt: 15 tokens, 35.977 tokens-per-sec
Generation: 15 tokens, 105.010 tokens-per-sec
Peak memory: 0.331 GB
```

## Usage

### Generate Training Data
```bash
source .venv/bin/activate
python -m src.data_gen.generate
```

### Train with LoRA
```bash
source .venv/bin/activate
./scripts/train.sh
```

### Evaluate
```bash
source .venv/bin/activate
./scripts/eval.sh
```

### Run Demo
```bash
source .venv/bin/activate
./scripts/demo.sh
```

## Project Structure

```
engram-poc/
├── data/
│   ├── patterns/         # Pattern definition YAML files
│   ├── train.jsonl       # Generated training data
│   └── valid.jsonl       # Generated validation data
├── src/
│   ├── data_gen/         # Training data generation
│   ├── eval/             # Evaluation framework
│   ├── demo/             # Demo scripts
│   └── config/           # Model configurations
├── adapters/             # Trained LoRA weights
├── results/              # Evaluation results
├── scripts/              # Shell scripts for training/eval/demo
└── docs/                 # Documentation
```

## Documentation

### Project Documentation
- [Architecture](docs/architecture.md) - System architecture and Engram concepts
- [PRD](docs/prd.md) - Product requirements document
- [Design](docs/design.md) - Technical design with code snippets
- [Plan](docs/plan.md) - Implementation plan and task breakdown
- [Status](docs/status.md) - Project status tracker

### Process Documentation
- [Process](docs/process.md) - Development workflow and processes
- [AI Agent Instructions](docs/ai_agent_instructions.md) - Instructions for AI coding agents
- [Tools](docs/tools.md) - Development tools and setup

### External References
- [Engram Paper (arXiv)](https://arxiv.org/abs/2601.07372)
- [DeepSeek Engram GitHub](https://github.com/deepseek-ai/Engram)
- [MLX-LM Documentation](https://github.com/ml-explore/mlx-examples/tree/main/llms/mlx_lm)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)

## Platform Support

### Phase 1: MLX / Apple Silicon (Current)
- Framework: MLX-LM
- Model: SmolLM-135M-Instruct
- Training: ~6 seconds for 100 iterations

### Phase 2: Unsloth / NVIDIA GPU (Planned)
- Framework: Unsloth + Transformers
- Larger model support
- Production deployment options

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

See [docs/process.md](docs/process.md) for development workflow and contribution guidelines.
