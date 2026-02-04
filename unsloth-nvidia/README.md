# Engram PoC - Unsloth / NVIDIA GPU Edition

A standalone implementation of the Engram PoC for **NVIDIA GPUs** using [Unsloth](https://github.com/unslothai/unsloth) for fast LoRA fine-tuning on CUDA.

> **Platform Note**: This directory is for **Linux/Windows with NVIDIA GPUs**. For Apple Silicon, use the [main project](../README.md) with MLX.

## What This Uses

- **Unsloth** - Optimized LoRA training (2-5x faster than HuggingFace)
- **PEFT/LoRA** - Parameter-efficient fine-tuning
- **PyTorch + CUDA** - NVIDIA GPU acceleration
- **Transformers** - HuggingFace model loading

## Overview

Engram introduces **conditional memory** for transformers - O(1) lookup operations instead of recomputing common patterns. This PoC demonstrates the concept through LoRA fine-tuning:

1. **Pattern Training**: Encode common patterns (code idioms, facts, formats)
2. **Fast Fine-tuning**: Train with Unsloth on NVIDIA GPUs
3. **Before/After Demo**: Compare baseline vs Engram-tuned responses

## Results

**Tested on:** NVIDIA GeForce RTX 5060 (16GB VRAM), CUDA 13.0, PyTorch 2.6.0+cu124

| Metric | Baseline | Engram-tuned | Notes |
|--------|----------|--------------|-------|
| Accuracy | 8.59% | 14.06% | **+63.6% relative improvement** |
| Avg Latency | 1335ms | 1449ms | Per-inference |
| Training Time | - | ~90s | 10 epochs, 243 examples |
| Test Examples | 128 | 128 | - |

### Demo Output
```
Prompt: Complete: for i in range(

Baseline:     len(data)):\n    if data[i] == '1':\n        return True...
Engram-tuned: len(data)):\n    if data[i] == '1':\n        return True...
```

> **Note**: Single-epoch training shows similar outputs between baseline and tuned models. Accuracy variance is expected with minimal training. The key demonstration is the fast LoRA fine-tuning pipeline working end-to-end on NVIDIA GPUs.

## Prerequisites

- NVIDIA GPU with CUDA support (RTX 20xx or newer)
- CUDA 11.8+ or 12.1+
- Python 3.10+
- [uv](https://github.com/astral-sh/uv) package manager
- 4GB+ VRAM (8GB+ recommended)

## Quick Start (Arch Linux / Any Linux with NVIDIA)

### 1. Clone and Navigate

```bash
git clone https://github.com/softwarewrighter/engram-poc.git
cd engram-poc/unsloth-nvidia
```

### 2. Setup Environment

```bash
# Create virtual environment with uv
uv venv

# Activate
source .venv/bin/activate

# Check your CUDA version
nvidia-smi

# Install PyTorch with CUDA (choose your CUDA version)
# CUDA 11.8:
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# CUDA 12.1+:
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install Unsloth (from git for latest)
uv pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Install remaining dependencies
uv pip install -r requirements.txt
```

### 2. Verify GPU

```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"
```

### 3. Run Full Pipeline

```bash
# Generate data, train, evaluate, and demo
./scripts/run_all.sh
```

Or run steps individually:

```bash
# 1. Generate training data
python -m src.data_gen

# 2. Train with Unsloth (~5 seconds)
./scripts/train.sh

# 3. Evaluate
./scripts/eval.sh

# 4. Run demo
./scripts/demo.sh
```

## Usage

### Training Options

```bash
# Basic training
./scripts/train.sh

# Custom options
python -m src.train \
    --model "HuggingFaceTB/SmolLM-135M-Instruct" \
    --epochs 1 \
    --batch-size 4 \
    --lora-rank 8

# With 4-bit quantization (for larger models)
python -m src.train \
    --model "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit" \
    --load-in-4bit
```

### Demo Options

```bash
# Interactive demo
./scripts/demo.sh

# Quick demo (non-interactive)
./scripts/demo.sh --quick

# Python demo
python -m src.demo --quick
```

## Project Structure

```
unsloth-nvidia/
├── README.md              # This file
├── requirements.txt       # Python dependencies
├── configs/
│   └── train_config.yaml  # Training configuration
├── data/
│   └── patterns/          # Symlink to ../data/patterns
├── scripts/
│   ├── run_all.sh         # Full pipeline
│   ├── train.sh           # Training script
│   ├── eval.sh            # Evaluation script
│   └── demo.sh            # Demo script
└── src/
    ├── data_gen.py        # Data generation
    ├── train.py           # Unsloth training
    ├── eval.py            # Evaluation
    └── demo.py            # Interactive demo
```

## Supported Models

| Model | VRAM Required | Notes |
|-------|---------------|-------|
| SmolLM-135M-Instruct | ~1GB | Default, fast iteration |
| Llama-3.1-8B-Instruct (4-bit) | ~6GB | Better quality |
| Mistral-7B-Instruct (4-bit) | ~5GB | Good balance |
| Qwen2-7B-Instruct (4-bit) | ~5GB | Multilingual |

## Troubleshooting

### CUDA Out of Memory
```bash
# Reduce batch size
python -m src.train --batch-size 1

# Enable gradient checkpointing
python -m src.train --gradient-checkpointing

# Use 4-bit quantization
python -m src.train --load-in-4bit
```

### Unsloth Import Error
```bash
# Reinstall in correct order
uv pip uninstall torch unsloth
uv pip install torch --index-url https://download.pytorch.org/whl/cu121
uv pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

### bitsandbytes Issues
```bash
uv pip uninstall bitsandbytes
uv pip install bitsandbytes --no-cache-dir
```

## Google Colab

For quick testing without local GPU:

```python
# In Colab notebook
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!git clone https://github.com/softwarewrighter/engram-poc.git
%cd engram-poc/unsloth-nvidia
!pip install -r requirements.txt
!./scripts/run_all.sh
```

## Comparison with MLX Version

| Feature | MLX (Apple Silicon) | Unsloth (NVIDIA) |
|---------|---------------------|------------------|
| Hardware | M1/M2/M3/M4 | RTX/Tesla/A100 |
| Memory | Unified | Dedicated VRAM |
| Training Speed | ~10 seconds | ~5 seconds |
| 4-bit Quant | Limited | Full support |
| Large Models | Memory limited | Supported |

## License

MIT License - see [LICENSE](../LICENSE) for details.

## References

- [Engram Paper (arXiv)](https://arxiv.org/abs/2601.07372)
- [Unsloth GitHub](https://github.com/unslothai/unsloth)
- [Main Engram PoC](../README.md)
