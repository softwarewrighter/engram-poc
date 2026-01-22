# GPU Setup Guide (Unsloth / NVIDIA)

This guide covers setting up the Engram PoC on NVIDIA GPUs using Unsloth for fast LoRA fine-tuning.

## Prerequisites

- NVIDIA GPU with CUDA support (RTX 20xx or newer recommended)
- CUDA 11.8+ or 12.1+ installed
- Python 3.10+
- 8GB+ VRAM (for SmolLM-135M, less for quantized)

## Installation

### 1. Create Virtual Environment

```bash
# Using venv
python -m venv .venv-gpu
source .venv-gpu/bin/activate

# Or using conda
conda create -n engram-gpu python=3.10
conda activate engram-gpu
```

### 2. Install PyTorch with CUDA

Check your CUDA version:
```bash
nvcc --version
# or
nvidia-smi
```

Install PyTorch for your CUDA version:
```bash
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 3. Install Unsloth

```bash
# Latest from GitHub (recommended)
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Or stable release
pip install unsloth
```

### 4. Install Remaining Dependencies

```bash
pip install -r requirements-gpu.txt
```

### 5. Verify Installation

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
python -c "from unsloth import FastLanguageModel; print('Unsloth OK')"
```

## Training

### Quick Start

```bash
# Generate training data (same as MLX)
python -m src.data_gen.generate

# Train with Unsloth
python -m src.train_gpu.train
```

### Training Script Options

```bash
python -m src.train_gpu.train \
    --model "HuggingFaceTB/SmolLM-135M-Instruct" \
    --output-dir "./adapters-gpu" \
    --epochs 1 \
    --batch-size 4 \
    --learning-rate 2e-4 \
    --lora-rank 8
```

### Using Larger Models

Unsloth supports 4-bit quantization for larger models:

```bash
# Llama 3.1 8B (requires ~6GB VRAM with 4-bit)
python -m src.train_gpu.train \
    --model "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit" \
    --load-in-4bit
```

## Evaluation

```bash
# Run GPU evaluation
python -m src.eval_gpu.compare \
    --model "HuggingFaceTB/SmolLM-135M-Instruct" \
    --adapter-path "./adapters-gpu"
```

## Demo

```bash
# GPU demo
python -m src.demo.demo_gpu

# Or use unified demo with auto-detection
python -m src.demo.demo_unified
```

## Troubleshooting

### CUDA Out of Memory

Reduce batch size or enable gradient checkpointing:
```bash
python -m src.train_gpu.train --batch-size 1 --gradient-checkpointing
```

### Unsloth Import Error

Make sure PyTorch is installed with CUDA before Unsloth:
```bash
pip uninstall torch unsloth
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install unsloth
```

### bitsandbytes Issues

On some systems, bitsandbytes needs manual setup:
```bash
pip uninstall bitsandbytes
pip install bitsandbytes --no-cache-dir
```

## Performance Comparison

| Platform | Model | Training Time (100 iters) | VRAM Usage |
|----------|-------|---------------------------|------------|
| Apple M2 (MLX) | SmolLM-135M | ~10 seconds | ~2GB unified |
| RTX 3090 (Unsloth) | SmolLM-135M | ~5 seconds | ~1GB |
| RTX 3090 (Unsloth) | Llama-3.1-8B-4bit | ~30 seconds | ~6GB |

## Colab Notebook

For quick testing without local GPU setup, use Google Colab:

```python
# Install in Colab
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Clone repo
!git clone https://github.com/softwarewrighter/engram-poc.git
%cd engram-poc

# Install dependencies
!pip install -r requirements-gpu.txt

# Run training
!python -m src.train_gpu.train
```
