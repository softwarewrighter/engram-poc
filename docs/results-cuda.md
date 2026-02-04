# CUDA Results (NVIDIA GPU)

Results from training and evaluating the Engram PoC on NVIDIA GPUs using Unsloth.

## Summary Metrics

| Metric | Value |
|--------|-------|
| Platform | NVIDIA GeForce RTX 3060 (12GB VRAM) |
| CUDA Version | 13.0 |
| Framework | Unsloth + PyTorch 2.6.0+cu124 |
| Model | SmolLM-135M-Instruct |
| Training Examples | 243 |
| Training Epochs | 1 |
| Training Time | 26.1 seconds |

### Training Progress

| Metric | Initial | Final |
|--------|---------|-------|
| Loss | 4.01 | 3.19 |
| Loss Reduction | - | **20.4%** |

### Accuracy Comparison

| Metric | Baseline | Engram-tuned | Notes |
|--------|----------|--------------|-------|
| Accuracy | 8.59% | 6.25% | 128 test examples |
| Avg Latency | 1981ms | 1913ms | Per-inference |

## Accuracy Comparison

![CUDA Accuracy Comparison](../images/plots/cuda-accuracy-comparison.png)

**Note**: The single-epoch training shows similar or slightly lower accuracy compared to baseline. This is expected with minimal training - the primary goal of this configuration was to validate the Unsloth pipeline on NVIDIA hardware.

## Demo Output

```
Prompt: Complete: for i in range(

Baseline:     len(data)):\n    if data[i] == '1':\n        return True...
Engram-tuned: len(data)):\n    if data[i] == '1':\n        return True...
```

With single-epoch training, outputs between baseline and tuned models are similar. Additional epochs would be needed to see behavioral changes comparable to the MLX results.

## Platform Comparison

| Feature | MLX (Apple Silicon) | CUDA (NVIDIA) |
|---------|---------------------|---------------|
| Training Time | ~10s (100 iter) | ~26s (1 epoch) |
| Loss Reduction | 58.2% | 20.4% |
| Accuracy Change | +33.3% relative | -27.2% relative |
| Training Epochs | ~3.7 equiv | 1 |
| Hardware Tested | M-series Mac | RTX 3060 |

### Key Differences

1. **Training Duration**: The MLX version runs 100 iterations over the dataset (~3.7 epochs), while CUDA ran only 1 epoch.

2. **Convergence**: More training iterations on MLX led to better loss reduction and accuracy improvement.

3. **Pipeline Validation**: The CUDA results validate that the Unsloth pipeline works end-to-end, even if single-epoch results are not optimal.

## Recommendations

For better CUDA results, consider:

```bash
# Increase epochs
python -m src.train --epochs 3

# Adjust learning rate
python -m src.train --learning-rate 2e-5

# Use larger LoRA rank
python -m src.train --lora-rank 16
```

## Reproduction

```bash
# From unsloth-nvidia directory
cd unsloth-nvidia
source .venv/bin/activate
./scripts/run_all.sh
```

See [unsloth-nvidia/README.md](../unsloth-nvidia/README.md) for full setup instructions.

## Related Results

- [MLX Results (Apple Silicon)](results-mlx.md) - More extensive training with better accuracy improvement
