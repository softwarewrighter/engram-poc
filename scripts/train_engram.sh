#!/bin/bash
# Train Engram memory module with auto device detection
#
# Usage:
#   ./scripts/train_engram.sh              # Auto mode (recommended settings)
#   ./scripts/train_engram.sh --use-lora   # With LoRA adapters
#   ./scripts/train_engram.sh --device-info # Check device info
#
# Environment:
#   ENGRAM_EPOCHS     - Number of epochs (default: 3)
#   ENGRAM_BATCH_SIZE - Batch size (default: auto)
#   ENGRAM_MEMORY     - Memory table size (default: auto)

set -e

cd "$(dirname "$0")/.."

echo "=========================================="
echo "Engram Memory Training"
echo "=========================================="

# Detect which Python/venv to use
if [ -f ".venv-torch/bin/python" ]; then
    PYTHON=".venv-torch/bin/python"
elif [ -f ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
else
    PYTHON="python3"
fi

echo "Using Python: $PYTHON"

# Check for --device-info flag
if [[ "$*" == *"--device-info"* ]]; then
    $PYTHON -m src.memory.train_engram --device-info
    exit 0
fi

# Default to auto mode unless specific args given
AUTO_MODE="--auto"
if [[ "$*" == *"--batch-size"* ]] || [[ "$*" == *"--memory-size"* ]]; then
    AUTO_MODE=""
fi

# Environment variable overrides
EXTRA_ARGS=""
if [ -n "$ENGRAM_EPOCHS" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --epochs $ENGRAM_EPOCHS"
fi
if [ -n "$ENGRAM_BATCH_SIZE" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --batch-size $ENGRAM_BATCH_SIZE"
    AUTO_MODE=""
fi
if [ -n "$ENGRAM_MEMORY" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --memory-size $ENGRAM_MEMORY"
    AUTO_MODE=""
fi

# Check for training data
if [ ! -f "data/train.jsonl" ]; then
    echo "Training data not found. Generating..."
    $PYTHON -m src.data_gen.generate
fi

# Run training
echo ""
echo "Starting training..."
$PYTHON -m src.memory.train_engram $AUTO_MODE $EXTRA_ARGS "$@"

echo ""
echo "=========================================="
echo "Training complete!"
echo "=========================================="
