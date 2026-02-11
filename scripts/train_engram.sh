#!/bin/bash
# Train Engram memory module
# Usage: ./scripts/train_engram.sh [--use-lora]

set -e

cd "$(dirname "$0")/.."

echo "=========================================="
echo "Engram Memory Training"
echo "=========================================="

# Default arguments
EPOCHS=3
BATCH_SIZE=4
MEMORY_SIZE=50000
USE_LORA=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --use-lora)
            USE_LORA="--use-lora"
            shift
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --memory-size)
            MEMORY_SIZE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "Configuration:"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Memory size: $MEMORY_SIZE"
echo "  Use LoRA: ${USE_LORA:-no}"
echo ""

# Check for training data
if [ ! -f "data/train.jsonl" ]; then
    echo "Training data not found. Generating..."
    python -m src.data_gen.generate
fi

# Run training
python -m src.memory.train_engram \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --memory-size "$MEMORY_SIZE" \
    $USE_LORA

echo ""
echo "Training complete!"
echo "Weights saved to: adapters-engram/"
