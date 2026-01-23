#!/usr/bin/env bash
# Engram PoC (Unsloth/NVIDIA) - Training Script

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Configuration
MODEL="${MODEL:-HuggingFaceTB/SmolLM-135M-Instruct}"
DATA_DIR="${DATA_DIR:-./data}"
ADAPTER_DIR="${ADAPTER_DIR:-./adapters}"
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
LORA_RANK="${LORA_RANK:-8}"

# Colors
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║      Engram PoC - Unsloth Training                         ║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check for training data
if [[ ! -f "${DATA_DIR}/train.jsonl" ]]; then
    echo "Generating training data..."
    python -m src.data_gen
fi

echo -e "${GREEN}Configuration:${NC}"
echo "  Model:         $MODEL"
echo "  Data:          $DATA_DIR"
echo "  Output:        $ADAPTER_DIR"
echo "  Epochs:        $EPOCHS"
echo "  Batch Size:    $BATCH_SIZE"
echo "  Learning Rate: $LEARNING_RATE"
echo "  LoRA Rank:     $LORA_RANK"
echo ""

python -m src.train \
    --model "$MODEL" \
    --output-dir "$ADAPTER_DIR" \
    --data-dir "$DATA_DIR" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --learning-rate "$LEARNING_RATE" \
    --lora-rank "$LORA_RANK" \
    "$@"
