#!/usr/bin/env bash
# Engram PoC - Training Script
# Fine-tunes a small model with LoRA on pattern data

set -euo pipefail

# Configuration
MODEL="${MODEL:-HuggingFaceTB/SmolLM-135M-Instruct}"
DATA_DIR="${DATA_DIR:-./data}"
ADAPTER_DIR="${ADAPTER_DIR:-./adapters}"
CONFIG_FILE="${CONFIG_FILE:-./configs/lora_config.yaml}"
ITERS="${ITERS:-100}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
NUM_LAYERS="${NUM_LAYERS:-16}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║           Engram PoC - LoRA Training                       ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check if data exists
if [[ ! -f "${DATA_DIR}/train.jsonl" ]]; then
    echo -e "${YELLOW}Training data not found. Generating...${NC}"
    python -m src.data_gen.generate
fi

echo -e "${GREEN}Configuration:${NC}"
echo "  Model:         $MODEL"
echo "  Data:          $DATA_DIR"
echo "  Adapter:       $ADAPTER_DIR"
echo "  Iterations:    $ITERS"
echo "  Batch Size:    $BATCH_SIZE"
echo "  Num Layers:    $NUM_LAYERS"
echo "  Learning Rate: $LEARNING_RATE"
echo ""

# Show data stats
TRAIN_COUNT=$(wc -l < "${DATA_DIR}/train.jsonl" | tr -d ' ')
VALID_COUNT=$(wc -l < "${DATA_DIR}/valid.jsonl" | tr -d ' ')
echo -e "${GREEN}Dataset:${NC}"
echo "  Training:   $TRAIN_COUNT examples"
echo "  Validation: $VALID_COUNT examples"
echo ""

# Create adapter directory
mkdir -p "$ADAPTER_DIR"

# Run training
echo -e "${GREEN}Starting LoRA fine-tuning...${NC}"
echo ""

time mlx_lm.lora \
    --model "$MODEL" \
    --train \
    --data "$DATA_DIR" \
    --adapter-path "$ADAPTER_DIR" \
    --iters "$ITERS" \
    --batch-size "$BATCH_SIZE" \
    --num-layers "$NUM_LAYERS" \
    --learning-rate "$LEARNING_RATE" \
    --steps-per-report 10 \
    --steps-per-eval 50

echo ""
echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}Training Complete!${NC}"
echo ""

# Show adapter info
if [[ -f "${ADAPTER_DIR}/adapters.safetensors" ]]; then
    ADAPTER_SIZE=$(ls -lh "${ADAPTER_DIR}/adapters.safetensors" | awk '{print $5}')
    echo -e "${GREEN}Adapter saved:${NC}"
    echo "  Path: ${ADAPTER_DIR}/adapters.safetensors"
    echo "  Size: ${ADAPTER_SIZE}"
else
    echo -e "${RED}Warning: Adapter file not found${NC}"
fi

echo ""
echo -e "${GREEN}To test the adapter:${NC}"
echo "  mlx_lm.generate --model $MODEL \\"
echo "    --adapter-path $ADAPTER_DIR \\"
echo "    --prompt \"Complete: for i in range(\""
