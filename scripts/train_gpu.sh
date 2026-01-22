#!/usr/bin/env bash
# Engram PoC - GPU Training Script (Unsloth)
# Fast LoRA fine-tuning on NVIDIA GPUs

set -euo pipefail

# Configuration
MODEL="${MODEL:-HuggingFaceTB/SmolLM-135M-Instruct}"
DATA_DIR="${DATA_DIR:-./data}"
ADAPTER_DIR="${ADAPTER_DIR:-./adapters-gpu}"
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
LORA_RANK="${LORA_RANK:-8}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║      Engram PoC - GPU Training (Unsloth)                   ║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check for CUDA
if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo -e "${RED}Error: CUDA not available${NC}"
    echo -e "${YELLOW}Make sure you have an NVIDIA GPU and CUDA installed${NC}"
    exit 1
fi

# Show GPU info
echo -e "${GREEN}GPU Info:${NC}"
python -c "import torch; print(f'  Device: {torch.cuda.get_device_name(0)}')"
python -c "import torch; print(f'  CUDA Version: {torch.version.cuda}')"
echo ""

# Check for training data
if [[ ! -f "${DATA_DIR}/train.jsonl" ]]; then
    echo -e "${YELLOW}Training data not found. Generating...${NC}"
    python -m src.data_gen.generate
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

# Run training
echo -e "${GREEN}Starting Unsloth training...${NC}"
echo ""

python -m src.train_gpu.train \
    --model "$MODEL" \
    --output-dir "$ADAPTER_DIR" \
    --data-dir "$DATA_DIR" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --learning-rate "$LEARNING_RATE" \
    --lora-rank "$LORA_RANK" \
    "$@"

echo ""
echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}Training Complete!${NC}"
echo ""
echo -e "${GREEN}Adapter saved to: ${ADAPTER_DIR}${NC}"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "  - Evaluate: python -m src.eval_gpu.compare"
echo "  - Demo: python -m src.demo.demo_unified"
