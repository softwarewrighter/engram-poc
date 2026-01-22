#!/usr/bin/env bash
# Engram PoC - GPU Evaluation Script
# Compare baseline vs tuned model on NVIDIA GPU

set -euo pipefail

# Configuration
MODEL="${MODEL:-HuggingFaceTB/SmolLM-135M-Instruct}"
ADAPTER_DIR="${ADAPTER_DIR:-./adapters-gpu}"
TEST_FILE="${TEST_FILE:-./data/test.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-./results}"
MAX_TOKENS="${MAX_TOKENS:-50}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║      Engram PoC - GPU Evaluation                           ║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check for CUDA
if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo -e "${RED}Error: CUDA not available${NC}"
    exit 1
fi

# Check prerequisites
if [[ ! -f "${TEST_FILE}" ]]; then
    echo -e "${YELLOW}Test file not found. Generating data...${NC}"
    python -m src.data_gen.generate
fi

if [[ ! -d "${ADAPTER_DIR}" ]]; then
    echo -e "${RED}Error: Adapter not found at ${ADAPTER_DIR}${NC}"
    echo -e "${YELLOW}Run ./scripts/train_gpu.sh first to train the model.${NC}"
    exit 1
fi

TEST_COUNT=$(wc -l < "${TEST_FILE}" | tr -d ' ')

echo -e "${GREEN}Configuration:${NC}"
echo "  Model:      $MODEL"
echo "  Adapter:    $ADAPTER_DIR"
echo "  Test File:  $TEST_FILE ($TEST_COUNT examples)"
echo "  Output:     $OUTPUT_DIR"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run evaluation
echo -e "${GREEN}Running GPU evaluation...${NC}"
echo ""

python -m src.eval_gpu.compare \
    --model "$MODEL" \
    --adapter-path "$ADAPTER_DIR" \
    --test-file "$TEST_FILE" \
    --output-dir "$OUTPUT_DIR" \
    --max-tokens "$MAX_TOKENS"

echo ""
echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}Evaluation Complete!${NC}"
echo ""
echo -e "${GREEN}Output: ${OUTPUT_DIR}/comparison_gpu.json${NC}"
