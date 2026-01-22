#!/usr/bin/env bash
# Engram PoC - Evaluation Script
# Compares baseline model vs Engram-tuned model

set -euo pipefail

# Configuration
MODEL="${MODEL:-HuggingFaceTB/SmolLM-135M-Instruct}"
ADAPTER_DIR="${ADAPTER_DIR:-./adapters}"
TEST_FILE="${TEST_FILE:-./data/test.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-./results}"
MAX_TOKENS="${MAX_TOKENS:-50}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║           Engram PoC - Model Evaluation                    ║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check prerequisites
if [[ ! -f "${TEST_FILE}" ]]; then
    echo -e "${YELLOW}Test file not found. Generating data...${NC}"
    python -m src.data_gen.generate
fi

if [[ ! -f "${ADAPTER_DIR}/adapters.safetensors" ]]; then
    echo -e "${RED}Error: Adapter not found at ${ADAPTER_DIR}${NC}"
    echo -e "${YELLOW}Run ./scripts/train.sh first to train the model.${NC}"
    exit 1
fi

# Count test examples
TEST_COUNT=$(wc -l < "${TEST_FILE}" | tr -d ' ')

echo -e "${GREEN}Configuration:${NC}"
echo "  Model:      $MODEL"
echo "  Adapter:    $ADAPTER_DIR"
echo "  Test File:  $TEST_FILE ($TEST_COUNT examples)"
echo "  Output:     $OUTPUT_DIR"
echo "  Max Tokens: $MAX_TOKENS"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run comparison
echo -e "${GREEN}Running evaluation...${NC}"
echo ""

python -m src.eval.compare \
    --model "$MODEL" \
    --adapter-path "$ADAPTER_DIR" \
    --test-file "$TEST_FILE" \
    --output-dir "$OUTPUT_DIR" \
    --max-tokens "$MAX_TOKENS"

echo ""
echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}Evaluation Complete!${NC}"
echo ""
echo -e "${GREEN}Output files:${NC}"
echo "  - ${OUTPUT_DIR}/baseline.json"
echo "  - ${OUTPUT_DIR}/tuned.json"
echo "  - ${OUTPUT_DIR}/comparison.json"
echo "  - ${OUTPUT_DIR}/evaluation_report.md"
