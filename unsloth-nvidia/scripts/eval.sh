#!/usr/bin/env bash
# Engram PoC (Unsloth/NVIDIA) - Evaluation Script

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Configuration
MODEL="${MODEL:-HuggingFaceTB/SmolLM-135M-Instruct}"
ADAPTER_DIR="${ADAPTER_DIR:-./adapters}"
TEST_FILE="${TEST_FILE:-./data/test.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-./results}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║      Engram PoC - Evaluation                               ║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check prerequisites
if [[ ! -f "${TEST_FILE}" ]]; then
    echo -e "${YELLOW}Test file not found. Generating data...${NC}"
    python -m src.data_gen
fi

if [[ ! -d "${ADAPTER_DIR}" ]]; then
    echo -e "${RED}Error: Adapter not found at ${ADAPTER_DIR}${NC}"
    echo -e "${YELLOW}Run ./scripts/train.sh first${NC}"
    exit 1
fi

echo -e "${GREEN}Configuration:${NC}"
echo "  Model:     $MODEL"
echo "  Adapter:   $ADAPTER_DIR"
echo "  Test File: $TEST_FILE"
echo "  Output:    $OUTPUT_DIR"
echo ""

python -m src.eval \
    --model "$MODEL" \
    --adapter-path "$ADAPTER_DIR" \
    --test-file "$TEST_FILE" \
    --output-dir "$OUTPUT_DIR" \
    "$@"
