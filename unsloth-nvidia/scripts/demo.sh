#!/usr/bin/env bash
# Engram PoC (Unsloth/NVIDIA) - Demo Script

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Configuration
MODEL="${MODEL:-HuggingFaceTB/SmolLM-135M-Instruct}"
ADAPTER_DIR="${ADAPTER_DIR:-./adapters}"

# Colors
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Check for adapter
if [[ ! -d "${ADAPTER_DIR}" ]]; then
    echo -e "${RED}Error: Adapter not found at ${ADAPTER_DIR}${NC}"
    echo -e "${YELLOW}Run ./scripts/train.sh first${NC}"
    exit 1
fi

python -m src.demo \
    --model "$MODEL" \
    --adapter-path "$ADAPTER_DIR" \
    "$@"
