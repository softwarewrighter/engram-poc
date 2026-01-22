#!/usr/bin/env bash
# Engram PoC - Demo Script
# Visual comparison of baseline vs Engram-tuned model

set -euo pipefail

# Configuration
MODEL="${MODEL:-HuggingFaceTB/SmolLM-135M-Instruct}"
ADAPTER_DIR="${ADAPTER_DIR:-./adapters}"
MAX_TOKENS="${MAX_TOKENS:-30}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m'

# Demo prompts
declare -a PROMPTS=(
    "Complete: for i in range("
    "Complete: if __name__ == "
    "Q: HTTP status code for 'Not Found'?\nA:"
    "Q: Default port for SSH?\nA:"
    "Format date: 2024-01-15 ->"
    "Fix: if x = 5:"
)

declare -a DESCRIPTIONS=(
    "Loop pattern completion"
    "Main guard pattern"
    "HTTP status codes"
    "Network ports"
    "Date formatting"
    "Assignment vs comparison fix"
)

print_header() {
    echo ""
    echo -e "${CYAN}======================================================================${NC}"
    echo -e "${CYAN}${BOLD}           ENGRAM PoC - Before/After Demonstration${NC}"
    echo -e "${CYAN}======================================================================${NC}"
    echo ""
    echo -e "${DIM}Comparing baseline model vs Engram-tuned model${NC}"
    echo -e "${DIM}Watch for: more concise, pattern-aligned responses${NC}"
    echo ""
}

check_prerequisites() {
    if [[ ! -f "${ADAPTER_DIR}/adapters.safetensors" ]]; then
        echo -e "${RED}Error: Adapter not found at ${ADAPTER_DIR}${NC}"
        echo -e "${YELLOW}Run ./scripts/train.sh first to train the model.${NC}"
        exit 1
    fi
}

run_comparison() {
    local prompt="$1"
    local description="$2"
    local index="$3"
    local total="$4"

    echo -e "${YELLOW}──────────────────────────────────────────────────────────────────────${NC}"
    echo -e "${YELLOW}[${index}/${total}] ${description}${NC}"
    echo -e "${YELLOW}──────────────────────────────────────────────────────────────────────${NC}"
    echo ""
    echo -e "${BOLD}Prompt:${NC} ${prompt}"
    echo ""

    # Get baseline response
    echo -e "${RED}Baseline:${NC}"
    mlx_lm.generate --model "$MODEL" \
        --prompt "$prompt" \
        --max-tokens "$MAX_TOKENS" 2>/dev/null | grep -v "^Prompt:" | grep -v "^Generation:" | grep -v "^Peak memory:" | grep -v "^====" | head -3
    echo ""

    # Get tuned response
    echo -e "${GREEN}Engram-tuned:${NC}"
    mlx_lm.generate --model "$MODEL" \
        --adapter-path "$ADAPTER_DIR" \
        --prompt "$prompt" \
        --max-tokens "$MAX_TOKENS" 2>/dev/null | grep -v "^Prompt:" | grep -v "^Generation:" | grep -v "^Peak memory:" | grep -v "^====" | head -3
    echo ""
}

print_summary() {
    echo -e "${CYAN}======================================================================${NC}"
    echo -e "${CYAN}${BOLD}                         SUMMARY${NC}"
    echo -e "${CYAN}======================================================================${NC}"
    echo ""
    echo -e "${BOLD}Key Observations:${NC}"
    echo "  - Engram-tuned model produces more concise outputs"
    echo "  - Pattern completion is more deterministic"
    echo "  - Factual answers are direct (no verbose explanations)"
    echo ""
    echo -e "${CYAN}======================================================================${NC}"
    echo -e "${GREEN}Demo complete!${NC}"
    echo ""
}

main() {
    print_header
    check_prerequisites

    echo -e "${CYAN}Model: ${MODEL}${NC}"
    echo -e "${CYAN}Adapter: ${ADAPTER_DIR}${NC}"
    echo ""

    local total=${#PROMPTS[@]}

    for i in "${!PROMPTS[@]}"; do
        local index=$((i + 1))
        run_comparison "${PROMPTS[$i]}" "${DESCRIPTIONS[$i]}" "$index" "$total"

        if [[ $index -lt $total ]]; then
            echo -e "${DIM}Press Enter for next example...${NC}"
            read -r
            echo ""
        fi
    done

    print_summary
}

# Run with --quick flag for non-interactive mode
if [[ "${1:-}" == "--quick" ]]; then
    print_header
    check_prerequisites
    echo -e "${CYAN}Running quick demo (non-interactive)...${NC}"
    echo ""

    # Just run first 3 examples without pausing
    for i in 0 1 2; do
        run_comparison "${PROMPTS[$i]}" "${DESCRIPTIONS[$i]}" "$((i + 1))" "3"
    done

    print_summary
else
    main
fi
