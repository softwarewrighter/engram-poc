#!/usr/bin/env bash
# Engram PoC (Unsloth/NVIDIA) - Full Pipeline
# Runs: data generation -> training -> evaluation -> demo

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# Configuration
MODEL="${MODEL:-HuggingFaceTB/SmolLM-135M-Instruct}"
SKIP_TRAIN="${SKIP_TRAIN:-false}"
SKIP_EVAL="${SKIP_EVAL:-false}"
SKIP_DEMO="${SKIP_DEMO:-false}"

print_banner() {
    echo -e "${GREEN}"
    echo "╔══════════════════════════════════════════════════════════════════════╗"
    echo "║                                                                      ║"
    echo "║   ███████╗███╗   ██╗ ██████╗ ██████╗  █████╗ ███╗   ███╗            ║"
    echo "║   ██╔════╝████╗  ██║██╔════╝ ██╔══██╗██╔══██╗████╗ ████║            ║"
    echo "║   █████╗  ██╔██╗ ██║██║  ███╗██████╔╝███████║██╔████╔██║            ║"
    echo "║   ██╔══╝  ██║╚██╗██║██║   ██║██╔══██╗██╔══██║██║╚██╔╝██║            ║"
    echo "║   ███████╗██║ ╚████║╚██████╔╝██║  ██║██║  ██║██║ ╚═╝ ██║            ║"
    echo "║   ╚══════╝╚═╝  ╚═══╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝            ║"
    echo "║                                                                      ║"
    echo "║              Unsloth / NVIDIA GPU Edition                            ║"
    echo "║                                                                      ║"
    echo "╚══════════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

print_step() {
    local step="$1"
    local description="$2"
    echo ""
    echo -e "${CYAN}╔════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║  ${BOLD}Step ${step}: ${description}${NC}"
    echo -e "${CYAN}╚════════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

check_cuda() {
    if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        echo -e "${RED}Error: CUDA not available${NC}"
        echo -e "${YELLOW}Make sure you have an NVIDIA GPU and CUDA installed${NC}"
        exit 1
    fi
    echo -e "${GREEN}GPU: $(python -c "import torch; print(torch.cuda.get_device_name(0))")${NC}"
}

step_data_gen() {
    print_step "1/4" "Generate Training Data"

    if [[ -f "data/train.jsonl" ]]; then
        echo -e "${YELLOW}Training data already exists. Regenerating...${NC}"
    fi

    python -m src.data_gen

    echo ""
    echo -e "${GREEN}Data generation complete.${NC}"
    echo "  - train.jsonl: $(wc -l < data/train.jsonl | tr -d ' ') examples"
    echo "  - valid.jsonl: $(wc -l < data/valid.jsonl | tr -d ' ') examples"
    echo "  - test.jsonl:  $(wc -l < data/test.jsonl | tr -d ' ') examples"
}

step_train() {
    print_step "2/4" "LoRA Fine-Tuning (Unsloth)"

    if [[ "$SKIP_TRAIN" == "true" ]]; then
        echo -e "${YELLOW}Skipping training (SKIP_TRAIN=true)${NC}"
        if [[ ! -d "adapters" ]]; then
            echo -e "${RED}Error: No existing adapter found.${NC}"
            exit 1
        fi
        return
    fi

    ./scripts/train.sh
}

step_eval() {
    print_step "3/4" "Evaluation"

    if [[ "$SKIP_EVAL" == "true" ]]; then
        echo -e "${YELLOW}Skipping evaluation (SKIP_EVAL=true)${NC}"
        return
    fi

    ./scripts/eval.sh
}

step_demo() {
    print_step "4/4" "Demo"

    if [[ "$SKIP_DEMO" == "true" ]]; then
        echo -e "${YELLOW}Skipping demo (SKIP_DEMO=true)${NC}"
        return
    fi

    ./scripts/demo.sh --quick
}

print_summary() {
    echo ""
    echo -e "${GREEN}╔══════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║                      PIPELINE COMPLETE                               ║${NC}"
    echo -e "${GREEN}╚══════════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${BOLD}Outputs:${NC}"
    echo "  - Training data:    data/{train,valid,test}.jsonl"
    echo "  - Adapter weights:  adapters/"
    echo "  - Evaluation:       results/comparison.json"
    echo ""
    echo -e "${BOLD}Next steps:${NC}"
    echo "  - Interactive demo: python -m src.demo"
    echo "  - Shell demo:       ./scripts/demo.sh"
    echo ""
}

main() {
    print_banner

    echo -e "${CYAN}Model: ${MODEL}${NC}"
    echo ""

    check_cuda

    step_data_gen
    step_train
    step_eval
    step_demo

    print_summary
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-train) SKIP_TRAIN=true; shift ;;
        --skip-eval) SKIP_EVAL=true; shift ;;
        --skip-demo) SKIP_DEMO=true; shift ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --skip-train    Skip training step"
            echo "  --skip-eval     Skip evaluation step"
            echo "  --skip-demo     Skip demo step"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

main
