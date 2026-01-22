#!/usr/bin/env bash
# Engram PoC - Full Pipeline Script
# Runs: data generation -> training -> evaluation -> demo

set -euo pipefail

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

print_step() {
    local step="$1"
    local description="$2"
    echo ""
    echo -e "${CYAN}╔════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║  ${BOLD}Step ${step}: ${description}${NC}"
    echo -e "${CYAN}╚════════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

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
    echo "║                    Proof of Concept Pipeline                         ║"
    echo "║                                                                      ║"
    echo "╚══════════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

check_venv() {
    if [[ -z "${VIRTUAL_ENV:-}" ]]; then
        if [[ -f ".venv/bin/activate" ]]; then
            echo -e "${YELLOW}Activating virtual environment...${NC}"
            source .venv/bin/activate
        else
            echo -e "${RED}Error: Virtual environment not found.${NC}"
            echo -e "${YELLOW}Run: uv venv && source .venv/bin/activate && uv pip install -r requirements.txt${NC}"
            exit 1
        fi
    fi
}

step_data_gen() {
    print_step "1/4" "Generate Training Data"

    if [[ -f "data/train.jsonl" ]] && [[ -f "data/valid.jsonl" ]]; then
        echo -e "${YELLOW}Training data already exists. Regenerating...${NC}"
    fi

    python -m src.data_gen.generate

    echo ""
    echo -e "${GREEN}Data generation complete.${NC}"
    echo "  - train.jsonl: $(wc -l < data/train.jsonl | tr -d ' ') examples"
    echo "  - valid.jsonl: $(wc -l < data/valid.jsonl | tr -d ' ') examples"
    echo "  - test.jsonl:  $(wc -l < data/test.jsonl | tr -d ' ') examples"
}

step_train() {
    print_step "2/4" "LoRA Fine-Tuning"

    if [[ "$SKIP_TRAIN" == "true" ]]; then
        echo -e "${YELLOW}Skipping training (SKIP_TRAIN=true)${NC}"
        if [[ ! -f "adapters/adapters.safetensors" ]]; then
            echo -e "${RED}Error: No existing adapter found. Cannot skip training.${NC}"
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
    echo "  - Adapter weights:  adapters/adapters.safetensors"
    echo "  - Evaluation:       results/comparison.json"
    echo "  - Report:           results/evaluation_report.md"
    echo ""
    echo -e "${BOLD}Next steps:${NC}"
    echo "  - Interactive demo: python -m src.demo.demo"
    echo "  - Shell demo:       ./scripts/demo.sh"
    echo "  - View report:      cat results/evaluation_report.md"
    echo ""
}

main() {
    print_banner

    echo -e "${CYAN}Model: ${MODEL}${NC}"
    echo ""

    check_venv

    step_data_gen
    step_train
    step_eval
    step_demo

    print_summary
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-train)
            SKIP_TRAIN=true
            shift
            ;;
        --skip-eval)
            SKIP_EVAL=true
            shift
            ;;
        --skip-demo)
            SKIP_DEMO=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --skip-train    Skip training step (use existing adapter)"
            echo "  --skip-eval     Skip evaluation step"
            echo "  --skip-demo     Skip demo step"
            echo "  --help          Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

main
