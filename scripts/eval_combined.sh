#!/bin/bash
# Run combined evaluation comparing all configurations
#
# Usage:
#   ./scripts/eval_combined.sh

set -e

cd "$(dirname "$0")/.."

echo "=========================================="
echo "Combined Evaluation"
echo "=========================================="

# Detect which Python/venv to use
if [ -f ".venv-torch/bin/python" ]; then
    PYTHON=".venv-torch/bin/python"
elif [ -f ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
else
    PYTHON="python3"
fi

echo "Using Python: $PYTHON"

# Check for test data
if [ ! -f "data/test.jsonl" ]; then
    echo "Test data not found. Generating..."
    $PYTHON -m src.data_gen.generate
fi

# Run evaluation
$PYTHON -m src.memory.eval_combined "$@"

echo ""
echo "=========================================="
echo "Evaluation complete!"
echo "=========================================="
