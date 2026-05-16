#!/usr/bin/env bash
# Train and evaluate all three architectures sequentially.
set -euo pipefail

PYTHON="conda run -n Vision python"
MAIN="main.py"

run_model() {
    local model="$1"
    echo "============================================"
    echo "  Training: $model"
    echo "============================================"
    python $MAIN --model "$model"

    echo "--------------------------------------------"
    echo "  Evaluating: $model"
    echo "--------------------------------------------"
    python $MAIN --model "$model" --eval --plots --benchmark
}

run_model resnet18
run_model simplecnn
run_model resnet34

echo "============================================"
echo "  All runs complete."
echo "============================================"
