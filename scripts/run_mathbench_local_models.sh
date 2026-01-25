#!/bin/bash
#
# Run MathBench_Combined on locally-served models (vLLM/SGLang/LMDeploy).
#
# Setup:
#   1. Clone the repo and install dependencies:
#      git clone <repo_url> && cd OckBench
#      python -m venv .venv && source .venv/bin/activate
#      uv pip install -r requirements.txt
#
#   2. Start your model server (e.g., vLLM):
#      vllm serve <model_name> --port 8000 --tensor-parallel-size <N>
#
#   3. Edit BASE_URL below to match your server, then run:
#      ./scripts/run_mathbench_local_models.sh [--dry-run]

set -e

cd "$(dirname "$0")/.."
source .venv/bin/activate

DATASET="data/MathBench_Combined.jsonl"
EVALUATOR="math"
CONCURRENCY=50
TIMEOUT=3600
DRY_RUN=false

[[ "$1" == "--dry-run" ]] && DRY_RUN=true

# =============================================================================
# CONFIGURE YOUR SERVER HERE
# =============================================================================
BASE_URL="http://localhost:8000/v1"
API_KEY="dummy"  # vLLM doesn't require a real key

echo "========================================"
echo "MathBench Local Models Sweep"
echo "Server: $BASE_URL"
echo "========================================"

run() {
    local model=$1
    local max_tokens=$2

    echo ""
    echo ">>> $model (max_tokens=$max_tokens)"

    local cmd="python main.py \
        --model $model \
        --provider generic \
        --base-url $BASE_URL \
        --api-key $API_KEY \
        --dataset-path $DATASET \
        --evaluator-type $EVALUATOR \
        --max-output-tokens $max_tokens \
        --concurrency $CONCURRENCY \
        --timeout $TIMEOUT"

    if $DRY_RUN; then
        echo "[dry-run] $cmd"
    else
        eval "$cmd" || echo "WARNING: $model failed"
    fi
}

# =============================================================================
# MODEL CONFIGURATIONS
# =============================================================================

# GLM-4.7 series (max context: 202752)
run "zai-org/GLM-4.7" 202752
run "zai-org/GLM-4.7-Flash" 202752

# DeepSeek-V3.2 (max context: 163840)
run "deepseek-ai/DeepSeek-V3.2" 163840

# Kimi-K2 series (max context: 262144)
run "moonshotai/Kimi-K2-Instruct" 262144
run "moonshotai/Kimi-K2-Thinking" 262144

# Qwen3-235B-A22B series (max context: 262144)
run "Qwen/Qwen3-235B-A22B-Instruct-2507" 262144
run "Qwen/Qwen3-235B-A22B-Thinking-2507" 262144

echo ""
echo "========================================"
echo "Complete. Results in results/"
echo "========================================"
