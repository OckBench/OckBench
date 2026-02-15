#!/bin/bash
#
# Run MathBench_Top200_All on locally-served models (vLLM/SGLang/LMDeploy).
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

DATASET="data/MathBench_Top200_All.jsonl"
EVALUATOR="math"
CONCURRENCY=64
TIMEOUT=3600
DRY_RUN=false

# [[ "$1" == "--dry-run" ]] && DRY_RUN=true

MODEL=${1:-"Qwen/Qwen3-235B-A22B-Instruct-2507"}
MAX_TOKENS=${2:-"262144"}    

# =============================================================================
# CONFIGURE YOUR SERVER HERE
# =============================================================================
BASE_URL="http://localhost:8000/v1"
API_KEY="dummy"  # vLLM doesn't require a real key

echo "========================================"
echo "MathBench Local Models Sweep"
echo "Server: $BASE_URL"
echo "========================================"

CACHE_DIR="cache"
mkdir -p "$CACHE_DIR"

run() {
    local model=$1
    local max_tokens=$2
    local enable_thinking=${3:-}  # optional: "true" or "false"

    local thinking_desc=""
    local thinking_flag=""
    local cache_suffix=""
    if [[ -n "$enable_thinking" ]]; then
        thinking_desc=" (thinking=$enable_thinking)"
        thinking_flag="--enable-thinking $enable_thinking"
        cache_suffix="_thinking-${enable_thinking}"
    fi

    # Deterministic cache path from model name
    local safe_name="${model//\//_}"
    local cache_file="${CACHE_DIR}/${safe_name}${cache_suffix}.jsonl"

    echo ""
    echo ">>> $model (max_tokens=$max_tokens)$thinking_desc"
    echo "    cache: $cache_file"

    local cmd="python main.py \
        --model $model \
        --provider generic \
        --base-url $BASE_URL \
        --api-key $API_KEY \
        --dataset-path $DATASET \
        --evaluator-type $EVALUATOR \
        --max-output-tokens $max_tokens \
        --concurrency $CONCURRENCY \
        --timeout $TIMEOUT \
        --cache $cache_file \
        $thinking_flag"

    if $DRY_RUN; then
        echo "[dry-run] $cmd"
    else
        eval "$cmd" || echo "WARNING: $model failed"
    fi
}

# =============================================================================
# MODEL CONFIGURATIONS
# =============================================================================
run $MODEL $MAX_TOKENS

exit 0
# run "Qwen/Qwen3-235B-A22B-Instruct-2507" 262144
# GLM-4.7 series (max context: 202752)
# run "zai-org/GLM-4.7" 202752
# run "zai-org/GLM-4.7-Flash" 202752
# GLM-4.7 series (max context: 202752) — run with thinking on and off
run "zai-org/GLM-4.7" 202752 true
run "zai-org/GLM-4.7" 202752 false
run "zai-org/GLM-4.7-Flash" 202752 true
run "zai-org/GLM-4.7-Flash" 202752 false

# GLM-5 (max context: 202752) — run with thinking on and off
run "zai-org/GLM-5" 202752 true
run "zai-org/GLM-5" 202752 false

# DeepSeek-V3.2 (max context: 163840) — run with thinking on and off
run "deepseek-ai/DeepSeek-V3.2" 163840 true
run "deepseek-ai/DeepSeek-V3.2" 163840 false

# Kimi-K2 series (max context: 262144)
run "moonshotai/Kimi-K2-Instruct" 262144
run "moonshotai/Kimi-K2-Thinking" 262144

# Kimi-K2.5 (max context: 262144)
run "moonshotai/Kimi-K2.5" 262144

# # Qwen3-235B-A22B series (max context: 262144)
# run "Qwen/Qwen3-235B-A22B-Instruct-2507" 262144
# run "Qwen/Qwen3-235B-A22B-Thinking-2507" 262144

echo ""
echo "========================================"
echo "Complete. Results in results/"
echo "========================================"
