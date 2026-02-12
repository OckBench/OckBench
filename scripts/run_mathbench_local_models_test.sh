#!/bin/bash
#
# Lite test version of run_mathbench_local_models.sh
# Uses test_small.jsonl (10 problems) and Qwen3-8B with small context window.
#

set -e

cd "$(dirname "$0")/.."
source .venv/bin/activate

DATASET="data/test_small.jsonl"
EVALUATOR="math"
CONCURRENCY=5
TIMEOUT=300
DRY_RUN=false

[[ "$1" == "--dry-run" ]] && DRY_RUN=true

# =============================================================================
# CONFIGURE YOUR SERVER HERE
# =============================================================================
BASE_URL="http://localhost:8000/v1"
API_KEY="dummy"

CACHE_DIR="cache"
mkdir -p "$CACHE_DIR"

echo "========================================"
echo "MathBench Local Models Sweep (TEST)"
echo "Server: $BASE_URL"
echo "========================================"

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
        --enforce-output-format \
        --cache $cache_file \
        $thinking_flag"

    if $DRY_RUN; then
        echo "[dry-run] $cmd"
    else
        eval "$cmd" || echo "WARNING: $model failed"
    fi
}

# =============================================================================
# TEST MODEL CONFIGURATIONS (small model, small context)
# =============================================================================

# Qwen3-8B with thinking on and off
run "Qwen/Qwen3-8B" 4096 true
run "Qwen/Qwen3-8B" 4096 false

echo ""
echo "========================================"
echo "Complete. Results in results/"
echo "Cache files in $CACHE_DIR/"
echo "========================================"
