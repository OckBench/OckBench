#!/bin/bash
#
# Run MathBench_Combined across OpenAI models with varying reasoning effort.
#
# Usage:
#   export OPENAI_API_KEY="sk-..."
#   ./scripts/run_mathbench_openai.sh [--dry-run]

set -e

# Change to project root and activate venv
cd "$(dirname "$0")/.."
source .venv/bin/activate

DATASET="data/MathBench_Combined.jsonl"
EVALUATOR="math"
CONCURRENCY=50
DRY_RUN=false

[[ "$1" == "--dry-run" ]] && DRY_RUN=true
[[ -z "$OPENAI_API_KEY" ]] && echo "Error: OPENAI_API_KEY not set" && exit 1

echo "========================================"
echo "MathBench OpenAI Sweep"
echo "========================================"

# Format: MODEL,MAX_TOKENS,REASONING_EFFORTS (comma-separated, empty=standard model)
# Standard models: single run, no reasoning_effort
# Reasoning models: run for each effort level
CONFIGS=(
    # Standard models
    "gpt-4o,16384,"
    "gpt-4.1,32768,"

    # Reasoning models - test low/medium/high
    "o3-mini,100000,low medium high"
    "o4-mini,100000,low medium high"
    "gpt-5-mini,128000,low medium high"
    "gpt-5.2,128000,low medium high"
)

run() {
    local model=$1
    local max_tokens=$2
    local effort=$3

    local name="$model"
    [[ -n "$effort" ]] && name="${model}_${effort}"

    echo ""
    echo ">>> $name (max_tokens=$max_tokens)"

    local cmd="python main.py \
        --model $model \
        --provider openai \
        --dataset-path $DATASET \
        --evaluator-type $EVALUATOR \
        --max-output-tokens $max_tokens \
        --concurrency $CONCURRENCY \
        --timeout 600"

    [[ -n "$effort" ]] && cmd="$cmd --reasoning-effort $effort"

    if $DRY_RUN; then
        echo "[dry-run] $cmd"
    else
        eval "$cmd" || echo "WARNING: $name failed"
    fi
}

for config in "${CONFIGS[@]}"; do
    IFS=',' read -r MODEL MAX_TOKENS EFFORTS <<< "$config"

    if [[ -z "$EFFORTS" ]]; then
        # Standard model
        run "$MODEL" "$MAX_TOKENS" ""
    else
        # Reasoning model - run each effort level
        for effort in $EFFORTS; do
            run "$MODEL" "$MAX_TOKENS" "$effort"
        done
    fi
done

echo ""
echo "========================================"
echo "Complete. Results in results/"
echo "========================================"
