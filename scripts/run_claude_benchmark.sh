#!/usr/bin/env bash
set -euo pipefail

# OckBench Claude Benchmark Runner
# Usage:
#   ./scripts/run_claude_benchmark.sh              # run all tasks (math, coding, science)
#   ./scripts/run_claude_benchmark.sh math          # run math only
#   ./scripts/run_claude_benchmark.sh coding        # run coding only
#   ./scripts/run_claude_benchmark.sh science       # run science only

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

# --- Config ---
BASE_URL="https://api.anthropic.com/v1/"
CONCURRENCY=20
TIMEOUT=600

declare -A MODEL_TOKENS=(
    ["claude-opus-4-6"]=128000
    ["claude-sonnet-4-6"]=64000
    ["claude-haiku-4-5-20251001"]=64000
)

MODELS=("claude-opus-4-6" "claude-sonnet-4-6" "claude-haiku-4-5-20251001")
ALL_TASKS=("math" "coding" "science")

# --- Preflight checks ---
if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
    echo "Error: ANTHROPIC_API_KEY is not set."
    echo "  export ANTHROPIC_API_KEY=\"your-key\""
    exit 1
fi

if [[ ! -f "main.py" ]]; then
    echo "Error: main.py not found. Run this script from the OckBench root directory."
    exit 1
fi

# --- Parse tasks to run ---
if [[ $# -gt 0 ]]; then
    TASKS=("$@")
else
    TASKS=("${ALL_TASKS[@]}")
fi

# Validate task names
for task in "${TASKS[@]}"; do
    if [[ ! " ${ALL_TASKS[*]} " =~ " ${task} " ]]; then
        echo "Error: unknown task '$task'. Valid tasks: ${ALL_TASKS[*]}"
        exit 1
    fi
done

mkdir -p cache results

# --- Run benchmarks ---
for task in "${TASKS[@]}"; do
    echo "========================================="
    echo " Task: $task"
    echo "========================================="
    for model in "${MODELS[@]}"; do
        max_tokens="${MODEL_TOKENS[$model]}"
        # Short name for cache file (e.g. opus, sonnet, haiku)
        short_name="${model#claude-}"
        short_name="${short_name%%-*}"
        cache_file="cache/${task}_${short_name}.jsonl"

        echo ""
        echo "--- $model | task=$task | max_tokens=$max_tokens ---"
        python main.py --provider chat_completion --model "$model" \
            --base-url "$BASE_URL" \
            --api-key "$ANTHROPIC_API_KEY" \
            --task "$task" --max-output-tokens "$max_tokens" \
            --output-dir "results/full_pool/${task}" \
            --concurrency "$CONCURRENCY" --timeout "$TIMEOUT" \
            --cache "$cache_file"
    done
done

echo ""
echo "========================================="
echo " All done! Result files:"
echo "========================================="
ls -1 results/full_pool/*/OckBench_*_claude-* 2>/dev/null || echo "(no result files found)"
