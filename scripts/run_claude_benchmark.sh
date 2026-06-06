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

# Math is scored by an OpenAI-compatible LLM judge. Anthropic's API is NOT
# OpenAI chat-completions compatible, so the judge must use a separate
# OpenAI-compatible endpoint (default OpenAI; override via env). Its key comes
# from JUDGE_API_KEY, falling back to OPENAI_API_KEY.
JUDGE_MODEL="${JUDGE_MODEL:-gpt-4o-mini}"
JUDGE_BASE_URL="${JUDGE_BASE_URL:-https://api.openai.com/v1}"
JUDGE_API_KEY="${JUDGE_API_KEY:-${OPENAI_API_KEY:-}}"

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

# Math needs an OpenAI-compatible judge key (the Anthropic key cannot judge).
for task in "${TASKS[@]}"; do
    if [[ "$task" == "math" && -z "$JUDGE_API_KEY" ]]; then
        echo "Error: math scoring needs an OpenAI-compatible judge."
        echo "  Set JUDGE_API_KEY (and optionally JUDGE_MODEL/JUDGE_BASE_URL), or OPENAI_API_KEY."
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

        # Math requires an OpenAI-compatible LLM judge (not the Anthropic endpoint).
        judge_args=()
        if [[ "$task" == "math" ]]; then
            judge_args=(--judge-model "$JUDGE_MODEL" --judge-base-url "$JUDGE_BASE_URL" --judge-api-key "$JUDGE_API_KEY")
        fi

        python main.py --provider chat_completion --model "$model" \
            --base-url "$BASE_URL" \
            --api-key "$ANTHROPIC_API_KEY" \
            --task "$task" --max-output-tokens "$max_tokens" \
            --output-dir "results/full_pool/${task}" \
            --concurrency "$CONCURRENCY" --timeout "$TIMEOUT" \
            --cache "$cache_file" \
            ${judge_args[@]+"${judge_args[@]}"}
    done
done

echo ""
echo "========================================="
echo " All done! Result files:"
echo "========================================="
ls -1 results/full_pool/*/OckBench_*_claude-* 2>/dev/null || echo "(no result files found)"
