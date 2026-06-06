#!/usr/bin/env bash
set -euo pipefail

# OckBench OpenAI Benchmark Runner
#
# Benchmarks OpenAI models with per-model reasoning efforts.
#
# Usage:
#   ./scripts/run_openai_benchmark.sh                          # run all models, all tasks
#   ./scripts/run_openai_benchmark.sh --tasks math             # math only
#   ./scripts/run_openai_benchmark.sh --models gpt-5.4         # one model only

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

# --- Config ---
CONCURRENCY=20
TIMEOUT=600
OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://api.openai.com/v1}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-4o-mini}"

# Model -> max output tokens
#   gpt-4.1:      32K output,  1M context
#   gpt-5:        128K output, 400K context
#   gpt-5-mini:   128K output, 400K context
#   gpt-5.4:      128K output, ~1M context
#   gpt-5.4-mini: 128K output, 400K context
declare -A MODEL_TOKENS=(
    ["gpt-4.1"]=32768
    ["gpt-5"]=128000
    ["gpt-5-mini"]=128000
    ["gpt-5.4"]=128000
    ["gpt-5.4-mini"]=128000
)

# Model -> reasoning efforts to benchmark
declare -A MODEL_EFFORTS=(
    ["gpt-4.1"]="none"
    ["gpt-5"]="low medium high"
    ["gpt-5-mini"]="low medium high"
    ["gpt-5.4"]="none low medium high xhigh"
    ["gpt-5.4-mini"]="none low medium high xhigh"
)

ALL_MODELS=("gpt-4.1" "gpt-5" "gpt-5-mini" "gpt-5.4" "gpt-5.4-mini")
ALL_TASKS=("math" "coding" "science")

# --- Preflight checks ---
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "Error: OPENAI_API_KEY is not set."
    echo "  export OPENAI_API_KEY=\"your-key\""
    exit 1
fi

if [[ ! -f "main.py" ]]; then
    echo "Error: main.py not found. Run from the OckBench root."
    exit 1
fi

# --- Parse args ---
MODELS=()
TASKS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --models)  IFS=' ' read -ra MODELS <<< "$2"; shift 2 ;;
        --tasks)   IFS=' ' read -ra TASKS <<< "$2"; shift 2 ;;
        *)         echo "Unknown arg: $1"; exit 1 ;;
    esac
done

[[ ${#MODELS[@]} -eq 0 ]] && MODELS=("${ALL_MODELS[@]}")
[[ ${#TASKS[@]} -eq 0 ]]  && TASKS=("${ALL_TASKS[@]}")

# Validate tasks
for task in "${TASKS[@]}"; do
    if [[ ! " ${ALL_TASKS[*]} " =~ " ${task} " ]]; then
        echo "Error: unknown task '$task'. Valid: ${ALL_TASKS[*]}"
        exit 1
    fi
done

mkdir -p cache results

# --- Run benchmarks ---
for model in "${MODELS[@]}"; do
    max_tokens="${MODEL_TOKENS[$model]}"

    IFS=' ' read -ra efforts <<< "${MODEL_EFFORTS[$model]}"

    for effort in "${efforts[@]}"; do
        for task in "${TASKS[@]}"; do
            # e.g. cache/math_gpt-5.4_medium.jsonl
            cache_file="cache/${task}_${model}_${effort}.jsonl"

            echo ""
            echo "========================================="
            echo " $model | effort=$effort | task=$task"
            echo "========================================="

            # Reasoning effort is now expressed via request_overrides (the old
            # --reasoning-effort flag was removed). For OpenAI reasoning models we
            # apply the full recipe: set the effort, redirect the budget to
            # max_completion_tokens, and drop max_tokens/temperature (which the
            # reasoning API rejects). Math is scored by a required LLM judge (its
            # key resolves from OPENAI_API_KEY).
            extra_args=()
            if [[ "$effort" != "none" ]]; then
                extra_args+=(
                    --request-set "reasoning_effort=$effort"
                    --request-set 'max_completion_tokens=${max_output_tokens}'
                    --request-unset max_tokens
                    --request-unset temperature
                )
            fi
            if [[ "$task" == "math" ]]; then
                extra_args+=(--judge-model "$JUDGE_MODEL" --judge-base-url "$OPENAI_BASE_URL" \
                             --judge-api-key "$OPENAI_API_KEY")
            fi

            python main.py --provider chat_completion --model "$model" \
                --api-key "$OPENAI_API_KEY" --base-url "$OPENAI_BASE_URL" \
                --task "$task" --max-output-tokens "$max_tokens" \
                --output-dir "results/full_pool/${task}" \
                --concurrency "$CONCURRENCY" --timeout "$TIMEOUT" \
                --cache "$cache_file" \
                ${extra_args[@]+"${extra_args[@]}"}
        done
    done
done

echo ""
echo "========================================="
echo " All done! Result files:"
echo "========================================="
ls -1 results/full_pool/*/OckBench_*_gpt-* 2>/dev/null || echo "(no result files found)"
