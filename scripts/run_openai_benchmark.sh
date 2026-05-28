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

            python main.py --provider chat_completion --model "$model" \
                --task "$task" --max-output-tokens "$max_tokens" \
                --output-dir "results/full_pool/${task}" \
                --reasoning-effort "$effort" \
                --concurrency "$CONCURRENCY" --timeout "$TIMEOUT" \
                --cache "$cache_file"
        done
    done
done

echo ""
echo "========================================="
echo " All done! Result files:"
echo "========================================="
ls -1 results/full_pool/*/OckBench_*_gpt-* 2>/dev/null || echo "(no result files found)"
