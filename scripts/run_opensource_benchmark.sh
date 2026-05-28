#!/usr/bin/env bash
set -euo pipefail

# OckBench Open-Source Model Benchmark Runner (SGLang)
#
# Usage:
#   ./scripts/run_opensource_benchmark.sh                    # run all models, all tasks
#   ./scripts/run_opensource_benchmark.sh --tasks math       # all models, math only
#   ./scripts/run_opensource_benchmark.sh --models "Qwen/Qwen3.5-4B Qwen/Qwen3.5-9B"  # specific models
#   ./scripts/run_opensource_benchmark.sh --tp 4             # override tensor parallelism
#
# Environment:
#   SGLANG_PORT    — server port (default: 8000)
#   SGLANG_EXTRA   — extra args passed to sglang (e.g. "--mem-fraction 0.9")

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

# --- Defaults ---
PORT="${SGLANG_PORT:-8000}"
BASE_URL="http://localhost:${PORT}/v1"
CONCURRENCY=20
TIMEOUT=3000
MAX_RETRIES=2
TEMPERATURE=0.0

ALL_MODELS=(
    "moonshotai/Kimi-K2.5"
    "zai-org/GLM-5"
    "MiniMaxAI/MiniMax-M2.5"
    "Qwen/Qwen3.5-397B-A17B"
    "Qwen/Qwen3.5-122B-A10B"
    "Qwen/Qwen3.5-35B-A3B"
    "Qwen/Qwen3.5-27B"
    "Qwen/Qwen3.5-9B"
    "Qwen/Qwen3.5-4B"
)

ALL_TASKS=("math" "coding" "science")

# Model -> tensor parallelism mapping (adjust based on your GPU setup)
get_tp() {
    case "$1" in
        *K2.5|*GLM-5|*M2.5|*397B*) echo 8 ;;
        *122B*|*27B*)                echo 4 ;;
        *35B-A3B*|*9B*)              echo 2 ;;
        *4B*)                        echo 1 ;;
        *)                           echo 4 ;;
    esac
}

# Model -> max context window mapping
get_context_window() {
    case "$1" in
        *Kimi-K2.5*)      echo 262144 ;;  # 256K
        *GLM-5*)          echo 202752 ;;  # ~200K
        *MiniMax-M2.5*)   echo 196608 ;;  # ~192K
        *Qwen3.5*)        echo 262144 ;;  # 256K
        *)                echo 131072 ;;  # 128K fallback
    esac
}

# --- Parse args ---
MODELS=()
TASKS=()
TP_OVERRIDE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --models)  IFS=' ' read -ra MODELS <<< "$2"; shift 2 ;;
        --tasks)   IFS=' ' read -ra TASKS <<< "$2"; shift 2 ;;
        --tp)      TP_OVERRIDE="$2"; shift 2 ;;
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

if [[ ! -f "main.py" ]]; then
    echo "Error: main.py not found. Run from the OckBench root."
    exit 1
fi

mkdir -p cache results

# --- Helper: wait for server to be ready ---
wait_for_server() {
    local url="$1"
    local max_wait=600  # 10 minutes
    local elapsed=0
    echo "Waiting for server at ${url} ..."
    while ! curl -s "${url}/models" > /dev/null 2>&1; do
        sleep 5
        elapsed=$((elapsed + 5))
        if [[ $elapsed -ge $max_wait ]]; then
            echo "Error: server did not start within ${max_wait}s"
            return 1
        fi
    done
    echo "Server is ready (took ${elapsed}s)"
}

# --- Helper: kill server ---
kill_server() {
    local pid="$1"
    if kill -0 "$pid" 2>/dev/null; then
        echo "Stopping server (PID $pid)..."
        kill "$pid"
        wait "$pid" 2>/dev/null || true
        echo "Server stopped."
    fi
}

# --- Main loop ---
for model in "${MODELS[@]}"; do
    # Short name for display and cache files (e.g. "Qwen3.5-4B")
    short="${model##*/}"

    tp="${TP_OVERRIDE:-$(get_tp "$model")}"
    ctx_window="$(get_context_window "$model")"

    echo ""
    echo "========================================="
    echo " Model: $model (TP=$tp, CTX=$ctx_window)"
    echo "========================================="

    # 1. Launch SGLang server
    echo "Starting SGLang server..."
    python -m sglang.launch_server \
        --model-path "$model" \
        --port "$PORT" \
        --tp "$tp" \
        ${SGLANG_EXTRA:-} \
        > "logs/sglang_${short}.log" 2>&1 &
    SERVER_PID=$!

    mkdir -p logs

    # Wait for server
    if ! wait_for_server "$BASE_URL"; then
        echo "Failed to start server for $model, skipping."
        kill_server "$SERVER_PID"
        continue
    fi

    # 2. Run benchmarks
    for task in "${TASKS[@]}"; do
        echo ""
        echo "--- $short | task=$task ---"

        cache_file="cache/${task}_${short}.jsonl"

        python main.py --provider chat_completion --model "$model" \
            --base-url "$BASE_URL" \
            --api-key dummy \
            --task "$task" \
            --output-dir "results/full_pool/${task}" \
            --max-context-window "$ctx_window" \
            --temperature "$TEMPERATURE" \
            --concurrency "$CONCURRENCY" \
            --timeout "$TIMEOUT" \
            --max-retries "$MAX_RETRIES" \
            --cache "$cache_file" \
            2>&1 | tee -a "logs/bench_${short}_${task}.log"
    done

    # 3. Stop server
    kill_server "$SERVER_PID"

done

echo ""
echo "========================================="
echo " All done! Result files:"
echo "========================================="
ls -1 results/full_pool/*/OckBench_* results/selected/*/OckBench_* 2>/dev/null || echo "(no result files found)"
