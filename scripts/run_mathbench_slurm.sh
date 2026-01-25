#!/bin/bash
#SBATCH --job-name=mathbench
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=500G
#SBATCH --exclusive
#SBATCH --time=72:00:00
#SBATCH --output=logs/mathbench_%j.out
#SBATCH --error=logs/mathbench_%j.err

# =============================================================================
# OckBench MathBench_Combined Evaluation Script
# Runs multiple local models sequentially using vLLM
# =============================================================================

set -e

# Configuration
OCKBENCH_DIR="/home/junxiong/zdu/OckBench"
DATASET_PATH="data/MathBench_Combined.jsonl"
DATASET_NAME="MathBench_Combined"
VLLM_PORT=8000
VLLM_HOST="localhost"
BASE_URL="http://${VLLM_HOST}:${VLLM_PORT}/v1"

# Activate virtual environment
cd "${OCKBENCH_DIR}"
source .venv/bin/activate

# Create logs directory
mkdir -p "${OCKBENCH_DIR}/logs"

# =============================================================================
# Model Definitions
# Format: "model_name|context_window|tensor_parallel|thinking_mode|display_name"
# thinking_mode: "default" (use model default), "true", "false", or "both"
# TP settings: 7B/8B=1, 14B=2, 30B/32B=4
# =============================================================================

declare -a MODELS=(
    # AReaL-boba-2 series (context: 40960) - TP=8
    "inclusionAI/AReaL-boba-2-8B|40960|8|default|AReaL-boba-2-8B"
    "inclusionAI/AReaL-boba-2-14B|40960|8|default|AReaL-boba-2-14B"
    "inclusionAI/AReaL-boba-2-32B|40960|8|default|AReaL-boba-2-32B"

    # AceReason-Nemotron series (context: 131072) - TP=8
    "nvidia/AceReason-Nemotron-7B|131072|8|default|AceReason-Nemotron-7B"
    "nvidia/AceReason-Nemotron-14B|131072|8|default|AceReason-Nemotron-14B"

    # DeepSeek-R1-Distill series (context: 131072) - TP=4 (28 heads not divisible by 8)
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B|131072|4|default|DeepSeek-R1-Distill-Qwen-7B"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B|131072|4|default|DeepSeek-R1-Distill-Qwen-14B"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B|131072|4|default|DeepSeek-R1-Distill-Qwen-32B"

    # Qwen3-2507 series (context: 262144) - TP=8
    "Qwen/Qwen3-4B-Thinking-2507|262144|8|default|Qwen3-4B-Thinking-2507"
    "Qwen/Qwen3-4B-Instruct-2507|262144|8|default|Qwen3-4B-Instruct-2507"
    "Qwen/Qwen3-30B-A3B-Thinking-2507|262144|8|default|Qwen3-30B-A3B-Thinking-2507"
    "Qwen/Qwen3-30B-A3B-Instruct-2507|262144|8|default|Qwen3-30B-A3B-Instruct-2507"

    # Qwen3 base series with thinking toggle (context: 40960) - TP=8
    # Run each model twice: with thinking enabled and disabled
    "Qwen/Qwen3-8B|40960|8|both|Qwen3-8B"
    "Qwen/Qwen3-14B|40960|8|both|Qwen3-14B"
    "Qwen/Qwen3-32B|40960|8|both|Qwen3-32B"
)

# =============================================================================
# Helper Functions
# =============================================================================

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

wait_for_vllm() {
    local max_wait=1800  # 30 minutes (TP=8 with CUDA graphs takes time)
    local waited=0
    local interval=10

    log "Waiting for vLLM server to be ready..."

    while [ $waited -lt $max_wait ]; do
        if curl -s "${BASE_URL}/models" > /dev/null 2>&1; then
            log "vLLM server is ready!"
            return 0
        fi
        sleep $interval
        waited=$((waited + interval))
        log "Still waiting... (${waited}s / ${max_wait}s)"
    done

    log "ERROR: vLLM server failed to start within ${max_wait} seconds"
    return 1
}

kill_vllm() {
    log "Stopping vLLM server..."

    # First try graceful shutdown with SIGTERM
    local pids=$(lsof -t -i:${VLLM_PORT} 2>/dev/null || true)
    if [ -n "$pids" ]; then
        log "Sending SIGTERM to port ${VLLM_PORT} processes..."
        echo "$pids" | xargs -r kill -15 2>/dev/null || true
    fi
    pkill -15 -f "vllm.entrypoints" 2>/dev/null || true

    # Wait for graceful shutdown
    sleep 10

    # Force kill any remaining processes
    pids=$(lsof -t -i:${VLLM_PORT} 2>/dev/null || true)
    if [ -n "$pids" ]; then
        log "Force killing remaining processes..."
        echo "$pids" | xargs -r kill -9 2>/dev/null || true
    fi
    pkill -9 -f "vllm.entrypoints" 2>/dev/null || true
    pkill -9 -f "multiproc_executor" 2>/dev/null || true
    pkill -9 -f "ray::" 2>/dev/null || true

    # Force GPU memory cleanup
    log "Cleaning up GPU memory..."
    python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

    # Wait for GPU memory to be released (important!)
    sleep 15
    log "vLLM server stopped"
}

launch_vllm() {
    local model=$1
    local context_window=$2
    local tp=$3

    log "Launching vLLM with model: ${model}"
    log "  Tensor Parallel: ${tp}"
    log "  Max Model Len: ${context_window}"

    # Launch vLLM in background
    python -m vllm.entrypoints.openai.api_server \
        --model "${model}" \
        --tensor-parallel-size "${tp}" \
        --max-model-len "${context_window}" \
        --port "${VLLM_PORT}" \
        --trust-remote-code \
        --disable-log-requests \
        --gpu-memory-utilization 0.95 \
        2>&1 | tee -a "${OCKBENCH_DIR}/logs/vllm_${model//\//_}.log" &

    VLLM_PID=$!
    log "vLLM launched with PID: ${VLLM_PID}"
}

run_benchmark() {
    local model=$1
    local context_window=$2
    local display_name=$3
    local enable_thinking=$4  # "default", "true", or "false"

    local thinking_arg=""
    local result_suffix=""

    if [ "$enable_thinking" = "true" ]; then
        thinking_arg="--enable-thinking true"
        result_suffix="_Thinking"
    elif [ "$enable_thinking" = "false" ]; then
        thinking_arg="--enable-thinking false"
        result_suffix="_Instruct"
    fi

    local experiment_name="${DATASET_NAME}_${display_name}${result_suffix}"

    log "Running benchmark: ${experiment_name}"
    log "  Model: ${model}"
    log "  Context Window: ${context_window}"
    log "  Enable Thinking: ${enable_thinking}"

    cd "${OCKBENCH_DIR}"

    python main.py \
        --provider generic \
        --model "${model}" \
        --base-url "${BASE_URL}" \
        --dataset-path "${DATASET_PATH}" \
        --dataset-name "${DATASET_NAME}" \
        --evaluator-type math \
        --max-context-window "${context_window}" \
        --concurrency 200 \
        --timeout 3600 \
        --enforce-output-format \
        --experiment-name "${experiment_name}" \
        ${thinking_arg}

    local exit_code=$?

    if [ $exit_code -eq 0 ]; then
        log "Benchmark completed successfully: ${experiment_name}"
    else
        log "WARNING: Benchmark failed with exit code ${exit_code}: ${experiment_name}"
    fi

    return $exit_code
}

# =============================================================================
# Main Execution
# =============================================================================

log "=========================================="
log "Starting MathBench_Combined Evaluation"
log "=========================================="

# Ensure clean state
kill_vllm

cd "${OCKBENCH_DIR}"

# Track results
declare -a COMPLETED=()
declare -a FAILED=()

for model_spec in "${MODELS[@]}"; do
    # Parse model specification
    IFS='|' read -r model context_window tp thinking_mode display_name <<< "$model_spec"

    log ""
    log "=========================================="
    log "Processing: ${display_name}"
    log "=========================================="

    # Launch vLLM
    launch_vllm "$model" "$context_window" "$tp"

    # Wait for server to be ready
    if ! wait_for_vllm; then
        log "ERROR: Failed to start vLLM for ${model}"
        kill_vllm
        FAILED+=("${display_name}")
        continue
    fi

    # Run benchmark based on thinking mode
    if [ "$thinking_mode" = "both" ]; then
        # Run with thinking enabled
        if run_benchmark "$model" "$context_window" "$display_name" "true"; then
            COMPLETED+=("${display_name}_Thinking")
        else
            FAILED+=("${display_name}_Thinking")
        fi

        # Run with thinking disabled
        if run_benchmark "$model" "$context_window" "$display_name" "false"; then
            COMPLETED+=("${display_name}_Instruct")
        else
            FAILED+=("${display_name}_Instruct")
        fi
    else
        # Run with default or specified thinking mode
        if run_benchmark "$model" "$context_window" "$display_name" "$thinking_mode"; then
            COMPLETED+=("${display_name}")
        else
            FAILED+=("${display_name}")
        fi
    fi

    # Stop vLLM
    kill_vllm

    log "Completed processing: ${display_name}"
done

# =============================================================================
# Summary
# =============================================================================

log ""
log "=========================================="
log "Evaluation Complete"
log "=========================================="
log ""
log "Completed (${#COMPLETED[@]}):"
for item in "${COMPLETED[@]}"; do
    log "  - ${item}"
done
log ""
log "Failed (${#FAILED[@]}):"
for item in "${FAILED[@]}"; do
    log "  - ${item}"
done
log ""
log "Results saved to: ${OCKBENCH_DIR}/results/"
