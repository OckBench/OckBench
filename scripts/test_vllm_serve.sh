#!/bin/bash
#SBATCH --job-name=vllm_test
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=500G
#SBATCH --exclusive
#SBATCH --nodelist=research-secure-19
#SBATCH --time=4:00:00
#SBATCH --output=logs/vllm_test_%j.out
#SBATCH --error=logs/vllm_test_%j.err

# =============================================================================
# Test vLLM serving with TP=4 for all models (no benchmarks)
# =============================================================================

set -e

OCKBENCH_DIR="/home/junxiong/zdu/OckBench"
VLLM_PORT=8000
BASE_URL="http://localhost:${VLLM_PORT}/v1"

cd "${OCKBENCH_DIR}"
source .venv/bin/activate
mkdir -p "${OCKBENCH_DIR}/logs"

# =============================================================================
# All models - TP=8 for non-DeepSeek, TP=4 for DeepSeek (28 heads not divisible by 8)
# Format: "model_name|context_window|tensor_parallel|display_name"
# =============================================================================

declare -a MODELS=(
    # AReaL-boba-2 series - TP=8
    "inclusionAI/AReaL-boba-2-8B|40960|8|AReaL-boba-2-8B"
    "inclusionAI/AReaL-boba-2-14B|40960|8|AReaL-boba-2-14B"
    "inclusionAI/AReaL-boba-2-32B|40960|8|AReaL-boba-2-32B"

    # AceReason-Nemotron series - TP=8
    "nvidia/AceReason-Nemotron-7B|131072|8|AceReason-Nemotron-7B"
    "nvidia/AceReason-Nemotron-14B|131072|8|AceReason-Nemotron-14B"

    # DeepSeek-R1-Distill series - TP=4 (28 heads not divisible by 8)
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B|131072|4|DeepSeek-R1-Distill-Qwen-7B"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B|131072|4|DeepSeek-R1-Distill-Qwen-14B"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B|131072|4|DeepSeek-R1-Distill-Qwen-32B"

    # Qwen3-2507 series - TP=8
    "Qwen/Qwen3-4B-Thinking-2507|262144|8|Qwen3-4B-Thinking-2507"
    "Qwen/Qwen3-4B-Instruct-2507|262144|8|Qwen3-4B-Instruct-2507"
    "Qwen/Qwen3-30B-A3B-Thinking-2507|262144|8|Qwen3-30B-A3B-Thinking-2507"
    "Qwen/Qwen3-30B-A3B-Instruct-2507|262144|8|Qwen3-30B-A3B-Instruct-2507"

    # Qwen3 base series - TP=8
    "Qwen/Qwen3-8B|40960|8|Qwen3-8B"
    "Qwen/Qwen3-14B|40960|8|Qwen3-14B"
    "Qwen/Qwen3-32B|40960|8|Qwen3-32B"
)

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

wait_for_vllm() {
    local max_wait=600
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

    log "Launching vLLM: ${model} (TP=${tp}, ctx=${context_window})"

    python -m vllm.entrypoints.openai.api_server \
        --model "${model}" \
        --tensor-parallel-size "${tp}" \
        --max-model-len "${context_window}" \
        --port "${VLLM_PORT}" \
        --trust-remote-code \
        --disable-log-requests \
        --gpu-memory-utilization 0.95 \
        2>&1 | tee -a "${OCKBENCH_DIR}/logs/vllm_serve_${model//\//_}.log" &

    VLLM_PID=$!
    log "vLLM launched with PID: ${VLLM_PID}"
}

# =============================================================================
# Main
# =============================================================================

log "=========================================="
log "Testing vLLM serving (TP=8 for non-DeepSeek, TP=4 for DeepSeek)"
log "=========================================="

kill_vllm

declare -a SUCCESS=()
declare -a FAILED=()

for model_spec in "${MODELS[@]}"; do
    IFS='|' read -r model context_window tp display_name <<< "$model_spec"

    log ""
    log "=========================================="
    log "Testing: ${display_name} (TP=${tp})"
    log "=========================================="

    launch_vllm "$model" "$context_window" "$tp"

    if wait_for_vllm; then
        log "SUCCESS: ${display_name} started successfully"
        SUCCESS+=("${display_name}")
    else
        log "FAILED: ${display_name} failed to start"
        FAILED+=("${display_name}")
    fi

    kill_vllm
done

# =============================================================================
# Summary
# =============================================================================

log ""
log "=========================================="
log "Test Complete"
log "=========================================="
log ""
log "SUCCESS (${#SUCCESS[@]}):"
for item in "${SUCCESS[@]}"; do
    log "  ✓ ${item}"
done
log ""
log "FAILED (${#FAILED[@]}):"
for item in "${FAILED[@]}"; do
    log "  ✗ ${item}"
done
