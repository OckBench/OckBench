#!/bin/bash
#SBATCH --job-name=mathbench_qwen3_30b_a3b
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=500G
#SBATCH --exclusive
#SBATCH --time=48:00:00
#SBATCH --output=logs/mathbench_qwen3_30b_a3b_%j.out
#SBATCH --error=logs/mathbench_qwen3_30b_a3b_%j.err

# =============================================================================
# OckBench MathBench - Qwen3-30B-A3B Evaluation
# Runs Qwen3-30B-A3B-Thinking-2507 and Qwen3-30B-A3B-Instruct-2507 simultaneously
# GPU allocation: Thinking(TP=4, GPU 0-3) + Instruct(TP=4, GPU 4-7) = 8 GPUs
# =============================================================================

set -e

# Configuration
OCKBENCH_DIR="/home/junxiong/zdu/OckBench"
DATASET_PATH="data/MathBench_Combined.jsonl"
DATASET_NAME="MathBench_Combined"
VLLM_HOST="localhost"

# Models
MODEL_THINKING="Qwen/Qwen3-30B-A3B-Thinking-2507"
MODEL_INSTRUCT="Qwen/Qwen3-30B-A3B-Instruct-2507"

# Port assignments
PORT_THINKING=8001
PORT_INSTRUCT=8002

# GPU assignments (TP=4 each)
GPUS_THINKING="0,1,2,3"
GPUS_INSTRUCT="4,5,6,7"

# Context window
CONTEXT_WINDOW=262144

# Activate virtual environment
cd "${OCKBENCH_DIR}"
source .venv/bin/activate

# Create logs directory
mkdir -p "${OCKBENCH_DIR}/logs"

# =============================================================================
# Helper Functions
# =============================================================================

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

wait_for_vllm() {
    local port=$1
    local model_name=$2
    local max_wait=1800
    local waited=0
    local interval=10
    local base_url="http://${VLLM_HOST}:${port}/v1"

    log "Waiting for vLLM server (${model_name}) on port ${port}..."

    while [ $waited -lt $max_wait ]; do
        if curl -s "${base_url}/models" > /dev/null 2>&1; then
            log "vLLM server (${model_name}) is ready on port ${port}!"
            return 0
        fi
        sleep $interval
        waited=$((waited + interval))
        if [ $((waited % 60)) -eq 0 ]; then
            log "Still waiting for ${model_name}... (${waited}s / ${max_wait}s)"
        fi
    done

    log "ERROR: vLLM server (${model_name}) failed to start within ${max_wait} seconds"
    return 1
}

kill_all_vllm() {
    log "Stopping all vLLM servers..."

    for port in ${PORT_THINKING} ${PORT_INSTRUCT}; do
        local pids=$(lsof -t -i:${port} 2>/dev/null || true)
        if [ -n "$pids" ]; then
            echo "$pids" | xargs -r kill -15 2>/dev/null || true
        fi
    done

    pkill -15 -f "vllm.entrypoints" 2>/dev/null || true
    sleep 10

    for port in ${PORT_THINKING} ${PORT_INSTRUCT}; do
        local pids=$(lsof -t -i:${port} 2>/dev/null || true)
        if [ -n "$pids" ]; then
            echo "$pids" | xargs -r kill -9 2>/dev/null || true
        fi
    done

    pkill -9 -f "vllm.entrypoints" 2>/dev/null || true
    pkill -9 -f "multiproc_executor" 2>/dev/null || true

    log "Cleaning up GPU memory..."
    python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
    sleep 10
    log "All vLLM servers stopped"
}

launch_vllm() {
    local model=$1
    local port=$2
    local tp=$3
    local gpus=$4
    local log_suffix=$5

    log "Launching vLLM: ${model}"
    log "  Port: ${port}, TP: ${tp}, GPUs: ${gpus}, Context: ${CONTEXT_WINDOW}"

    CUDA_VISIBLE_DEVICES="${gpus}" python -m vllm.entrypoints.openai.api_server \
        --model "${model}" \
        --tensor-parallel-size "${tp}" \
        --max-model-len "${CONTEXT_WINDOW}" \
        --port "${port}" \
        --trust-remote-code \
        --disable-log-requests \
        --gpu-memory-utilization 0.95 \
        2>&1 | tee -a "${OCKBENCH_DIR}/logs/vllm_${log_suffix}_$$.log" &

    log "vLLM launched for ${model} with PID: $!"
}

run_benchmark() {
    local model=$1
    local port=$2
    local display_name=$3
    local enable_thinking=$4
    local log_file=$5

    local base_url="http://${VLLM_HOST}:${port}/v1"
    local thinking_arg=""

    if [ "$enable_thinking" = "true" ]; then
        thinking_arg="--enable-thinking true"
    else
        thinking_arg="--enable-thinking false"
    fi

    local experiment_name="${DATASET_NAME}_${display_name}"

    log "Running benchmark: ${experiment_name} (port ${port})"

    python main.py \
        --provider generic \
        --model "${model}" \
        --base-url "${base_url}" \
        --dataset-path "${DATASET_PATH}" \
        --dataset-name "${DATASET_NAME}" \
        --evaluator-type math \
        --max-context-window "${CONTEXT_WINDOW}" \
        --concurrency 50 \
        --timeout 7200 \
        --enforce-output-format \
        --experiment-name "${experiment_name}" \
        ${thinking_arg} \
        2>&1 | tee -a "${log_file}"

    return ${PIPESTATUS[0]}
}

# =============================================================================
# Main Execution
# =============================================================================

log "=========================================="
log "Starting Qwen3-30B-A3B Evaluation"
log "  Thinking (TP=4, GPU ${GPUS_THINKING}, port ${PORT_THINKING})"
log "  Instruct (TP=4, GPU ${GPUS_INSTRUCT}, port ${PORT_INSTRUCT})"
log "=========================================="

# Ensure clean state
kill_all_vllm

# =============================================================================
# Launch both vLLM servers
# =============================================================================

log ""
log "=========================================="
log "Phase 1: Launching vLLM servers"
log "=========================================="

launch_vllm "${MODEL_THINKING}" "${PORT_THINKING}" 4 "${GPUS_THINKING}" "Qwen3-30B-A3B-Thinking"
launch_vllm "${MODEL_INSTRUCT}" "${PORT_INSTRUCT}" 4 "${GPUS_INSTRUCT}" "Qwen3-30B-A3B-Instruct"

# Wait for servers
log ""
log "Waiting for servers to initialize..."

SERVERS_READY=true

if ! wait_for_vllm "${PORT_THINKING}" "Qwen3-30B-A3B-Thinking"; then
    log "ERROR: Thinking model failed to start"
    SERVERS_READY=false
fi

if ! wait_for_vllm "${PORT_INSTRUCT}" "Qwen3-30B-A3B-Instruct"; then
    log "ERROR: Instruct model failed to start"
    SERVERS_READY=false
fi

if [ "$SERVERS_READY" = false ]; then
    log "ERROR: Not all servers started. Aborting."
    kill_all_vllm
    exit 1
fi

log ""
log "All vLLM servers are ready!"

# =============================================================================
# Run benchmarks in parallel
# =============================================================================

log ""
log "=========================================="
log "Phase 2: Running benchmarks"
log "=========================================="

LOG_THINKING="${OCKBENCH_DIR}/logs/benchmark_Qwen3-30B-A3B-Thinking_$$.log"
LOG_INSTRUCT="${OCKBENCH_DIR}/logs/benchmark_Qwen3-30B-A3B-Instruct_$$.log"

run_benchmark "${MODEL_THINKING}" "${PORT_THINKING}" "Qwen3-30B-A3B-Thinking-2507" "true" "${LOG_THINKING}" &
PID_THINKING=$!

run_benchmark "${MODEL_INSTRUCT}" "${PORT_INSTRUCT}" "Qwen3-30B-A3B-Instruct-2507" "false" "${LOG_INSTRUCT}" &
PID_INSTRUCT=$!

log "Benchmarks launched:"
log "  Thinking PID: ${PID_THINKING}"
log "  Instruct PID: ${PID_INSTRUCT}"

# Wait for completion
log "Waiting for benchmarks to complete..."

EXIT_THINKING=0
EXIT_INSTRUCT=0

wait ${PID_THINKING} || EXIT_THINKING=$?
log "Thinking completed with exit code: ${EXIT_THINKING}"

wait ${PID_INSTRUCT} || EXIT_INSTRUCT=$?
log "Instruct completed with exit code: ${EXIT_INSTRUCT}"

# =============================================================================
# Cleanup
# =============================================================================

log ""
log "=========================================="
log "Cleaning up"
log "=========================================="

kill_all_vllm

# =============================================================================
# Summary
# =============================================================================

log ""
log "=========================================="
log "Qwen3-30B-A3B Evaluation Complete"
log "=========================================="
log ""
log "Results:"
log "  Thinking: $([ ${EXIT_THINKING} -eq 0 ] && echo 'SUCCESS' || echo 'FAILED')"
log "  Instruct: $([ ${EXIT_INSTRUCT} -eq 0 ] && echo 'SUCCESS' || echo 'FAILED')"
log ""
log "Results saved to: ${OCKBENCH_DIR}/results/"
