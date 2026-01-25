#!/bin/bash
#SBATCH --job-name=mathbench_qwen3
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=500G
#SBATCH --exclusive
#SBATCH --time=48:00:00
#SBATCH --output=logs/mathbench_qwen3_%j.out
#SBATCH --error=logs/mathbench_qwen3_%j.err

# =============================================================================
# OckBench MathBench - Qwen3 Parallel Evaluation
# Runs Qwen3-8B, Qwen3-14B, and Qwen3-32B simultaneously on a single 8-GPU node
# GPU allocation: 8B(TP=1) + 14B(TP=2) + 32B(TP=4) = 7 GPUs
# =============================================================================

set -e

# Configuration
OCKBENCH_DIR="/home/junxiong/zdu/OckBench"
DATASET_PATH="data/MathBench_Combined.jsonl"
DATASET_NAME="MathBench_Combined"
VLLM_HOST="localhost"

# Port assignments for each model
PORT_8B=8001
PORT_14B=8002
PORT_32B=8003

# GPU assignments (0-indexed)
# 8B:  GPU 0        (TP=1, 1 GPU)
# 14B: GPU 1-2      (TP=2, 2 GPUs)
# 32B: GPU 3-6      (TP=4, 4 GPUs)
# GPU 7 is spare
GPUS_8B="0"
GPUS_14B="1,2"
GPUS_32B="3,4,5,6"

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

    # Kill by ports
    for port in ${PORT_8B} ${PORT_14B} ${PORT_32B}; do
        local pids=$(lsof -t -i:${port} 2>/dev/null || true)
        if [ -n "$pids" ]; then
            echo "$pids" | xargs -r kill -15 2>/dev/null || true
        fi
    done

    pkill -15 -f "vllm.entrypoints" 2>/dev/null || true
    sleep 10

    # Force kill
    for port in ${PORT_8B} ${PORT_14B} ${PORT_32B}; do
        local pids=$(lsof -t -i:${port} 2>/dev/null || true)
        if [ -n "$pids" ]; then
            echo "$pids" | xargs -r kill -9 2>/dev/null || true
        fi
    done

    pkill -9 -f "vllm.entrypoints" 2>/dev/null || true
    pkill -9 -f "multiproc_executor" 2>/dev/null || true

    # GPU cleanup
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
    local context_window=$5
    local log_suffix=$6

    log "Launching vLLM: ${model}"
    log "  Port: ${port}, TP: ${tp}, GPUs: ${gpus}, Context: ${context_window}"

    CUDA_VISIBLE_DEVICES="${gpus}" python -m vllm.entrypoints.openai.api_server \
        --model "${model}" \
        --tensor-parallel-size "${tp}" \
        --max-model-len "${context_window}" \
        --port "${port}" \
        --trust-remote-code \
        --disable-log-requests \
        --gpu-memory-utilization 0.95 \
        2>&1 | tee -a "${OCKBENCH_DIR}/logs/vllm_parallel_${log_suffix}.log" &

    log "vLLM launched for ${model} with PID: $!"
}

run_benchmark() {
    local model=$1
    local port=$2
    local context_window=$3
    local display_name=$4
    local enable_thinking=$5
    local log_file=$6

    local base_url="http://${VLLM_HOST}:${port}/v1"
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

    log "Running benchmark: ${experiment_name} (port ${port})"

    python main.py \
        --provider generic \
        --model "${model}" \
        --base-url "${base_url}" \
        --dataset-path "${DATASET_PATH}" \
        --dataset-name "${DATASET_NAME}" \
        --evaluator-type math \
        --max-context-window "${context_window}" \
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
log "Starting Qwen3 Parallel Evaluation"
log "  Qwen3-8B  (TP=1, GPU ${GPUS_8B}, port ${PORT_8B})"
log "  Qwen3-14B (TP=2, GPU ${GPUS_14B}, port ${PORT_14B})"
log "  Qwen3-32B (TP=4, GPU ${GPUS_32B}, port ${PORT_32B})"
log "=========================================="

# Ensure clean state
kill_all_vllm

cd "${OCKBENCH_DIR}"

# =============================================================================
# Launch all vLLM servers
# =============================================================================

log ""
log "=========================================="
log "Phase 1: Launching all vLLM servers"
log "=========================================="

launch_vllm "Qwen/Qwen3-8B" "${PORT_8B}" 1 "${GPUS_8B}" 40960 "Qwen3-8B"
launch_vllm "Qwen/Qwen3-14B" "${PORT_14B}" 2 "${GPUS_14B}" 40960 "Qwen3-14B"
launch_vllm "Qwen/Qwen3-32B" "${PORT_32B}" 4 "${GPUS_32B}" 40960 "Qwen3-32B"

# Wait for all servers to be ready
log ""
log "Waiting for all servers to initialize..."

SERVERS_READY=true

if ! wait_for_vllm "${PORT_8B}" "Qwen3-8B"; then
    log "ERROR: Qwen3-8B failed to start"
    SERVERS_READY=false
fi

if ! wait_for_vllm "${PORT_14B}" "Qwen3-14B"; then
    log "ERROR: Qwen3-14B failed to start"
    SERVERS_READY=false
fi

if ! wait_for_vllm "${PORT_32B}" "Qwen3-32B"; then
    log "ERROR: Qwen3-32B failed to start"
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
# Run benchmarks (each model runs Thinking → Instruct independently)
# =============================================================================

log ""
log "=========================================="
log "Phase 2: Running benchmarks (each model runs Thinking → Instruct independently)"
log "=========================================="

# Helper function to run both modes for a single model
run_model_both_modes() {
    local model=$1
    local port=$2
    local context_window=$3
    local display_name=$4

    local LOG_THINKING="${OCKBENCH_DIR}/logs/benchmark_${display_name}_Thinking_$$.log"
    local LOG_INSTRUCT="${OCKBENCH_DIR}/logs/benchmark_${display_name}_Instruct_$$.log"

    log "${display_name}: Starting Thinking mode"
    if run_benchmark "${model}" "${port}" "${context_window}" "${display_name}" "true" "${LOG_THINKING}"; then
        log "${display_name}: Thinking mode SUCCESS"
    else
        log "${display_name}: Thinking mode FAILED"
    fi

    log "${display_name}: Starting Instruct mode"
    if run_benchmark "${model}" "${port}" "${context_window}" "${display_name}" "false" "${LOG_INSTRUCT}"; then
        log "${display_name}: Instruct mode SUCCESS"
    else
        log "${display_name}: Instruct mode FAILED"
    fi

    log "${display_name}: Both modes completed"
}

# Launch each model's full sequence (Thinking → Instruct) in parallel
run_model_both_modes "Qwen/Qwen3-8B" "${PORT_8B}" 40960 "Qwen3-8B" &
PID_8B=$!

run_model_both_modes "Qwen/Qwen3-14B" "${PORT_14B}" 40960 "Qwen3-14B" &
PID_14B=$!

run_model_both_modes "Qwen/Qwen3-32B" "${PORT_32B}" 40960 "Qwen3-32B" &
PID_32B=$!

log "All model sequences launched:"
log "  Qwen3-8B  (Thinking → Instruct) PID: ${PID_8B}"
log "  Qwen3-14B (Thinking → Instruct) PID: ${PID_14B}"
log "  Qwen3-32B (Thinking → Instruct) PID: ${PID_32B}"

# Wait for all models to complete their full sequences
log "Waiting for all models to complete both modes..."

EXIT_8B=0
EXIT_14B=0
EXIT_32B=0

wait ${PID_8B} || EXIT_8B=$?
log "Qwen3-8B completed with exit code: ${EXIT_8B}"

wait ${PID_14B} || EXIT_14B=$?
log "Qwen3-14B completed with exit code: ${EXIT_14B}"

wait ${PID_32B} || EXIT_32B=$?
log "Qwen3-32B completed with exit code: ${EXIT_32B}"

log ""
log "Final results:"
log "  Qwen3-8B:  $([ ${EXIT_8B} -eq 0 ] && echo 'SUCCESS' || echo 'FAILED')"
log "  Qwen3-14B: $([ ${EXIT_14B} -eq 0 ] && echo 'SUCCESS' || echo 'FAILED')"
log "  Qwen3-32B: $([ ${EXIT_32B} -eq 0 ] && echo 'SUCCESS' || echo 'FAILED')"

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
log "Qwen3 Parallel Evaluation Complete"
log "=========================================="
log ""
log "Results saved to: ${OCKBENCH_DIR}/results/"
log ""
log "Check individual logs:"
log "  ${OCKBENCH_DIR}/logs/vllm_parallel_Qwen3-*.log"
log "  ${OCKBENCH_DIR}/logs/benchmark_Qwen3-*_*.log"
