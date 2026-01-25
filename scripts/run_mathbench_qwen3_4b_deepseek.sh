#!/bin/bash
#SBATCH --job-name=mathbench_qwen3_4b_deepseek
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=500G
#SBATCH --exclusive
#SBATCH --time=48:00:00
#SBATCH --output=logs/mathbench_qwen3_4b_deepseek_%j.out
#SBATCH --error=logs/mathbench_qwen3_4b_deepseek_%j.err

# =============================================================================
# OckBench MathBench - Qwen3-4B + DeepSeek Distill Evaluation
# Runs 4 models simultaneously:
#   - Qwen3-4B-Thinking-2507    (TP=1, GPU 0, context 262144)
#   - Qwen3-4B-Instruct-2507    (TP=1, GPU 1, context 262144)
#   - DeepSeek-R1-Distill-Qwen-14B (TP=2, GPU 2-3, context 131072)
#   - DeepSeek-R1-Distill-Qwen-32B (TP=4, GPU 4-7, context 131072)
# Total: 1 + 1 + 2 + 4 = 8 GPUs
# =============================================================================

set -e

# Configuration
OCKBENCH_DIR="/home/junxiong/zdu/OckBench"
DATASET_PATH="data/MathBench_Combined.jsonl"
DATASET_NAME="MathBench_Combined"
VLLM_HOST="localhost"

# Models
MODEL_QWEN_4B_THINKING="Qwen/Qwen3-4B-Thinking-2507"
MODEL_QWEN_4B_INSTRUCT="Qwen/Qwen3-4B-Instruct-2507"
MODEL_DEEPSEEK_14B="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
MODEL_DEEPSEEK_32B="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"

# Port assignments
PORT_QWEN_4B_THINKING=8001
PORT_QWEN_4B_INSTRUCT=8002
PORT_DEEPSEEK_14B=8003
PORT_DEEPSEEK_32B=8004

# GPU assignments
GPUS_QWEN_4B_THINKING="0"
GPUS_QWEN_4B_INSTRUCT="1"
GPUS_DEEPSEEK_14B="2,3"
GPUS_DEEPSEEK_32B="4,5,6,7"

# Context windows (different per model)
CONTEXT_QWEN_4B=262144
CONTEXT_DEEPSEEK=131072

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

    for port in ${PORT_QWEN_4B_THINKING} ${PORT_QWEN_4B_INSTRUCT} ${PORT_DEEPSEEK_14B} ${PORT_DEEPSEEK_32B}; do
        local pids=$(lsof -t -i:${port} 2>/dev/null || true)
        if [ -n "$pids" ]; then
            echo "$pids" | xargs -r kill -15 2>/dev/null || true
        fi
    done

    pkill -15 -f "vllm.entrypoints" 2>/dev/null || true
    sleep 10

    for port in ${PORT_QWEN_4B_THINKING} ${PORT_QWEN_4B_INSTRUCT} ${PORT_DEEPSEEK_14B} ${PORT_DEEPSEEK_32B}; do
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
    local context=$5
    local log_suffix=$6

    log "Launching vLLM: ${model}"
    log "  Port: ${port}, TP: ${tp}, GPUs: ${gpus}, Context: ${context}"

    CUDA_VISIBLE_DEVICES="${gpus}" python -m vllm.entrypoints.openai.api_server \
        --model "${model}" \
        --tensor-parallel-size "${tp}" \
        --max-model-len "${context}" \
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
    local context=$5
    local log_file=$6

    local base_url="http://${VLLM_HOST}:${port}/v1"
    local thinking_arg=""

    if [ "$enable_thinking" = "true" ]; then
        thinking_arg="--enable-thinking true"
    elif [ "$enable_thinking" = "false" ]; then
        thinking_arg="--enable-thinking false"
    fi
    # If enable_thinking is empty, don't pass the flag (for DeepSeek models)

    local experiment_name="${DATASET_NAME}_${display_name}"

    log "Running benchmark: ${experiment_name} (port ${port})"

    python main.py \
        --provider generic \
        --model "${model}" \
        --base-url "${base_url}" \
        --dataset-path "${DATASET_PATH}" \
        --dataset-name "${DATASET_NAME}" \
        --evaluator-type math \
        --max-context-window "${context}" \
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
log "Starting Qwen3-4B + DeepSeek Distill Evaluation"
log "  Qwen3-4B-Thinking    (TP=1, GPU ${GPUS_QWEN_4B_THINKING}, port ${PORT_QWEN_4B_THINKING}, ctx ${CONTEXT_QWEN_4B})"
log "  Qwen3-4B-Instruct    (TP=1, GPU ${GPUS_QWEN_4B_INSTRUCT}, port ${PORT_QWEN_4B_INSTRUCT}, ctx ${CONTEXT_QWEN_4B})"
log "  DeepSeek-Distill-14B (TP=2, GPU ${GPUS_DEEPSEEK_14B}, port ${PORT_DEEPSEEK_14B}, ctx ${CONTEXT_DEEPSEEK})"
log "  DeepSeek-Distill-32B (TP=4, GPU ${GPUS_DEEPSEEK_32B}, port ${PORT_DEEPSEEK_32B}, ctx ${CONTEXT_DEEPSEEK})"
log "=========================================="

# Ensure clean state
kill_all_vllm

# =============================================================================
# Launch all vLLM servers
# =============================================================================

log ""
log "=========================================="
log "Phase 1: Launching vLLM servers"
log "=========================================="

launch_vllm "${MODEL_QWEN_4B_THINKING}" "${PORT_QWEN_4B_THINKING}" 1 "${GPUS_QWEN_4B_THINKING}" "${CONTEXT_QWEN_4B}" "Qwen3-4B-Thinking"
launch_vllm "${MODEL_QWEN_4B_INSTRUCT}" "${PORT_QWEN_4B_INSTRUCT}" 1 "${GPUS_QWEN_4B_INSTRUCT}" "${CONTEXT_QWEN_4B}" "Qwen3-4B-Instruct"
launch_vllm "${MODEL_DEEPSEEK_14B}" "${PORT_DEEPSEEK_14B}" 2 "${GPUS_DEEPSEEK_14B}" "${CONTEXT_DEEPSEEK}" "DeepSeek-Distill-14B"
launch_vllm "${MODEL_DEEPSEEK_32B}" "${PORT_DEEPSEEK_32B}" 4 "${GPUS_DEEPSEEK_32B}" "${CONTEXT_DEEPSEEK}" "DeepSeek-Distill-32B"

# Wait for all servers
log ""
log "Waiting for servers to initialize..."

SERVERS_READY=true

if ! wait_for_vllm "${PORT_QWEN_4B_THINKING}" "Qwen3-4B-Thinking"; then
    log "ERROR: Qwen3-4B-Thinking failed to start"
    SERVERS_READY=false
fi

if ! wait_for_vllm "${PORT_QWEN_4B_INSTRUCT}" "Qwen3-4B-Instruct"; then
    log "ERROR: Qwen3-4B-Instruct failed to start"
    SERVERS_READY=false
fi

if ! wait_for_vllm "${PORT_DEEPSEEK_14B}" "DeepSeek-Distill-14B"; then
    log "ERROR: DeepSeek-Distill-14B failed to start"
    SERVERS_READY=false
fi

if ! wait_for_vllm "${PORT_DEEPSEEK_32B}" "DeepSeek-Distill-32B"; then
    log "ERROR: DeepSeek-Distill-32B failed to start"
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

LOG_QWEN_4B_THINKING="${OCKBENCH_DIR}/logs/benchmark_Qwen3-4B-Thinking_$$.log"
LOG_QWEN_4B_INSTRUCT="${OCKBENCH_DIR}/logs/benchmark_Qwen3-4B-Instruct_$$.log"
LOG_DEEPSEEK_14B="${OCKBENCH_DIR}/logs/benchmark_DeepSeek-Distill-14B_$$.log"
LOG_DEEPSEEK_32B="${OCKBENCH_DIR}/logs/benchmark_DeepSeek-Distill-32B_$$.log"

run_benchmark "${MODEL_QWEN_4B_THINKING}" "${PORT_QWEN_4B_THINKING}" "Qwen3-4B-Thinking-2507" "true" "${CONTEXT_QWEN_4B}" "${LOG_QWEN_4B_THINKING}" &
PID_QWEN_4B_THINKING=$!

run_benchmark "${MODEL_QWEN_4B_INSTRUCT}" "${PORT_QWEN_4B_INSTRUCT}" "Qwen3-4B-Instruct-2507" "false" "${CONTEXT_QWEN_4B}" "${LOG_QWEN_4B_INSTRUCT}" &
PID_QWEN_4B_INSTRUCT=$!

run_benchmark "${MODEL_DEEPSEEK_14B}" "${PORT_DEEPSEEK_14B}" "DeepSeek-R1-Distill-Qwen-14B" "" "${CONTEXT_DEEPSEEK}" "${LOG_DEEPSEEK_14B}" &
PID_DEEPSEEK_14B=$!

run_benchmark "${MODEL_DEEPSEEK_32B}" "${PORT_DEEPSEEK_32B}" "DeepSeek-R1-Distill-Qwen-32B" "" "${CONTEXT_DEEPSEEK}" "${LOG_DEEPSEEK_32B}" &
PID_DEEPSEEK_32B=$!

log "Benchmarks launched:"
log "  Qwen3-4B-Thinking PID: ${PID_QWEN_4B_THINKING}"
log "  Qwen3-4B-Instruct PID: ${PID_QWEN_4B_INSTRUCT}"
log "  DeepSeek-Distill-14B PID: ${PID_DEEPSEEK_14B}"
log "  DeepSeek-Distill-32B PID: ${PID_DEEPSEEK_32B}"

# Wait for completion
log "Waiting for benchmarks to complete..."

EXIT_QWEN_4B_THINKING=0
EXIT_QWEN_4B_INSTRUCT=0
EXIT_DEEPSEEK_14B=0
EXIT_DEEPSEEK_32B=0

wait ${PID_QWEN_4B_THINKING} || EXIT_QWEN_4B_THINKING=$?
log "Qwen3-4B-Thinking completed with exit code: ${EXIT_QWEN_4B_THINKING}"

wait ${PID_QWEN_4B_INSTRUCT} || EXIT_QWEN_4B_INSTRUCT=$?
log "Qwen3-4B-Instruct completed with exit code: ${EXIT_QWEN_4B_INSTRUCT}"

wait ${PID_DEEPSEEK_14B} || EXIT_DEEPSEEK_14B=$?
log "DeepSeek-Distill-14B completed with exit code: ${EXIT_DEEPSEEK_14B}"

wait ${PID_DEEPSEEK_32B} || EXIT_DEEPSEEK_32B=$?
log "DeepSeek-Distill-32B completed with exit code: ${EXIT_DEEPSEEK_32B}"

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
log "Qwen3-4B + DeepSeek Distill Evaluation Complete"
log "=========================================="
log ""
log "Results:"
log "  Qwen3-4B-Thinking:    $([ ${EXIT_QWEN_4B_THINKING} -eq 0 ] && echo 'SUCCESS' || echo 'FAILED')"
log "  Qwen3-4B-Instruct:    $([ ${EXIT_QWEN_4B_INSTRUCT} -eq 0 ] && echo 'SUCCESS' || echo 'FAILED')"
log "  DeepSeek-Distill-14B: $([ ${EXIT_DEEPSEEK_14B} -eq 0 ] && echo 'SUCCESS' || echo 'FAILED')"
log "  DeepSeek-Distill-32B: $([ ${EXIT_DEEPSEEK_32B} -eq 0 ] && echo 'SUCCESS' || echo 'FAILED')"
log ""
log "Results saved to: ${OCKBENCH_DIR}/results/"
