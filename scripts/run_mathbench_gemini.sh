#!/bin/bash
# =============================================================================
# OckBench MathBench - Gemini Models Evaluation
# Runs a series of Gemini models on MathBench_Combined dataset
# =============================================================================

set -e

# Configuration
OCKBENCH_DIR="/home/junxiong/zdu/OckBench"
DATASET_PATH="data/MathBench_Combined.jsonl"
DATASET_NAME="MathBench_Combined"

# =============================================================================
# MODELS TO EVALUATE
# =============================================================================
MODELS=(
    # "gemini-3-pro-preview"    # DONE: 64.32% accuracy
    "gemini-3-flash-preview"
    "gemini-2.5-pro"
    "gemini-2.5-flash"
    # "gemini-2.5-flash-lite"   # DONE: 51.77% accuracy
)

# =============================================================================
# PARAMETERS
# =============================================================================
MAX_OUTPUT_TOKENS=65536
CONCURRENCY=50
TEMPERATURE=0.0
TIMEOUT=300
MAX_RETRIES=3

# =============================================================================
# Helper Functions
# =============================================================================

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

check_api_key() {
    if [ -z "${GEMINI_API_KEY}" ]; then
        log "ERROR: GEMINI_API_KEY environment variable is not set"
        log "Please set it with: export GEMINI_API_KEY='your-api-key'"
        exit 1
    fi
    log "GEMINI_API_KEY is set"
}

run_benchmark() {
    local model=$1
    local display_name=$(echo "${model}" | sed 's/[\/:]/_/g')
    local experiment_name="${DATASET_NAME}_${display_name}"

    log "=========================================="
    log "Running: ${model}"
    log "  Max output tokens: ${MAX_OUTPUT_TOKENS}"
    log "  Concurrency: ${CONCURRENCY}"
    log "=========================================="

    python main.py \
        --provider gemini \
        --model "${model}" \
        --dataset-path "${DATASET_PATH}" \
        --dataset-name "${DATASET_NAME}" \
        --evaluator-type math \
        --max-output-tokens "${MAX_OUTPUT_TOKENS}" \
        --temperature "${TEMPERATURE}" \
        --concurrency "${CONCURRENCY}" \
        --timeout "${TIMEOUT}" \
        --max-retries "${MAX_RETRIES}" \
        --enforce-output-format \
        --experiment-name "${experiment_name}" \
        2>&1 | tee -a "${OCKBENCH_DIR}/logs/gemini_${display_name}.log"

    return ${PIPESTATUS[0]}
}

# =============================================================================
# Main Execution
# =============================================================================

log "=========================================="
log "OckBench MathBench - Gemini Evaluation"
log "=========================================="
log "Dataset: ${DATASET_PATH}"
log "Models: ${MODELS[*]}"
log ""

check_api_key

# Change to OckBench directory and activate virtual environment
cd "${OCKBENCH_DIR}"
source .venv/bin/activate

mkdir -p "${OCKBENCH_DIR}/logs"

SUCCESSFUL=0
FAILED=0

for model in "${MODELS[@]}"; do
    log ""
    if run_benchmark "${model}"; then
        log "SUCCESS: ${model}"
        SUCCESSFUL=$((SUCCESSFUL + 1))
    else
        log "FAILED: ${model}"
        FAILED=$((FAILED + 1))
    fi
    sleep 5
done

log ""
log "=========================================="
log "Complete: ${SUCCESSFUL} succeeded, ${FAILED} failed"
log "Results: ${OCKBENCH_DIR}/results/"
log "=========================================="

[ ${FAILED} -gt 0 ] && exit 1
exit 0
