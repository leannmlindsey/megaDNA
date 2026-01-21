#!/bin/bash

# Interactive script for running megaDNA inference WITHOUT sbatch
# Usage: bash run_inference_interactive.sh [wrapper_script.sh]
#
# This script reads configuration from wrapper_run_inference.sh (or specify another)
# and runs the job directly on the current node.

# Source the wrapper to get all the environment variables
WRAPPER_SCRIPT="${1:-wrapper_run_inference.sh}"

if [ ! -f "${WRAPPER_SCRIPT}" ]; then
    echo "ERROR: Wrapper script not found: ${WRAPPER_SCRIPT}"
    echo "Usage: bash run_inference_interactive.sh [wrapper_script.sh]"
    exit 1
fi

echo "============================================================"
echo "Loading configuration from: ${WRAPPER_SCRIPT}"
echo "============================================================"

# Source the wrapper but just get the exports
source <(grep "^export" "${WRAPPER_SCRIPT}")

# Now run the main script logic

echo ""
echo "megaDNA Inference (Interactive Mode)"
echo "============================================================"
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo ""

# Load modules (comment out if not on Biowulf/HPC)
module load conda 2>/dev/null || true
module load CUDA/12.8 2>/dev/null || true

# Set CUDA_HOME if not set
if [ -z "${CUDA_HOME}" ]; then
    NVCC_PATH=$(which nvcc 2>/dev/null)
    if [ -n "${NVCC_PATH}" ]; then
        export CUDA_HOME=$(dirname $(dirname "${NVCC_PATH}"))
    fi
fi

# Activate conda environment
source activate megadna

# Note: Removed PYTHONNOUSERSITE=1 to allow ~/.local packages

# Check GPU availability
echo ""
echo "GPU Information:"
nvidia-smi 2>/dev/null || echo "No GPU detected or nvidia-smi not available"
echo ""
echo "Python environment:"
which python
python --version
echo ""

# Set defaults
BATCH_SIZE=${BATCH_SIZE:-8}
MAX_LENGTH=${MAX_LENGTH:-96000}
POOLING=${POOLING:-mean}
LAYER=${LAYER:-middle}
THRESHOLD=${THRESHOLD:-0.5}

# Validate required parameters
if [ -z "${INPUT_CSV}" ]; then
    echo "ERROR: INPUT_CSV is not set"
    exit 1
fi

if [ -z "${MODEL_PATH}" ]; then
    echo "ERROR: MODEL_PATH is not set"
    exit 1
fi

if [ -z "${CLASSIFIER_PATH}" ]; then
    echo "ERROR: CLASSIFIER_PATH is not set"
    exit 1
fi

# Navigate to repo root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${SCRIPT_DIR}/.." || exit
echo "Working directory: $(pwd)"

# Set output path
if [ -z "${OUTPUT_CSV}" ]; then
    OUTPUT_CSV="${INPUT_CSV%.csv}_predictions.csv"
fi

echo ""
echo "============================================================"
echo "Configuration:"
echo "============================================================"
echo "  megaDNA Model: ${MODEL_PATH}"
echo "  Classifier: ${CLASSIFIER_PATH}"
echo "  Input CSV: ${INPUT_CSV}"
echo "  Output CSV: ${OUTPUT_CSV}"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Max length: ${MAX_LENGTH}"
echo "  Pooling: ${POOLING}"
echo "  Layer: ${LAYER}"
echo "  Threshold: ${THRESHOLD}"
echo "============================================================"
echo ""

# Run inference
python inference_megadna.py \
    --input_csv="${INPUT_CSV}" \
    --model_path="${MODEL_PATH}" \
    --classifier_path="${CLASSIFIER_PATH}" \
    --output_csv="${OUTPUT_CSV}" \
    --batch_size=${BATCH_SIZE} \
    --max_length=${MAX_LENGTH} \
    --pooling="${POOLING}" \
    --layer="${LAYER}" \
    --threshold=${THRESHOLD} \
    --save_metrics

EXIT_CODE=$?

echo ""
echo "============================================================"
echo "Job completed at: $(date)"
echo "Exit code: ${EXIT_CODE}"
echo "Predictions saved to: ${OUTPUT_CSV}"
echo "============================================================"

exit ${EXIT_CODE}
