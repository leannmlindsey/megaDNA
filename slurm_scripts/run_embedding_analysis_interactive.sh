#!/bin/bash

# Interactive script for running megaDNA embedding analysis WITHOUT sbatch
# Usage: bash run_embedding_analysis_interactive.sh [wrapper_script.sh]
#
# This script reads configuration from wrapper_run_embedding_analysis.sh (or specify another)
# and runs the job directly on the current node.

# Source the wrapper to get all the environment variables
# Change this path if your wrapper has a different name
WRAPPER_SCRIPT="${1:-wrapper_run_embedding_analysis.sh}"

if [ ! -f "${WRAPPER_SCRIPT}" ]; then
    echo "ERROR: Wrapper script not found: ${WRAPPER_SCRIPT}"
    echo "Usage: bash run_embedding_analysis_interactive.sh [wrapper_script.sh]"
    exit 1
fi

echo "============================================================"
echo "Loading configuration from: ${WRAPPER_SCRIPT}"
echo "============================================================"

# Source the wrapper but just get the exports
source <(grep "^export" "${WRAPPER_SCRIPT}")

# Now run the main script logic

echo ""
echo "megaDNA Embedding Analysis (Interactive Mode)"
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

# Set defaults for optional parameters
BATCH_SIZE=${BATCH_SIZE:-8}
MAX_LENGTH=${MAX_LENGTH:-96000}
POOLING=${POOLING:-mean}
LAYER=${LAYER:-middle}
SEED=${SEED:-42}
NN_EPOCHS=${NN_EPOCHS:-100}
NN_HIDDEN_DIM=${NN_HIDDEN_DIM:-256}
NN_LR=${NN_LR:-0.001}
INCLUDE_RANDOM_BASELINE=${INCLUDE_RANDOM_BASELINE:-false}

# Validate required parameters
if [ -z "${CSV_DIR}" ]; then
    echo "ERROR: CSV_DIR is not set"
    exit 1
fi

if [ -z "${MODEL_PATH}" ]; then
    echo "ERROR: MODEL_PATH is not set"
    exit 1
fi

# Navigate to repo root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${SCRIPT_DIR}/.." || exit
echo "Working directory: $(pwd)"

# Set output directory
OUTPUT_DIR=${OUTPUT_DIR:-./results/embedding_analysis/$(basename ${CSV_DIR})}
mkdir -p "${OUTPUT_DIR}"

echo ""
echo "============================================================"
echo "Configuration:"
echo "============================================================"
echo "  Model: ${MODEL_PATH}"
echo "  CSV dir: ${CSV_DIR}"
echo "  Output dir: ${OUTPUT_DIR}"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Max length: ${MAX_LENGTH}"
echo "  Pooling: ${POOLING}"
echo "  Layer: ${LAYER}"
echo "  Seed: ${SEED}"
echo "  NN epochs: ${NN_EPOCHS}"
echo "  NN hidden dim: ${NN_HIDDEN_DIM}"
echo "  NN learning rate: ${NN_LR}"
echo "  Include random baseline: ${INCLUDE_RANDOM_BASELINE}"
echo "============================================================"
echo ""

# Build random baseline flag
RANDOM_BASELINE_FLAG=""
if [ "${INCLUDE_RANDOM_BASELINE}" == "true" ]; then
    RANDOM_BASELINE_FLAG="--include_random_baseline"
fi

# Run embedding analysis
python embedding_analysis_megadna.py \
    --csv_dir="${CSV_DIR}" \
    --model_path="${MODEL_PATH}" \
    --output_dir="${OUTPUT_DIR}" \
    --batch_size=${BATCH_SIZE} \
    --max_length=${MAX_LENGTH} \
    --pooling="${POOLING}" \
    --layer="${LAYER}" \
    --seed=${SEED} \
    --nn_epochs=${NN_EPOCHS} \
    --nn_hidden_dim=${NN_HIDDEN_DIM} \
    --nn_lr=${NN_LR} \
    ${RANDOM_BASELINE_FLAG}

EXIT_CODE=$?

echo ""
echo "============================================================"
echo "Job completed at: $(date)"
echo "Exit code: ${EXIT_CODE}"
echo "Results saved to: ${OUTPUT_DIR}"
echo "============================================================"

exit ${EXIT_CODE}
