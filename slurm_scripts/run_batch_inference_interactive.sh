#!/bin/bash

# Interactive script for running megaDNA batch inference WITHOUT sbatch
# Usage: bash run_batch_inference_interactive.sh [wrapper_script.sh]
#
# This script reads configuration from wrapper_run_batch_inference.sh (or specify another)
# and runs inference directly on the current node (sequentially for each input file).

# Source the wrapper to get all the environment variables
WRAPPER_SCRIPT="${1:-wrapper_run_batch_inference.sh}"

if [ ! -f "${WRAPPER_SCRIPT}" ]; then
    echo "ERROR: Wrapper script not found: ${WRAPPER_SCRIPT}"
    echo "Usage: bash run_batch_inference_interactive.sh [wrapper_script.sh]"
    exit 1
fi

echo "============================================================"
echo "Loading configuration from: ${WRAPPER_SCRIPT}"
echo "============================================================"

# Extract variable assignments from wrapper (lines with = that aren't comments)
source <(grep -E '^[A-Z_]+=' "${WRAPPER_SCRIPT}" | grep -v '^#')

echo ""
echo "megaDNA Batch Inference (Interactive Mode)"
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

# Check GPU availability
echo ""
echo "GPU Information:"
nvidia-smi 2>/dev/null || echo "No GPU detected or nvidia-smi not available"
echo ""
echo "Python environment:"
which python
python --version
echo ""

# Navigate to repo root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${SCRIPT_DIR}/.." || exit
echo "Working directory: $(pwd)"

# Validate configuration
if [ "${INPUT_LIST}" == "/path/to/input_files.txt" ] || [ -z "${INPUT_LIST}" ]; then
    echo "ERROR: INPUT_LIST is not set properly in wrapper"
    exit 1
fi

if [ "${OUTPUT_DIR}" == "/path/to/output_directory" ] || [ -z "${OUTPUT_DIR}" ]; then
    echo "ERROR: OUTPUT_DIR is not set properly in wrapper"
    exit 1
fi

if [ "${MODEL_PATH}" == "/path/to/finetuned_model.pt" ] || [ -z "${MODEL_PATH}" ]; then
    echo "ERROR: MODEL_PATH is not set properly in wrapper"
    exit 1
fi

# Verify input list exists
if [ ! -f "${INPUT_LIST}" ]; then
    echo "ERROR: Input list file not found: ${INPUT_LIST}"
    exit 1
fi

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Set defaults
BATCH_SIZE=${BATCH_SIZE:-8}
MAX_LENGTH=${MAX_LENGTH:-96000}
THRESHOLD=${THRESHOLD:-0.5}

echo ""
echo "============================================================"
echo "Configuration:"
echo "============================================================"
echo "  Input list: ${INPUT_LIST}"
echo "  Output dir: ${OUTPUT_DIR}"
echo "  Fine-tuned Model: ${MODEL_PATH}"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Max length: ${MAX_LENGTH}"
echo "  Threshold: ${THRESHOLD}"
echo "============================================================"
echo ""

# Count input files
NUM_FILES=$(grep -c -v '^[[:space:]]*$' "${INPUT_LIST}" || echo 0)
echo "Found ${NUM_FILES} input files to process"
echo ""

# Process each input file sequentially
COUNT=0
while IFS= read -r INPUT_CSV || [ -n "${INPUT_CSV}" ]; do
    # Skip empty lines and comments
    if [[ -z "${INPUT_CSV}" ]] || [[ "${INPUT_CSV}" =~ ^[[:space:]]*# ]]; then
        continue
    fi

    # Trim whitespace
    INPUT_CSV=$(echo "${INPUT_CSV}" | xargs)

    # Validate input file exists
    if [ ! -f "${INPUT_CSV}" ]; then
        echo "WARNING: Input file not found, skipping: ${INPUT_CSV}"
        continue
    fi

    COUNT=$((COUNT + 1))
    INPUT_BASENAME=$(basename "${INPUT_CSV}" .csv)
    OUTPUT_CSV="${OUTPUT_DIR}/${INPUT_BASENAME}_predictions.csv"

    echo "============================================================"
    echo "Processing file ${COUNT}/${NUM_FILES}: ${INPUT_BASENAME}"
    echo "  Input:  ${INPUT_CSV}"
    echo "  Output: ${OUTPUT_CSV}"
    echo "============================================================"

    python inference_megadna.py \
        --input_csv="${INPUT_CSV}" \
        --model_path="${MODEL_PATH}" \
        --output_csv="${OUTPUT_CSV}" \
        --batch_size=${BATCH_SIZE} \
        --max_length=${MAX_LENGTH} \
        --threshold=${THRESHOLD} \
        --save_metrics

    echo ""

done < "${INPUT_LIST}"

echo "============================================================"
echo "Batch Inference Complete"
echo "============================================================"
echo "Processed ${COUNT} files"
echo "Results saved to: ${OUTPUT_DIR}"
echo "Job completed at: $(date)"
echo "============================================================"
