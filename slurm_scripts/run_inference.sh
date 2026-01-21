#!/bin/bash
#SBATCH --job-name=megadna_inf
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32g
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00
#SBATCH --output=megadna_inf_%j.out
#SBATCH --error=megadna_inf_%j.err

# Biowulf batch script for megaDNA inference
# Usage: sbatch run_inference.sh
#
# Required environment variables:
#   INPUT_CSV: Path to CSV file with 'sequence' column
#   MODEL_PATH: Path to megaDNA model checkpoint (.pt file)
#   CLASSIFIER_PATH: Path to trained classifier (.pt or .pkl)

echo "============================================================"
echo "megaDNA Inference"
echo "============================================================"
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# Load modules
module load conda
module load CUDA/12.8

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

# Check GPU
echo ""
echo "GPU Information:"
nvidia-smi
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
