#!/bin/bash
#SBATCH --job-name=megadna_emb
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64g
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --output=megadna_emb_%j.out
#SBATCH --error=megadna_emb_%j.err

# Biowulf batch script for megaDNA embedding analysis
# Usage: sbatch run_embedding_analysis.sh
#
# Required environment variables:
#   CSV_DIR: Path to directory containing train.csv, dev.csv, test.csv
#   MODEL_PATH: Path to megaDNA model checkpoint (.pt file)

echo "============================================================"
echo "megaDNA Embedding Analysis"
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

# Ignore user site-packages
export PYTHONNOUSERSITE=1

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
