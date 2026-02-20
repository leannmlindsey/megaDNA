#!/bin/bash

# Multi-seed wrapper for megaDNA embedding analysis on Biowulf
#
# Submits one SLURM job per seed (1-10), each writing to:
#   ${BASE_OUTPUT_DIR}/seed_${SEED}/
#
# Usage:
#   1. Edit the configuration section below
#   2. Run: bash wrapper_run_embedding_analysis_multiseed.sh

#####################################################################
# CONFIGURATION - Edit this section
#####################################################################

# === REQUIRED: Data CSVs ===
# Directory containing train.csv, dev.csv/val.csv, test.csv
export CSV_DIR="/home/lindseylm/lindseylm/lambda_final/merged_datasets_filtered/2k"

# === REQUIRED: Pretrained megaDNA Model ===
export MODEL_PATH="/data/lindseylm/GLM_EVALUATIONS/MODELS/MEGADNA/megaDNA/megaDNA_phage_145M.pt"

# === REQUIRED: Base Output Directory ===
# Each seed will get a subdirectory: seed_1/, seed_2/, ... seed_10/
BASE_OUTPUT_DIR="./output/embedding_analysis/lambda_filtered_2k"

# === Seeds to run ===
SEEDS=(1 2 3 4 5 6 7 8 9 10)

# === OPTIONAL: Embedding Analysis Parameters ===
export BATCH_SIZE="8"
export MAX_LENGTH="96000"
export POOLING="mean"
export LAYER="middle"
export NN_EPOCHS="100"
export NN_HIDDEN_DIM="256"
export NN_LR="0.001"
export INCLUDE_RANDOM_BASELINE="false"

#####################################################################
# END CONFIGURATION
#####################################################################

# Validate configuration
if [ ! -d "${CSV_DIR}" ]; then
    echo "ERROR: CSV_DIR not found: ${CSV_DIR}"
    exit 1
fi

if [ ! -f "${MODEL_PATH}" ]; then
    echo "ERROR: MODEL_PATH not found: ${MODEL_PATH}"
    exit 1
fi

# Create base output directory
mkdir -p "${BASE_OUTPUT_DIR}"

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "=========================================="
echo "Submitting megaDNA Embedding Analysis Jobs"
echo "=========================================="
echo "CSV dir:    ${CSV_DIR}"
echo "Model:      ${MODEL_PATH}"
echo "Output dir: ${BASE_OUTPUT_DIR}"
echo "Seeds:      ${SEEDS[*]}"
echo ""
echo "Parameters:"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Max length: ${MAX_LENGTH}"
echo "  Pooling:    ${POOLING}"
echo "  Layer:      ${LAYER}"
echo "  NN epochs:  ${NN_EPOCHS}"
echo "  NN hidden:  ${NN_HIDDEN_DIM}"
echo "  NN lr:      ${NN_LR}"
echo "=========================================="
echo ""

# Track submitted jobs
SUBMITTED_JOBS=()

for SEED in "${SEEDS[@]}"; do
    export SEED
    export OUTPUT_DIR="${BASE_OUTPUT_DIR}/seed_${SEED}"

    echo "Submitting seed ${SEED} -> ${OUTPUT_DIR}"

    JOB_ID=$(sbatch \
        --job-name="megadna_emb_s${SEED}" \
        --output="${BASE_OUTPUT_DIR}/slurm_seed_${SEED}_%j.out" \
        --error="${BASE_OUTPUT_DIR}/slurm_seed_${SEED}_%j.err" \
        --export=ALL \
        "${SCRIPT_DIR}/run_embedding_analysis.sh" | awk '{print $NF}')

    echo "  Job ID: ${JOB_ID}"
    SUBMITTED_JOBS+=("${JOB_ID}")
done

echo ""
echo "=========================================="
echo "Submission Complete"
echo "=========================================="
echo "Total jobs submitted: ${#SUBMITTED_JOBS[@]}"
echo "Job IDs: ${SUBMITTED_JOBS[*]}"
echo ""
echo "Monitor jobs with: squeue -u \$USER"
echo "Output directory: ${BASE_OUTPUT_DIR}"
echo ""
echo "After all jobs complete, aggregate results with:"
echo "  python aggregate_results.py \\"
echo "      --results_dir ${BASE_OUTPUT_DIR} \\"
echo "      --pattern 'seed_*' --mode embedding_analysis"
echo "=========================================="
