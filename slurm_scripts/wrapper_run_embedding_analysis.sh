#!/bin/bash

# Wrapper script for running megaDNA embedding analysis on Biowulf
#
# Usage:
#   1. Edit the configuration section below
#   2. Run: bash wrapper_run_embedding_analysis.sh
#
# Or submit directly with environment variables:
#   sbatch --export=ALL,CSV_DIR=/path/to/data,MODEL_PATH=/path/to/model.pt run_embedding_analysis.sh

#####################################################################
# CONFIGURATION - Edit this section
#####################################################################

# === REQUIRED: Dataset Configuration ===
# Path to directory containing train.csv, dev.csv (or val.csv), test.csv
export CSV_DIR="/path/to/your/csv/data"

# === REQUIRED: Model Configuration ===
# Path to megaDNA model checkpoint (.pt file)
# Download from: https://huggingface.co/lingxusb/megaDNA_updated
export MODEL_PATH="/path/to/megaDNA_phage_145M.pt"

# === OPTIONAL: Output Directory ===
# Leave empty to use default: ./results/embedding_analysis/$(basename $CSV_DIR)
export OUTPUT_DIR=""

# === OPTIONAL: Hyperparameters ===
export BATCH_SIZE="8"
export MAX_LENGTH="96000"          # megaDNA supports up to 96kb
export POOLING="mean"              # Options: mean, max, cls
export LAYER="middle"              # Options: local (196d), middle (256d), global (512d), all (concatenated)
export SEED="42"

# === OPTIONAL: 3-Layer NN Parameters ===
export NN_EPOCHS="100"
export NN_HIDDEN_DIM="256"
export NN_LR="0.001"

# === OPTIONAL: Include Random Baseline ===
# Set to "true" to also run analysis on randomly initialized model for comparison
export INCLUDE_RANDOM_BASELINE="false"

#####################################################################
# END CONFIGURATION
#####################################################################

# Validate configuration
if [ "${CSV_DIR}" == "/path/to/your/csv/data" ]; then
    echo "ERROR: Please set CSV_DIR to your actual data directory"
    exit 1
fi

if [ "${MODEL_PATH}" == "/path/to/megaDNA_phage_145M.pt" ]; then
    echo "ERROR: Please set MODEL_PATH to your megaDNA model checkpoint"
    exit 1
fi

# Verify files exist
if [ ! -d "${CSV_DIR}" ]; then
    echo "ERROR: CSV_DIR does not exist: ${CSV_DIR}"
    exit 1
fi

if [ ! -f "${CSV_DIR}/train.csv" ]; then
    echo "ERROR: train.csv not found in ${CSV_DIR}"
    exit 1
fi

if [ ! -f "${CSV_DIR}/test.csv" ]; then
    echo "ERROR: test.csv not found in ${CSV_DIR}"
    exit 1
fi

# Check for dev.csv or val.csv
if [ ! -f "${CSV_DIR}/dev.csv" ] && [ ! -f "${CSV_DIR}/val.csv" ]; then
    echo "ERROR: Neither dev.csv nor val.csv found in ${CSV_DIR}"
    exit 1
fi

if [ ! -f "${MODEL_PATH}" ]; then
    echo "ERROR: Model file not found: ${MODEL_PATH}"
    exit 1
fi

# Get dataset name for job naming
DATASET_NAME=$(basename "${CSV_DIR}")

# Set default output directory if not specified
if [ -z "${OUTPUT_DIR}" ]; then
    export OUTPUT_DIR="./results/embedding_analysis/${DATASET_NAME}"
fi

echo "=========================================="
echo "Submitting megaDNA Embedding Analysis Job"
echo "=========================================="
echo "Dataset: ${DATASET_NAME}"
echo "CSV dir: ${CSV_DIR}"
echo "Model: ${MODEL_PATH}"
echo "Output dir: ${OUTPUT_DIR}"
echo ""
echo "Parameters:"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Max length: ${MAX_LENGTH}"
echo "  Pooling: ${POOLING}"
echo "  Layer: ${LAYER}"
echo "  Seed: ${SEED}"
echo ""
echo "3-Layer NN:"
echo "  Epochs: ${NN_EPOCHS}"
echo "  Hidden dim: ${NN_HIDDEN_DIM}"
echo "  Learning rate: ${NN_LR}"
echo ""
echo "Include random baseline: ${INCLUDE_RANDOM_BASELINE}"
echo "=========================================="

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Submit job
echo "Submitting job..."
sbatch --export=ALL \
    --job-name="megadna_emb_${DATASET_NAME}" \
    "${SCRIPT_DIR}/run_embedding_analysis.sh"

echo ""
echo "Job submitted. Monitor with: squeue -u \$USER"
