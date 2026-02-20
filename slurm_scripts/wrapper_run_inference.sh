#!/bin/bash

# Wrapper script for running megaDNA inference on Biowulf
#
# Usage:
#   1. Edit the configuration section below
#   2. Run: bash wrapper_run_inference.sh
#
# Or submit directly with environment variables:
#   sbatch --export=ALL,INPUT_CSV=/path/to/test.csv,MODEL_PATH=/path/to/finetuned_model.pt run_inference.sh

#####################################################################
# CONFIGURATION - Edit this section
#####################################################################

# === REQUIRED: Input CSV ===
# Path to CSV file with 'sequence' column (and optionally 'label')
export INPUT_CSV="/path/to/your/test.csv"

# === REQUIRED: Pretrained megaDNA backbone ===
# Path to pretrained megaDNA model checkpoint (.pt file)
export MODEL_PATH="/path/to/megaDNA_model.pt"

# === REQUIRED: Classifier and Scaler ===
# Path to trained 3-layer NN checkpoint (from embedding_analysis_megadna.py)
export CLASSIFIER_PATH="/path/to/three_layer_nn_pretrained.pt"
# Path to saved StandardScaler (from embedding_analysis_megadna.py)
export SCALER_PATH="/path/to/three_layer_nn_pretrained_scaler.pkl"

# === OPTIONAL: Embedding Parameters ===
export LAYER="middle"                  # Embedding layer: local, middle, global, all
export POOLING="mean"                  # Pooling strategy: mean, max, cls

# === OPTIONAL: Output CSV ===
# Leave empty to use default: input_csv with _predictions suffix
export OUTPUT_CSV=""

# === OPTIONAL: Inference Parameters ===
export BATCH_SIZE="8"
export MAX_LENGTH="96000"          # megaDNA supports up to 96kb
export THRESHOLD="0.5"             # Classification threshold for prob_1

#####################################################################
# END CONFIGURATION
#####################################################################

# Validate configuration
if [ "${INPUT_CSV}" == "/path/to/your/test.csv" ]; then
    echo "ERROR: Please set INPUT_CSV to your actual input file"
    exit 1
fi

if [ "${MODEL_PATH}" == "/path/to/megaDNA_model.pt" ]; then
    echo "ERROR: Please set MODEL_PATH to your pretrained megaDNA backbone"
    exit 1
fi

if [ "${CLASSIFIER_PATH}" == "/path/to/three_layer_nn_pretrained.pt" ]; then
    echo "ERROR: Please set CLASSIFIER_PATH to your trained classifier checkpoint"
    exit 1
fi

if [ "${SCALER_PATH}" == "/path/to/three_layer_nn_pretrained_scaler.pkl" ]; then
    echo "ERROR: Please set SCALER_PATH to your saved scaler"
    exit 1
fi

# Verify files exist
if [ ! -f "${INPUT_CSV}" ]; then
    echo "ERROR: Input CSV not found: ${INPUT_CSV}"
    exit 1
fi

if [ ! -f "${MODEL_PATH}" ]; then
    echo "ERROR: Model file not found: ${MODEL_PATH}"
    exit 1
fi

if [ ! -f "${CLASSIFIER_PATH}" ]; then
    echo "ERROR: Classifier file not found: ${CLASSIFIER_PATH}"
    exit 1
fi

if [ ! -f "${SCALER_PATH}" ]; then
    echo "ERROR: Scaler file not found: ${SCALER_PATH}"
    exit 1
fi

# Get input name for job naming
INPUT_NAME=$(basename "${INPUT_CSV}" .csv)

echo "=========================================="
echo "Submitting megaDNA Inference Job"
echo "=========================================="
echo "Input: ${INPUT_CSV}"
echo "Pretrained Model: ${MODEL_PATH}"
echo "Classifier: ${CLASSIFIER_PATH}"
echo "Scaler: ${SCALER_PATH}"
echo "Output: ${OUTPUT_CSV:-<auto>}"
echo ""
echo "Parameters:"
echo "  Layer:      ${LAYER}"
echo "  Pooling:    ${POOLING}"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Max length: ${MAX_LENGTH}"
echo "  Threshold:  ${THRESHOLD}"
echo "=========================================="

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Submit job
echo "Submitting job..."
sbatch --export=ALL \
    --job-name="megadna_inf_${INPUT_NAME}" \
    "${SCRIPT_DIR}/run_inference.sh"

echo ""
echo "Job submitted. Monitor with: squeue -u \$USER"
