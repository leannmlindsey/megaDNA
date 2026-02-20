#!/bin/bash

# Wrapper script for running batch inference with megaDNA on Biowulf
#
# Usage:
#   1. Edit the configuration section below
#   2. Run: bash wrapper_run_batch_inference.sh

#####################################################################
# CONFIGURATION - Edit this section
#####################################################################

# === REQUIRED: Input Files ===
# Path to text file containing one input CSV path per line
# Example contents of input_files.txt:
#   /path/to/dataset1.csv
#   /path/to/dataset2.csv
#   /path/to/dataset3.csv
INPUT_LIST="/path/to/input_files.txt"

# === REQUIRED: Output Directory ===
# All predictions and SLURM logs will be saved here
OUTPUT_DIR="/path/to/output_directory"

# === REQUIRED: Pretrained megaDNA backbone ===
# Path to pretrained megaDNA model checkpoint (.pt file)
MODEL_PATH="/path/to/megaDNA_model.pt"

# === REQUIRED: Classifier and Scaler ===
# Path to trained 3-layer NN checkpoint (from embedding_analysis_megadna.py)
CLASSIFIER_PATH="/path/to/three_layer_nn_pretrained.pt"
# Path to saved StandardScaler (from embedding_analysis_megadna.py)
SCALER_PATH="/path/to/three_layer_nn_pretrained_scaler.pkl"

# === OPTIONAL: Embedding Parameters ===
LAYER="middle"
POOLING="mean"

# === OPTIONAL: Inference Parameters ===
BATCH_SIZE="8"
MAX_LENGTH="96000"
THRESHOLD="0.5"

#####################################################################
# END CONFIGURATION
#####################################################################

# Validate configuration
if [ "${INPUT_LIST}" == "/path/to/input_files.txt" ]; then
    echo "ERROR: Please set INPUT_LIST to your input files list"
    exit 1
fi

if [ "${OUTPUT_DIR}" == "/path/to/output_directory" ]; then
    echo "ERROR: Please set OUTPUT_DIR to your output directory"
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
if [ ! -f "${INPUT_LIST}" ]; then
    echo "ERROR: Input list file not found: ${INPUT_LIST}"
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

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "=========================================="
echo "Submitting megaDNA Batch Inference Jobs"
echo "=========================================="
echo "Input list: ${INPUT_LIST}"
echo "Output dir: ${OUTPUT_DIR}"
echo ""
echo "Model Configuration:"
echo "  Pretrained Model: ${MODEL_PATH}"
echo "  Classifier: ${CLASSIFIER_PATH}"
echo "  Scaler: ${SCALER_PATH}"
echo ""
echo "Inference Parameters:"
echo "  Layer:      ${LAYER}"
echo "  Pooling:    ${POOLING}"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Max length: ${MAX_LENGTH}"
echo "  Threshold:  ${THRESHOLD}"
echo "=========================================="

# Call the batch submission script
"${SCRIPT_DIR}/submit_batch_inference.sh" \
    --input_list "${INPUT_LIST}" \
    --output_dir "${OUTPUT_DIR}" \
    --model_path "${MODEL_PATH}" \
    --classifier_path "${CLASSIFIER_PATH}" \
    --scaler_path "${SCALER_PATH}" \
    --layer "${LAYER}" \
    --pooling "${POOLING}" \
    --batch_size "${BATCH_SIZE}" \
    --max_length "${MAX_LENGTH}" \
    --threshold "${THRESHOLD}"
