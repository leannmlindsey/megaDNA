#!/bin/bash
#
# Batch Inference Submission Script for megaDNA
#
# This script submits multiple SLURM jobs for inference on a list of input files.
#
# Usage:
#   ./submit_batch_inference.sh \
#       --input_list /path/to/input_files.txt \
#       --output_dir /path/to/output_directory \
#       --model_path /path/to/finetuned_model.pt
#
# The input_list file should contain one input CSV path per line.
#

set -e

# Default values
BATCH_SIZE=8
MAX_LENGTH=96000
THRESHOLD=0.5

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --input_list)
            INPUT_LIST="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --max_length)
            MAX_LENGTH="$2"
            shift 2
            ;;
        --threshold)
            THRESHOLD="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Required arguments:"
            echo "  --input_list FILE       Text file with one input CSV path per line"
            echo "  --output_dir DIR        Directory to store all output files"
            echo "  --model_path PATH       Path to fine-tuned megaDNA model checkpoint (.pt file)"
            echo ""
            echo "Optional arguments:"
            echo "  --batch_size N          Batch size for inference (default: 8)"
            echo "  --max_length N          Maximum sequence length (default: 96000)"
            echo "  --threshold F           Classification threshold (default: 0.5)"
            echo "  --help                  Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "${INPUT_LIST}" ]; then
    echo "ERROR: --input_list is required"
    echo "Use --help for usage information"
    exit 1
fi

if [ -z "${OUTPUT_DIR}" ]; then
    echo "ERROR: --output_dir is required"
    echo "Use --help for usage information"
    exit 1
fi

if [ -z "${MODEL_PATH}" ]; then
    echo "ERROR: --model_path is required"
    echo "Use --help for usage information"
    exit 1
fi

# Validate input list file exists
if [ ! -f "${INPUT_LIST}" ]; then
    echo "ERROR: Input list file not found: ${INPUT_LIST}"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "${OUTPUT_DIR}"

# Get the directory of this script (for finding run_inference.sh)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
INFERENCE_SCRIPT="${SCRIPT_DIR}/run_inference.sh"

if [ ! -f "${INFERENCE_SCRIPT}" ]; then
    echo "ERROR: Inference script not found: ${INFERENCE_SCRIPT}"
    exit 1
fi

echo "============================================================"
echo "megaDNA Batch Inference Submission"
echo "============================================================"
echo "Input list:      ${INPUT_LIST}"
echo "Output dir:      ${OUTPUT_DIR}"
echo "Model path:      ${MODEL_PATH}"
echo "Batch size:      ${BATCH_SIZE}"
echo "Max length:      ${MAX_LENGTH}"
echo "Threshold:       ${THRESHOLD}"
echo "============================================================"
echo ""

# Count input files
NUM_FILES=$(grep -c -v '^[[:space:]]*$' "${INPUT_LIST}" || echo 0)
echo "Found ${NUM_FILES} input files to process"
echo ""

# Track submitted jobs
SUBMITTED_JOBS=()

# Read input list and submit jobs
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

    # Generate output filename
    INPUT_BASENAME=$(basename "${INPUT_CSV}" .csv)
    OUTPUT_CSV="${OUTPUT_DIR}/${INPUT_BASENAME}_predictions.csv"

    echo "Submitting job for: ${INPUT_BASENAME}"
    echo "  Input:  ${INPUT_CSV}"
    echo "  Output: ${OUTPUT_CSV}"

    # Submit SLURM job
    JOB_ID=$(sbatch \
        --job-name="megadna_inf_${INPUT_BASENAME}" \
        --output="${OUTPUT_DIR}/slurm_${INPUT_BASENAME}_%j.out" \
        --error="${OUTPUT_DIR}/slurm_${INPUT_BASENAME}_%j.err" \
        --export=ALL,INPUT_CSV="${INPUT_CSV}",OUTPUT_CSV="${OUTPUT_CSV}",MODEL_PATH="${MODEL_PATH}",BATCH_SIZE="${BATCH_SIZE}",MAX_LENGTH="${MAX_LENGTH}",THRESHOLD="${THRESHOLD}" \
        "${INFERENCE_SCRIPT}" | awk '{print $NF}')

    echo "  Job ID: ${JOB_ID}"
    SUBMITTED_JOBS+=("${JOB_ID}")
    echo ""

done < "${INPUT_LIST}"

echo "============================================================"
echo "Submission Complete"
echo "============================================================"
echo "Total jobs submitted: ${#SUBMITTED_JOBS[@]}"
echo "Job IDs: ${SUBMITTED_JOBS[*]}"
echo ""
echo "Monitor jobs with: squeue -u \$USER"
echo "Output directory: ${OUTPUT_DIR}"
echo "============================================================"
