#!/bin/bash
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#
# Inference for the winning seed of VARIANT on one CSV. Reads
# <REPL_OUTPUT_DIR>/winners.json to find the winning seed dir, then runs the
# EXISTING inference_megadna.py entry point with that seed's trained 3-layer NN
# classifier + scaler. No model/experiment code is modified.
#
# Required env:
#   REPO_ROOT
#   REPL_OUTPUT_DIR
#   VARIANT
#   MODEL_PATH         pretrained megaDNA backbone (.pt)
#   INPUT_CSV          path to the CSV to predict on
#   OUTPUT_FILENAME    name for the predictions CSV (e.g. test_predictions.csv)
#   MAX_LENGTH         max sequence length (bp) for this window
# Optional env:
#   LAYER (middle), POOLING (mean), BATCH_SIZE (8), THRESHOLD (0.5),
#   CONDA_ENV (megadna)


echo "=== inference ${VARIANT}  input=${INPUT_CSV}  output=${OUTPUT_FILENAME} ==="
echo "Started at: $(date)  Node: $(hostname)  Job: ${SLURM_JOB_ID:-N/A}"

module load CUDA/12.8
source /data/lindseylm/conda/etc/profile.d/conda.sh
if [ -z "${CUDA_HOME}" ]; then
    NVCC_PATH=$(which nvcc 2>/dev/null)
    if [ -n "${NVCC_PATH}" ]; then
        export CUDA_HOME=$(dirname $(dirname "${NVCC_PATH}"))
    fi
fi
conda activate "${CONDA_ENV:-megadna}"
if [ "${CONDA_DEFAULT_ENV}" != "${CONDA_ENV:-megadna}" ]; then
    echo "ERROR: could not activate conda env '${CONDA_ENV:-megadna}' (active: '${CONDA_DEFAULT_ENV:-none}'). Aborting." >&2
    exit 1
fi
echo "  conda env: ${CONDA_DEFAULT_ENV}   python: $(command -v python)"
export CUDA_VISIBLE_DEVICES=0

if [ -z "${REPO_ROOT:-}" ]; then
    echo "ERROR: REPO_ROOT is not set; the launcher must pass it via --export"; exit 1
fi
cd "${REPO_ROOT}"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

if [ -z "${MODEL_PATH:-}" ]; then
    echo "ERROR: MODEL_PATH is not set"; exit 1
fi

LAYER=${LAYER:-middle}
POOLING=${POOLING:-mean}
BATCH_SIZE=${BATCH_SIZE:-8}
MAX_LENGTH=${MAX_LENGTH:-96000}
THRESHOLD=${THRESHOLD:-0.5}

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
WINNERS_JSON="${REPL_OUTPUT_DIR}/winners.json"
if [ ! -f "${WINNERS_JSON}" ]; then
    echo "ERROR: ${WINNERS_JSON} not found (select_best_model must run first)"; exit 1
fi

# print_winner_exports.py emits shlex-quoted exports (WINNER_PATH, WINNER_SEED,
# WINNER_TYPE) read from winners.json[VARIANT]. WINNER_PATH is the winning seed
# dir; its three_layer_nn_pretrained.pt + _scaler.pkl are the classifier+scaler.
eval "$(python "${SCRIPT_DIR}/print_winner_exports.py" "${WINNERS_JSON}" "${VARIANT}")"

CLASSIFIER_PATH="${WINNER_PATH}/three_layer_nn_pretrained.pt"
SCALER_PATH="${WINNER_PATH}/three_layer_nn_pretrained_scaler.pkl"

echo "  winner seed:   ${WINNER_SEED}"
echo "  winner path:   ${WINNER_PATH}"
echo "  classifier:    ${CLASSIFIER_PATH}"
echo "  scaler:        ${SCALER_PATH}"

if [ ! -f "${CLASSIFIER_PATH}" ]; then
    echo "ERROR: classifier not found: ${CLASSIFIER_PATH}"; exit 1
fi
if [ ! -f "${SCALER_PATH}" ]; then
    echo "ERROR: scaler not found: ${SCALER_PATH}"; exit 1
fi

OUTPUT_DIR="${REPL_OUTPUT_DIR}/inference/${VARIANT}"
mkdir -p "${OUTPUT_DIR}"

# inference_megadna.py writes <output_csv with .csv -> _metrics.json> next to the
# predictions CSV when --save_metrics and labels are present.
python inference_megadna.py \
    --input_csv="${INPUT_CSV}" \
    --model_path="${MODEL_PATH}" \
    --classifier_path="${CLASSIFIER_PATH}" \
    --scaler_path="${SCALER_PATH}" \
    --output_csv="${OUTPUT_DIR}/${OUTPUT_FILENAME}" \
    --layer="${LAYER}" \
    --pooling="${POOLING}" \
    --batch_size=${BATCH_SIZE} \
    --max_length=${MAX_LENGTH} \
    --threshold=${THRESHOLD} \
    --save_metrics

echo "Done: $(date)"
