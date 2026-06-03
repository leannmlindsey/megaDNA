#!/bin/bash
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#
# Stage 1 of megaDNA LAMBDA replication: the per-seed "trainable" stage.
#
# megaDNA does NOT finetune — it is a frozen generative backbone. The trainable
# unit is the embedding analysis itself: extract embeddings from the pretrained
# backbone, then train a linear probe + a 3-layer NN. This job runs ONE
# (variant, seed) by calling the EXISTING embedding_analysis_megadna.py entry
# point with EXPLICIT output paths. It does NOT modify any model/experiment code.
#
# It writes per seed (read downstream by select_best_model.py / inference):
#   <OUTPUT_DIR>/embedding_analysis_results.json   (test MCC under pretrained_nn_mcc)
#   <OUTPUT_DIR>/three_layer_nn_pretrained.pt       (winning seed's classifier)
#   <OUTPUT_DIR>/three_layer_nn_pretrained_scaler.pkl
#
# Required env:
#   REPO_ROOT          repo root (holds embedding_analysis_megadna.py)
#   REPL_OUTPUT_DIR    per-length replication output dir (outputs/<LEN>)
#   LAMBDA_DIR         train/val/test CSV directory (LAMBDA_v1 train_val_test/<LEN>)
#   VARIANT            megadna
#   SEED               integer
#   MODEL_PATH         pretrained megaDNA backbone (.pt)
#   MAX_LENGTH         max sequence length (bp) for this window
# Optional env (with defaults):
#   LAYER, POOLING, NN_EPOCHS, NN_HIDDEN_DIM, NN_LR, BATCH_SIZE,
#   INCLUDE_RANDOM_BASELINE, CONDA_ENV (megadna)


echo "=== embedding-analysis (train stage) ${VARIANT} seed=${SEED} len=${LEN:-?} ==="
echo "Started at: $(date)  Node: $(hostname)  Job: ${SLURM_JOB_ID:-N/A}"

# Activate conda (bare style — no set -e; see conda-activate gotcha). The existing
# megaDNA scripts intentionally do NOT set PYTHONNOUSERSITE — follow that.
module load conda
module load CUDA/12.8
if [ -z "${CUDA_HOME}" ]; then
    NVCC_PATH=$(which nvcc 2>/dev/null)
    if [ -n "${NVCC_PATH}" ]; then
        export CUDA_HOME=$(dirname $(dirname "${NVCC_PATH}"))
    fi
fi
source activate "${CONDA_ENV:-megadna}"
echo "  conda env: ${CONDA_DEFAULT_ENV:-<none>}   python: $(command -v python || echo none)"
export CUDA_VISIBLE_DEVICES=0

if [ -z "${REPO_ROOT:-}" ]; then
    echo "ERROR: REPO_ROOT is not set; the launcher must pass it via --export"; exit 1
fi
cd "${REPO_ROOT}"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

if [ -z "${MODEL_PATH:-}" ]; then
    echo "ERROR: MODEL_PATH is not set"; exit 1
fi
if [ ! -f "${MODEL_PATH}" ]; then
    echo "ERROR: MODEL_PATH not found: ${MODEL_PATH}"; exit 1
fi

LAYER=${LAYER:-middle}
POOLING=${POOLING:-mean}
NN_EPOCHS=${NN_EPOCHS:-100}
NN_HIDDEN_DIM=${NN_HIDDEN_DIM:-256}
NN_LR=${NN_LR:-0.001}
BATCH_SIZE=${BATCH_SIZE:-8}
MAX_LENGTH=${MAX_LENGTH:-96000}
INCLUDE_RANDOM_BASELINE=${INCLUDE_RANDOM_BASELINE:-false}

# embedding_analysis_megadna.py reads {train,test}.csv and dev.csv OR val.csv
# directly from --csv_dir (it already falls back to val.csv), so no staging is
# needed — point it straight at the LAMBDA_v1 window directory.
CSV_DIR="${LAMBDA_DIR}"
if [ ! -d "${CSV_DIR}" ]; then
    echo "ERROR: CSV_DIR not found: ${CSV_DIR}"; exit 1
fi

OUTPUT_DIR="${REPL_OUTPUT_DIR}/finetune/${VARIANT}/seed-${SEED}"
mkdir -p "${OUTPUT_DIR}"

echo "  backbone:     ${MODEL_PATH}"
echo "  csv dir:      ${CSV_DIR}"
echo "  output:       ${OUTPUT_DIR}"
echo "  layer=${LAYER}  pooling=${POOLING}  max_length=${MAX_LENGTH}  seed=${SEED}"
echo "  nn_epochs=${NN_EPOCHS}  nn_hidden_dim=${NN_HIDDEN_DIM}  nn_lr=${NN_LR}  random_baseline=${INCLUDE_RANDOM_BASELINE}"

RANDOM_BASELINE_FLAG=""
if [ "${INCLUDE_RANDOM_BASELINE}" == "true" ]; then
    RANDOM_BASELINE_FLAG="--include_random_baseline"
fi

# Writes embedding_analysis_results.json + three_layer_nn_pretrained.pt +
# three_layer_nn_pretrained_scaler.pkl to OUTPUT_DIR.
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

if [ -f "${OUTPUT_DIR}/embedding_analysis_results.json" ]; then
    echo "  wrote ${OUTPUT_DIR}/embedding_analysis_results.json"
else
    echo "  WARNING: ${OUTPUT_DIR}/embedding_analysis_results.json not found — analysis may have failed"
fi

echo "Done: $(date)"
