#!/bin/bash
#
# megaDNA LAMBDA_v1 replication — STAGE 1: fire off all per-seed jobs.
#
# megaDNA does NOT finetune. The trainable per-seed unit is the embedding
# analysis (3-layer NN on frozen backbone embeddings). For each segment length
# in SEGMENT_LENGTHS, this submits one embedding-analysis sbatch job per
# (variant, seed). All jobs run in parallel (no --dependency chaining). Once they
# all complete, run run_lambda_inference.sh to pick the best seed and run
# inference using that seed's trained classifier + scaler.
#
# Usage:
#   1. Edit lambda_replication.conf — confirm LAMBDA_BASE, OUTPUT_DIR, MODEL_PATH.
#   2. bash slurm_scripts/lambda_replication/run_lambda_training.sh
#   3. Wait for jobs: squeue -u $USER ; bash .../check_training.sh
#   4. bash slurm_scripts/lambda_replication/run_lambda_inference.sh


SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# REPO_ROOT is the repo root (holds embedding_analysis_megadna.py,
# inference_megadna.py). slurm_scripts/lambda_replication -> ../.. == root.
REPO_ROOT="$( cd "${SCRIPT_DIR}/../.." && pwd )"
CONFIG="${SCRIPT_DIR}/lambda_replication.conf"

if [ ! -f "${CONFIG}" ]; then
    echo "ERROR: missing ${CONFIG}"; exit 1
fi
# shellcheck disable=SC1090
source "${CONFIG}"

# --- validate -----------------------------------------------------------------

if [[ "${LAMBDA_BASE}" == /path/to/* ]] || [[ "${OUTPUT_DIR}" == /path/to/* ]]; then
    echo "ERROR: edit ${CONFIG} — LAMBDA_BASE or OUTPUT_DIR still set to placeholder"
    exit 1
fi
[ -d "${LAMBDA_BASE}/train_val_test" ] || {
    echo "ERROR: ${LAMBDA_BASE}/train_val_test not found (expected LAMBDA_v1 layout)"
    exit 1
}
if [ -z "${MODEL_PATH}" ]; then
    echo "ERROR: MODEL_PATH is empty (set the pretrained megaDNA .pt in ${CONFIG})"; exit 1
fi
if [ ! -f "${MODEL_PATH}" ]; then
    echo "ERROR: MODEL_PATH not found: ${MODEL_PATH}"; exit 1
fi
if [ -z "${SEGMENT_LENGTHS}" ]; then
    echo "ERROR: SEGMENT_LENGTHS is empty"; exit 1
fi

# Validate per-length input dirs exist before submitting anything.
RUN_LENGTHS=""
for LEN in ${SEGMENT_LENGTHS}; do
    LDIR="${LAMBDA_BASE}/train_val_test/${LEN}"
    if [ ! -d "${LDIR}" ]; then
        echo "WARNING: ${LDIR} not found — skipping ${LEN}"; continue
    fi
    [ -f "${LDIR}/train.csv" ] || { echo "ERROR: ${LDIR}/train.csv not found"; exit 1; }
    [ -f "${LDIR}/test.csv" ]  || { echo "ERROR: ${LDIR}/test.csv not found"; exit 1; }
    if [ ! -f "${LDIR}/dev.csv" ] && [ ! -f "${LDIR}/val.csv" ]; then
        echo "ERROR: ${LDIR} must contain dev.csv or val.csv"; exit 1
    fi
    RUN_LENGTHS="${RUN_LENGTHS} ${LEN}"
done
RUN_LENGTHS="$(echo "${RUN_LENGTHS}" | xargs)"
if [ -z "${RUN_LENGTHS}" ]; then
    echo "ERROR: no runnable lengths after validation"; exit 1
fi

mkdir -p "${OUTPUT_DIR}/logs"
LOGDIR="${OUTPUT_DIR}/logs"

# --- summary ------------------------------------------------------------------

echo "============================================================"
echo "megaDNA LAMBDA replication — Stage 1: per-seed embedding analysis"
echo "============================================================"
echo "  LAMBDA_BASE:     ${LAMBDA_BASE}"
echo "  OUTPUT_DIR:      ${OUTPUT_DIR}"
echo "  REPO_ROOT:       ${REPO_ROOT}"
echo "  MODEL_PATH:      ${MODEL_PATH}"
echo "  SEGMENT_LENGTHS: ${RUN_LENGTHS}"
echo "  VARIANTS:        ${VARIANTS}"
echo "  SEEDS:           ${SEEDS}"
echo "  LAYER/POOLING:   ${LAYER}/${POOLING}"
echo "  NN params:       epochs=${NN_EPOCHS} hidden=${NN_HIDDEN_DIM} lr=${NN_LR} random_baseline=${INCLUDE_RANDOM_BASELINE}"
echo "============================================================"

# --- common sbatch flags ------------------------------------------------------

FT_FLAGS=(--partition=gpu --gres=gpu:a100:1 --mem="${FT_MEM}" --time="${FT_TIME}" --cpus-per-task=8)

# REPO_ROOT is propagated to every job so they can cd to the real repo — SLURM
# stages each job script to /var/spool/slurm/... where BASH_SOURCE[0] can't
# recover the original location.
FT_ENV_BASE="REPO_ROOT=${REPO_ROOT},CONDA_ENV=${CONDA_ENV},MODEL_PATH=${MODEL_PATH},LAYER=${LAYER},POOLING=${POOLING},NN_EPOCHS=${NN_EPOCHS},NN_HIDDEN_DIM=${NN_HIDDEN_DIM},NN_LR=${NN_LR},BATCH_SIZE=${BATCH_SIZE},INCLUDE_RANDOM_BASELINE=${INCLUDE_RANDOM_BASELINE:-false}"

NUM_JOBS=0

for LEN in ${RUN_LENGTHS}; do
    LAMBDA_DIR="${LAMBDA_BASE}/train_val_test/${LEN}"
    REPL_LEN_DIR="${OUTPUT_DIR}/${LEN}"
    mkdir -p "${REPL_LEN_DIR}"

    # Resolve per-window max sequence length (bp).
    ml_var="MAX_LENGTH_${LEN}";  MAX_LENGTH="${!ml_var:-96000}"

    echo ""
    echo "--- length: ${LEN} (max_length=${MAX_LENGTH}) ---"
    echo "    lambda dir:   ${LAMBDA_DIR}"
    echo "    output dir:   ${REPL_LEN_DIR}"

    for VARIANT in ${VARIANTS}; do
        for SEED in ${SEEDS}; do
            JOB="ft_${LEN}_${VARIANT}_s${SEED}"
            echo "    submitting ${JOB}..."
            sbatch \
                --job-name="${JOB}" \
                --output="${LOGDIR}/${JOB}_%j.out" \
                --error="${LOGDIR}/${JOB}_%j.err" \
                "${FT_FLAGS[@]}" \
                --export="ALL,REPL_OUTPUT_DIR=${REPL_LEN_DIR},LAMBDA_DIR=${LAMBDA_DIR},${FT_ENV_BASE},VARIANT=${VARIANT},SEED=${SEED},LEN=${LEN},MAX_LENGTH=${MAX_LENGTH}" \
                "${SCRIPT_DIR}/lambda_finetune_job.sh"
            NUM_JOBS=$((NUM_JOBS + 1))
        done
    done
done

echo ""
echo "Submitted ${NUM_JOBS} jobs. Monitor with: squeue -u \$USER"
echo "When all jobs are done, run:"
echo "  bash ${SCRIPT_DIR}/run_lambda_inference.sh"
