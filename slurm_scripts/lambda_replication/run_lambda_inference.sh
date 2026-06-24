#!/bin/bash
#
# megaDNA LAMBDA_v1 replication — STAGE 2: pick the best seed per variant and
# submit all inference jobs.
#
# For each segment length in SEGMENT_LENGTHS:
#   1. Run select_best_model.py to pick the per-variant winning seed by TEST-set
#      MCC (key pretrained_nn_mcc) across the embedding-analysis seeds; writes
#      winners.json.
#   2. Submit one diagnostic-inference job per (variant, dataset):
#        - test       train_val_test/<LEN>/test.csv              (Surface A)
#        - fpr        fpr_test/<LEN>/bacteria_segments_<LEN>.csv  (Surface B, auto-derived)
#        - gc_control shuffled_controls/<LEN>/test_shuffled.csv   (Surface B, auto-derived)
#        - fnr        FNR_<LEN> if set and the file exists        (Surface B, optional)
#      Any missing diagnostic is WARNED-and-SKIPPED (defensive; not fatal).
#   3. If GENOME_WIDE_<LEN> is a directory of *.csv, submit one genome-wide
#      inference job per CSV per variant (Surface C). NO genome-level clustering
#      / aggregation job is chained — that is done CENTRALLY by the harvest
#      pipeline (this repo's genome_predictions_megadna.py is NOT used). The
#      canonical genome_wide_<stem>_predictions.csv files land in
#      inference/<variant>/ for the harvest glob.
#
# There is NO separate "embedding analysis" job in Stage 2: megaDNA's trainable
# stage IS the embedding analysis, already run per seed in Stage 1 (each seed's
# embedding_analysis_results.json mirrors the reference's Surface-D outputs).
#
# Re-running is safe: each inference job overwrites its own predictions CSV.
#
# Usage (after run_lambda_training.sh has finished — verify with `squeue`):
#   bash slurm_scripts/lambda_replication/run_lambda_inference.sh


# This lambda_replication dir, resolved from the launcher's own location so it is
# correct no matter where the repo is cloned (login-node launcher; the sbatch job
# bodies get REPO_ROOT via --export, see below).
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
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
if [ -z "${MODEL_PATH}" ]; then
    echo "ERROR: MODEL_PATH is empty (set the pretrained megaDNA .pt in ${CONFIG})"; exit 1
fi
if [ ! -f "${MODEL_PATH}" ]; then
    echo "ERROR: MODEL_PATH not found: ${MODEL_PATH}"; exit 1
fi
if [ -z "${SEGMENT_LENGTHS}" ]; then
    echo "ERROR: SEGMENT_LENGTHS is empty"; exit 1
fi

# Only run lengths that actually have a finetune/ dir (i.e. Stage 1 ran).
RUN_LENGTHS=""
for LEN in ${SEGMENT_LENGTHS}; do
    if [ ! -d "${OUTPUT_DIR}/${LEN}/finetune" ]; then
        echo "WARNING: ${OUTPUT_DIR}/${LEN}/finetune missing — skipping ${LEN}"
        echo "         (run run_lambda_training.sh first and wait for jobs to finish)"
        continue
    fi
    RUN_LENGTHS="${RUN_LENGTHS} ${LEN}"
done
RUN_LENGTHS="$(echo "${RUN_LENGTHS}" | xargs)"
[ -n "${RUN_LENGTHS}" ] || { echo "ERROR: no lengths with completed Stage 1"; exit 1; }

mkdir -p "${OUTPUT_DIR}/logs"
LOGDIR="${OUTPUT_DIR}/logs"

# --- common sbatch flags ------------------------------------------------------

INF_FLAGS=(--account="${SLURM_ACCOUNT}" --partition="${SLURM_PARTITION}" ${SLURM_GPUS} --mem="${INF_MEM}" --time="${INF_TIME}" --cpus-per-task=8)

echo "============================================================"
echo "megaDNA LAMBDA replication — Stage 2: winners + inference"
echo "============================================================"
echo "  LAMBDA_BASE:     ${LAMBDA_BASE}"
echo "  OUTPUT_DIR:      ${OUTPUT_DIR}"
echo "  MODEL_PATH:      ${MODEL_PATH}"
echo "  SEGMENT_LENGTHS: ${RUN_LENGTHS}"
echo "  VARIANTS:        ${VARIANTS}"
echo "============================================================"

NUM_JOBS=0
cd "${REPO_ROOT}"

for LEN in ${RUN_LENGTHS}; do
    echo ""
    echo "--- length: ${LEN} ---"

    REPL_LEN_DIR="${OUTPUT_DIR}/${LEN}"

    ml_var="MAX_LENGTH_${LEN}"; MAX_LENGTH="${!ml_var:-96000}"

    # --- select winners (login-node; reads JSON only) ---
    echo "  selecting best seed per variant..."
    ALLOW_PARTIAL_FLAG=""
    if [ "${ALLOW_PARTIAL_TRAINING:-false}" = "true" ]; then
        ALLOW_PARTIAL_FLAG="--allow-partial"
    fi
    python "${SCRIPT_DIR}/select_best_model.py" \
        --output_dir "${REPL_LEN_DIR}" \
        --variants ${VARIANTS} \
        ${ALLOW_PARTIAL_FLAG}

    # --- assemble diagnostic dataset list (name -> path) ---
    declare -a DIAG_NAMES DIAG_PATHS
    DIAG_NAMES=(test fpr gc_control)
    DIAG_PATHS=(
        "${LAMBDA_BASE}/train_val_test/${LEN}/test.csv"
        "${LAMBDA_BASE}/fpr_test/${LEN}/bacteria_segments_${LEN}.csv"
        "${LAMBDA_BASE}/shuffled_controls/${LEN}/test_shuffled.csv"
    )

    # Optional FNR — indirect lookup on FNR_<LEN>.
    fnr_var="FNR_${LEN}"
    FNR_PATH="${!fnr_var:-}"
    if [ -n "${FNR_PATH}" ]; then
        if [ -f "${FNR_PATH}" ]; then
            DIAG_NAMES+=(fnr)
            DIAG_PATHS+=("${FNR_PATH}")
        else
            echo "  WARNING: ${fnr_var}=${FNR_PATH} not found — skipping fnr for ${LEN}"
        fi
    fi

    # Warn-and-SKIP any built-in diagnostic that is missing (defensive; not fatal).
    declare -a RUN_NAMES RUN_PATHS
    RUN_NAMES=(); RUN_PATHS=()
    for i in "${!DIAG_NAMES[@]}"; do
        if [ -f "${DIAG_PATHS[$i]}" ]; then
            RUN_NAMES+=("${DIAG_NAMES[$i]}")
            RUN_PATHS+=("${DIAG_PATHS[$i]}")
        else
            echo "  WARNING: diagnostic '${DIAG_NAMES[$i]}' missing: ${DIAG_PATHS[$i]} — skipping"
        fi
    done

    # --- assemble genome-wide CSV list (directory of *.csv) ---
    declare -a GW_CSVS=()
    gw_var="GENOME_WIDE_${LEN}"
    GW_PATH="${!gw_var:-}"
    if [ -n "${GW_PATH}" ]; then
        if [ -f "${GW_PATH}" ]; then
            GW_CSVS=("${GW_PATH}")
        elif [ -d "${GW_PATH}" ]; then
            shopt -s nullglob
            for csv in "${GW_PATH}"/*.csv; do
                GW_CSVS+=("${csv}")
            done
            shopt -u nullglob
            [ "${#GW_CSVS[@]}" -eq 0 ] && \
                echo "  WARNING: ${gw_var}=${GW_PATH} has no *.csv — skipping genome-wide for ${LEN}"
        else
            echo "  WARNING: ${gw_var}=${GW_PATH} not a file/dir — skipping genome-wide for ${LEN}"
        fi
        [ "${#GW_CSVS[@]}" -gt 0 ] && echo "  genome-wide CSVs for ${LEN}: ${#GW_CSVS[@]} file(s)"
    fi

    # Which variants actually have winners for this length.
    WINNERS_JSON="${REPL_LEN_DIR}/winners.json"
    HAVE_VARIANTS=$(python -c "import json; print(' '.join(json.load(open('${WINNERS_JSON}')).keys()))")

    for VARIANT in ${VARIANTS}; do
        # Skip prediction surfaces if no winning seed for this variant.
        if [[ " ${HAVE_VARIANTS} " != *" ${VARIANT} "* ]]; then
            echo "    skip ${VARIANT} predictions: no winner (Stage 1 incomplete?)"
            continue
        fi

        INF_ENV="REPO_ROOT=${REPO_ROOT},CONDA_ENV=${CONDA_ENV},MODEL_PATH=${MODEL_PATH},REPL_OUTPUT_DIR=${REPL_LEN_DIR},VARIANT=${VARIANT},LAYER=${LAYER},POOLING=${POOLING},MAX_LENGTH=${MAX_LENGTH},BATCH_SIZE=${BATCH_SIZE},THRESHOLD=${INF_THRESHOLD}"

        # Diagnostic inference (Surfaces A + B)
        for i in "${!RUN_NAMES[@]}"; do
            NAME="${RUN_NAMES[$i]}"
            CSV="${RUN_PATHS[$i]}"
            JOB="inf_${LEN}_${VARIANT}_${NAME}"
            echo "    submitting ${JOB}..."
            sbatch \
                --job-name="${JOB}" \
                --output="${LOGDIR}/${JOB}_%j.out" \
                --error="${LOGDIR}/${JOB}_%j.err" \
                "${INF_FLAGS[@]}" \
                --export="ALL,${INF_ENV},INPUT_CSV=${CSV},OUTPUT_FILENAME=${NAME}_predictions.csv" \
                "${SCRIPT_DIR}/lambda_inference_job.sh"
            NUM_JOBS=$((NUM_JOBS + 1))
        done

        # PHROG annotated set (Surface D) — phrog-annotated phage subset, 2k only
        # (PHROG_<LEN>, only PHROG_2k is defined). Written with the model-prefixed
        # CANONICAL name megaDNA_<stem>_predictions.csv that the central PHROG
        # table script globs for. inference_megadna.py copies all input columns,
        # so phrog_category / phrog_db_category pass straight through.
        phrog_var="PHROG_${LEN}"
        PHROG_PATH="${!phrog_var:-}"
        if [ -n "${PHROG_PATH}" ]; then
            if [ -f "${PHROG_PATH}" ]; then
                stem=$(basename "${PHROG_PATH}" .csv)
                JOB="phroginf_${LEN}_${VARIANT}_${stem}"
                echo "    submitting ${JOB}..."
                sbatch \
                    --job-name="${JOB}" \
                    --output="${LOGDIR}/${JOB}_%j.out" \
                    --error="${LOGDIR}/${JOB}_%j.err" \
                    "${INF_FLAGS[@]}" \
                    --export="ALL,${INF_ENV},INPUT_CSV=${PHROG_PATH},OUTPUT_FILENAME=megaDNA_${stem}_predictions.csv" \
                    "${SCRIPT_DIR}/lambda_inference_job.sh"
                NUM_JOBS=$((NUM_JOBS + 1))
            else
                echo "    WARNING: ${phrog_var}=${PHROG_PATH} not found — skipping PHROG for ${LEN}"
            fi
        fi

        # Genome-wide inference (Surface C) — one job per CSV. No aggregate
        # clustering job: genome-level analysis is centralized in harvest.
        if [ "${#GW_CSVS[@]}" -gt 0 ]; then
            for csv in "${GW_CSVS[@]}"; do
                stem=$(basename "${csv}" .csv)
                JOB="gwinf_${LEN}_${VARIANT}_${stem}"
                echo "    submitting ${JOB}..."
                sbatch \
                    --job-name="${JOB}" \
                    --output="${LOGDIR}/${JOB}_%j.out" \
                    --error="${LOGDIR}/${JOB}_%j.err" \
                    "${INF_FLAGS[@]}" \
                    --export="ALL,${INF_ENV},INPUT_CSV=${csv},OUTPUT_FILENAME=genome_wide_${stem}_predictions.csv" \
                    "${SCRIPT_DIR}/lambda_inference_job.sh"
                NUM_JOBS=$((NUM_JOBS + 1))
            done
        fi
    done

    unset DIAG_NAMES DIAG_PATHS RUN_NAMES RUN_PATHS GW_CSVS
done

echo ""
echo "Submitted ${NUM_JOBS} jobs. Monitor with: squeue -u \$USER"
echo "Results: ${OUTPUT_DIR}/<LEN>/inference/"
