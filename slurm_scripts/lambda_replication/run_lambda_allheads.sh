#!/bin/bash
#
# megaDNA — genome-wide predictions for BOTH frozen-embedding probe heads
# (linear probe + 3-layer NN) across all genome-wide CSVs. Fills the missing
# LP head (the canonical pipeline deploys only the NN winner) in ONE embedding
# pass per CSV.
#
# PREREQUISITE: megaDNA's old embedding_analysis_megadna.py saved only the NN.
# The probe-save fix (linear_probe_pretrained.pkl) must be committed, pulled on
# Delta, and the embedding analysis RE-RUN so the LP artifact exists in the
# winner seed dir (winners.json[VARIANT].path) BEFORE this driver runs. A job
# hard-fails if the LP artifact is missing.
#
# Reuses lambda_replication.conf. Submits one lambda_allheads_job.sh per
# (LEN, variant, genome CSV); each job resolves the winner seed dir itself.
#
# Usage (login node, repo pulled; the job self-activates megadna):
#   bash slurm_scripts/lambda_replication/run_lambda_allheads.sh [LEN ...]

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIG="${SCRIPT_DIR}/lambda_replication.conf"
JOB="${SCRIPT_DIR}/lambda_allheads_job.sh"
[ -f "${CONFIG}" ] || { echo "ERROR: missing ${CONFIG}"; exit 1; }
[ -f "${JOB}" ]    || { echo "ERROR: missing ${JOB}"; exit 1; }
# shellcheck disable=SC1090
source "${CONFIG}"

LENS=("$@"); [ "${#LENS[@]}" -gt 0 ] || read -ra LENS <<< "${SEGMENT_LENGTHS:-2k 4k 8k}"

mkdir -p "${OUTPUT_DIR}/logs"
LOGDIR="${OUTPUT_DIR}/logs"
INF_FLAGS=(--account="${SLURM_ACCOUNT}" --partition="${SLURM_PARTITION}" ${SLURM_GPUS} --mem="${INF_MEM}" --time="${INF_TIME}" --cpus-per-task=8)

echo "============================================================"
echo "megaDNA — all-heads (LP + NN) genome-wide"
echo "  OUTPUT_DIR: ${OUTPUT_DIR}   LENGTHS: ${LENS[*]}   VARIANTS: ${VARIANTS}"
echo "============================================================"

NUM=0
for LEN in "${LENS[@]}"; do
    REPL_LEN_DIR="${OUTPUT_DIR}/${LEN}"
    ml_var="MAX_LENGTH_${LEN}"; MAX_LENGTH="${!ml_var:-8200}"
    gw_var="GENOME_WIDE_${LEN}"; GW_PATH="${!gw_var:-}"
    if [ -z "${GW_PATH}" ] || [ ! -d "${GW_PATH}" ]; then
        echo "WARNING: no genome-wide dir for ${LEN} (${GW_PATH:-unset}) — skipping"; continue
    fi
    WINNERS_JSON="${REPL_LEN_DIR}/winners.json"
    if [ ! -f "${WINNERS_JSON}" ]; then
        echo "WARNING: no winners.json for ${LEN} (${WINNERS_JSON}) — skipping"; continue
    fi
    for VARIANT in ${VARIANTS}; do
        # Resolve the winner seed dir + check the LP probe exists before submitting.
        eval "$(python "${SCRIPT_DIR}/print_winner_exports.py" "${WINNERS_JSON}" "${VARIANT}" 2>/dev/null)"
        if [ -z "${WINNER_PATH:-}" ] || [ ! -f "${WINNER_PATH}/linear_probe_pretrained.pkl" ]; then
            echo "WARNING: no LP probe in winner dir for ${LEN}/${VARIANT} (${WINNER_PATH:-unset}) — re-run embedding analysis (probe-save fix) first; skipping"; continue
        fi
        shopt -s nullglob; gw_csvs=("${GW_PATH}"/*.csv); shopt -u nullglob
        [ "${#gw_csvs[@]}" -gt 0 ] || { echo "WARNING: ${GW_PATH} has no *.csv — skipping ${LEN}/${VARIANT}"; continue; }
        echo "--- ${LEN}/${VARIANT} (winner seed ${WINNER_SEED:-?}): ${#gw_csvs[@]} genome CSV(s)  max_length=${MAX_LENGTH} ---"
        for csv in "${gw_csvs[@]}"; do
            stem="$(basename "${csv}" .csv)"; J="gwheads_${LEN}_${VARIANT}_${stem}"
            sbatch --job-name="${J}" \
                --output="${LOGDIR}/${J}_%j.out" --error="${LOGDIR}/${J}_%j.err" \
                "${INF_FLAGS[@]}" \
                --export="ALL,REPO_ROOT=${REPO_ROOT},CONDA_ENV=${CONDA_ENV},CONDA_BASE=${CONDA_BASE},REPL_OUTPUT_DIR=${REPL_LEN_DIR},VARIANT=${VARIANT},MODEL_PATH=${MODEL_PATH},INPUT_CSV=${csv},MAX_LENGTH=${MAX_LENGTH},LAYER=${LAYER},POOLING=${POOLING},BATCH_SIZE=${BATCH_SIZE},THRESHOLD=${INF_THRESHOLD}" \
                "${JOB}"
            NUM=$((NUM+1))
        done
    done
done
echo ""
echo "Submitted ${NUM} all-heads genome-wide jobs. Monitor: squeue -u \$USER"
echo "Output: ${OUTPUT_DIR}/<LEN>/genome_wide_heads/<variant>/{lp,nn}/"
