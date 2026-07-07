#!/bin/bash
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#
# All-heads genome-wide (linear probe + 3-layer NN, ONE embedding pass) for ONE
# genome-wide CSV — megaDNA. megaDNA is an embedding-probe model (no fine-tuning);
# the canonical pipeline deploys only the NN winner genome-wide. This adds the
# LINEAR probe so both heads have genome-wide coverage.
#
# The probe artifacts live in the WINNER seed dir (winners.json[VARIANT].path),
# resolved via print_winner_exports.py exactly as lambda_inference_job.sh does.
# After the embedding_analysis_megadna.py probe-save fix + an embedding re-run,
# that dir also holds linear_probe_pretrained.pkl.
#
# Writes, under REPL_OUTPUT_DIR/genome_wide_heads/<variant>/:
#   lp/genome_wide_<stem>_predictions.csv (+ _metrics.json)
#   nn/genome_wide_<stem>_predictions.csv (+ _metrics.json)
#
# Required env: REPO_ROOT, REPL_OUTPUT_DIR, VARIANT, MODEL_PATH, INPUT_CSV, MAX_LENGTH
# Optional env: LAYER(middle), POOLING(mean), BATCH_SIZE(8), THRESHOLD(0.5),
#               CONDA_ENV(megadna), CONDA_BASE

echo "=== all-heads genome-wide  variant=${VARIANT}  input=${INPUT_CSV} ==="
echo "Started: $(date)  Node: $(hostname)  Job: ${SLURM_JOB_ID:-N/A}"

module load cuda 2>/dev/null || true
source "${CONDA_BASE:-/u/llindsey1/miniconda3}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-megadna}"
echo "  conda env: ${CONDA_DEFAULT_ENV:-none}   python: $(command -v python)"

export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

if [ -z "${REPO_ROOT:-}" ]; then echo "ERROR: REPO_ROOT not set"; exit 1; fi
if [ ! -f "${INPUT_CSV:-/nonexistent}" ]; then echo "ERROR: INPUT_CSV not found: ${INPUT_CSV:-<unset>}"; exit 1; fi
cd "${REPO_ROOT}"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

# Resolve the winner seed dir (holds the probe artifacts).
SCRIPT_DIR="$(dirname "$(find "${REPO_ROOT}" -path '*lambda_replication/print_winner_exports.py' 2>/dev/null | head -1)")"
WINNERS_JSON="${REPL_OUTPUT_DIR}/winners.json"
if [ ! -f "${WINNERS_JSON}" ]; then echo "ERROR: ${WINNERS_JSON} not found"; exit 1; fi
eval "$(python "${SCRIPT_DIR}/print_winner_exports.py" "${WINNERS_JSON}" "${VARIANT}")"
EMB_DIR="${WINNER_PATH}"

for f in linear_probe_pretrained.pkl three_layer_nn_pretrained.pt three_layer_nn_pretrained_scaler.pkl; do
    if [ ! -f "${EMB_DIR}/${f}" ]; then
        echo "ERROR: missing probe artifact ${EMB_DIR}/${f}"
        echo "       re-run embedding_analysis_megadna.py (probe-save fix) for the winner seed first"
        exit 1
    fi
done

OUT_DIR="${REPL_OUTPUT_DIR}/genome_wide_heads/${VARIANT}"
GW_DIR="$(dirname "${INPUT_CSV}")"
STEM="$(basename "${INPUT_CSV}" .csv)"

echo "  winner seed:  ${WINNER_SEED:-?}   embedding_dir: ${EMB_DIR}"
python genome_wide_all_heads_megadna.py \
    --model_path "${MODEL_PATH}" \
    --embedding_dir "${EMB_DIR}" \
    --input_dir "${GW_DIR}" --pattern "${STEM}.csv" \
    --output_dir "${OUT_DIR}" \
    --layer "${LAYER:-middle}" \
    --pooling "${POOLING:-mean}" \
    --max_length "${MAX_LENGTH:-8200}" \
    --batch_size "${BATCH_SIZE:-8}" \
    --threshold "${THRESHOLD:-0.5}" \
    --save_metrics

echo "Done: $(date)"
