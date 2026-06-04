#!/bin/bash
#
# megaDNA LAMBDA replication — check that all STAGE 1 (per-seed embedding
# analysis) jobs finished.
#
# Reads the same lambda_replication.conf as the launcher, then for every
# (length, variant, seed) cell reports:
#   RESULTS  embedding_analysis_results.json present (analysis finished)
#   MODEL    three_layer_nn_pretrained.pt present (trained classifier saved)
#   MCC      test-set NN MCC from embedding_analysis_results.json (pretrained_nn_mcc)
#   LOG      whether the matching SLURM .out log ended with "Done:"
# and lists any non-empty .err files (potential failures).
#
# Usage:
#   bash slurm_scripts/lambda_replication/check_training.sh
#
# Run this after run_lambda_training.sh and before run_lambda_inference.sh.

# Absolute path to this lambda_replication dir on Biowulf (hardcoded so it is
# correct no matter what directory the script is launched/submitted from).
SCRIPT_DIR="/vf/users/lindseylm/GLM_EVALUATIONS/NAR_GENOMICS_LAMBDA_REPO/megaDNA/slurm_scripts/lambda_replication"
CONFIG="${SCRIPT_DIR}/lambda_replication.conf"

if [ ! -f "${CONFIG}" ]; then
    echo "ERROR: missing ${CONFIG}"; exit 1
fi
# shellcheck disable=SC1090
source "${CONFIG}"

if [ -z "${OUTPUT_DIR}" ]; then
    echo "ERROR: OUTPUT_DIR is empty (check ${CONFIG})"; exit 1
fi
if [ ! -d "${OUTPUT_DIR}" ]; then
    echo "ERROR: OUTPUT_DIR not found: ${OUTPUT_DIR}"; exit 1
fi

RUN_LENGTHS="$(echo "${SEGMENT_LENGTHS}" | xargs)"
LOGDIR="${OUTPUT_DIR}/logs"

echo "============================================================"
echo "megaDNA LAMBDA replication — Stage 1 check"
echo "============================================================"
echo "  OUTPUT_DIR:      ${OUTPUT_DIR}"
echo "  SEGMENT_LENGTHS: ${RUN_LENGTHS}"
echo "  VARIANTS:        ${VARIANTS}"
echo "  SEEDS:           ${SEEDS}"
echo "============================================================"
echo ""

TOTAL=0
OK=0

printf "%-4s %-9s %-5s  %-8s  %-8s  %-8s  %s\n" LEN VARIANT SEED RESULTS MODEL MCC LOG
for LEN in ${RUN_LENGTHS}; do
    for VARIANT in ${VARIANTS}; do
        for SEED in ${SEEDS}; do
            TOTAL=$((TOTAL + 1))
            D="${OUTPUT_DIR}/${LEN}/finetune/${VARIANT}/seed-${SEED}"

            if [ -f "${D}/embedding_analysis_results.json" ]; then R=ok; else R=MISSING; fi

            if [ -f "${D}/three_layer_nn_pretrained.pt" ]; then M=ok; else M=MISSING; fi

            MCC=$(python - "${D}/embedding_analysis_results.json" 2>/dev/null <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
    v = d.get('pretrained_nn_mcc', d.get('nn_mcc', d.get('pretrained_linear_probe_mcc')))
    print(f"{v:.4f}" if isinstance(v, (int, float)) else "?")
except Exception:
    print("-")
PY
)

            LOG=$(ls -t "${LOGDIR}/ft_${LEN}_${VARIANT}_s${SEED}_"*.out 2>/dev/null | head -1)
            if [ -n "${LOG}" ] && grep -q "^Done:" "${LOG}"; then
                L=done
            elif [ -n "${LOG}" ]; then
                L="NO 'Done:'"
            else
                L="no .out"
            fi

            [ "${R}" = ok ] && [ "${M}" = ok ] && [ "${L}" = done ] && OK=$((OK + 1))

            printf "%-4s %-9s %-5s  %-8s  %-8s  %-8s  %s\n" \
                "${LEN}" "${VARIANT}" "${SEED}" "${R}" "${M}" "${MCC}" "${L}"
        done
    done
done

echo ""
echo "Healthy: ${OK} / ${TOTAL}"

echo ""
echo "=== non-empty .err files (potential failures) ==="
ERRS=$(find "${LOGDIR}" -name "ft_*.err" -size +0c -printf "%s  %p\n" 2>/dev/null | sort -rn)
if [ -n "${ERRS}" ]; then
    echo "${ERRS}"
else
    echo "  (none)"
fi
