#!/bin/bash
#
# megaDNA LAMBDA replication — check that all STAGE 2 inference jobs finished.
#
# Reads the same lambda_replication.conf as the launcher, then for every
# (length, variant) reports the outputs run_lambda_inference.sh should produce:
#   WINNER     winners.json lists this variant (best seed picked)
#   <diag>     inference/<variant>/<diag>_predictions.csv (+ _predictions_metrics.json),
#              for test / fpr / gc_control / fnr (fnr only if FNR_<LEN> set+exists),
#              with accuracy & mcc straight from the metrics JSON
#   GENOME     genome_wide_*_predictions.csv count vs CSVs in GENOME_WIDE_<LEN>
# then lists any non-empty .err files from inference jobs.
#
# (There is no EMBED row: megaDNA's embedding analysis is the Stage 1 per-seed
# job, checked by check_training.sh.)
#
# Usage:
#   bash slurm_scripts/lambda_replication/check_inference.sh
#
# Run this after run_lambda_inference.sh and squeue shows the jobs done.

# This lambda_replication dir, resolved from this script's own location so it is
# correct no matter where the repo is cloned.
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
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

# Print "accuracy / mcc" from an inference metrics JSON, or a dash if absent.
metrics_line() {
    python - "$1" 2>/dev/null <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
    acc = d.get("accuracy"); mcc = d.get("mcc")
    fa = f"{acc:.4f}" if isinstance(acc, (int, float)) else "?"
    fm = f"{mcc:.4f}" if isinstance(mcc, (int, float)) else "?"
    print(f"acc={fa} mcc={fm}")
except Exception:
    print("-")
PY
}

echo "============================================================"
echo "megaDNA LAMBDA replication — inference check"
echo "============================================================"
echo "  OUTPUT_DIR:      ${OUTPUT_DIR}"
echo "  SEGMENT_LENGTHS: ${RUN_LENGTHS}"
echo "  VARIANTS:        ${VARIANTS}"
echo "============================================================"

for LEN in ${RUN_LENGTHS}; do
    REPL_LEN_DIR="${OUTPUT_DIR}/${LEN}"
    WINNERS_JSON="${REPL_LEN_DIR}/winners.json"

    # diagnostics expected for this length: include fnr only if FNR_<LEN> set+exists.
    DIAGS="test fpr gc_control"
    fnr_var="FNR_${LEN}"
    if [ -n "${!fnr_var:-}" ] && [ -f "${!fnr_var}" ]; then
        DIAGS="${DIAGS} fnr"
    fi

    # genome-wide: how many CSVs did we point at?
    gw_var="GENOME_WIDE_${LEN}"
    GW_PATH="${!gw_var:-}"
    GW_EXPECTED=0
    if [ -n "${GW_PATH}" ]; then
        if [ -f "${GW_PATH}" ]; then
            GW_EXPECTED=1
        elif [ -d "${GW_PATH}" ]; then
            shopt -s nullglob
            gw_files=("${GW_PATH}"/*.csv)
            shopt -u nullglob
            GW_EXPECTED="${#gw_files[@]}"
        fi
    fi

    echo ""
    echo "######## length: ${LEN} ########"
    if [ -f "${WINNERS_JSON}" ]; then
        HAVE_VARIANTS=$(python -c "import json;print(' '.join(json.load(open('${WINNERS_JSON}')).keys()))" 2>/dev/null)
    else
        HAVE_VARIANTS=""
        echo "  WARNING: winners.json MISSING — run_lambda_inference.sh may not have run"
    fi

    for VARIANT in ${VARIANTS}; do
        echo ""
        echo "  --- variant: ${VARIANT} ---"
        INF_DIR="${REPL_LEN_DIR}/inference/${VARIANT}"

        # winner?
        if [[ " ${HAVE_VARIANTS} " == *" ${VARIANT} "* ]]; then
            echo "    WINNER   ok"
        else
            echo "    WINNER   MISSING (no winning seed — predictions skipped)"
        fi

        # diagnostics
        for NAME in ${DIAGS}; do
            CSV="${INF_DIR}/${NAME}_predictions.csv"
            MJSON="${INF_DIR}/${NAME}_predictions_metrics.json"
            if [ -f "${CSV}" ]; then
                if [ -f "${MJSON}" ]; then
                    printf "    %-10s ok   %s\n" "${NAME}" "$(metrics_line "${MJSON}")"
                else
                    printf "    %-10s ok   (no _metrics.json — labels absent?)\n" "${NAME}"
                fi
            else
                printf "    %-10s MISSING\n" "${NAME}"
            fi
        done

        # PHROG annotated set (2k only) — canonical model-prefixed name; confirm
        # the phrog_db_category column survived inference (the table needs it).
        phrog_var="PHROG_${LEN}"
        PHROG_PATH="${!phrog_var:-}"
        if [ -n "${PHROG_PATH}" ] && [ -f "${PHROG_PATH}" ]; then
            stem=$(basename "${PHROG_PATH}" .csv)
            PCSV="${INF_DIR}/megaDNA_${stem}_predictions.csv"
            PJSON="${INF_DIR}/megaDNA_${stem}_predictions_metrics.json"
            if [ -f "${PCSV}" ]; then
                if head -1 "${PCSV}" | grep -q "phrog_db_category"; then
                    HASCAT="phrog_db_category ok"
                else
                    HASCAT="phrog_db_category MISSING"
                fi
                if [ -f "${PJSON}" ]; then
                    printf "    %-10s ok   %s  [%s]\n" "phrog" "$(metrics_line "${PJSON}")" "${HASCAT}"
                else
                    printf "    %-10s ok   (no _metrics.json)  [%s]\n" "phrog" "${HASCAT}"
                fi
            else
                printf "    %-10s MISSING (%s)\n" "phrog" "megaDNA_${stem}_predictions.csv"
            fi
        fi

        # genome-wide
        if [ "${GW_EXPECTED}" -gt 0 ]; then
            shopt -s nullglob
            gw_pred=("${INF_DIR}"/genome_wide_*_predictions.csv)
            shopt -u nullglob
            GW_GOT="${#gw_pred[@]}"
            if [ "${GW_GOT}" -eq "${GW_EXPECTED}" ]; then
                GWS=ok
            else
                GWS="INCOMPLETE"
            fi
            printf "    %-10s %s  predictions=%s/%s\n" \
                "genome" "${GWS}" "${GW_GOT}" "${GW_EXPECTED}"
        fi
    done
done

echo ""
echo "=== non-empty .err files (potential failures) ==="
ERRS=$(find "${LOGDIR}" \( -name "inf_*.err" -o -name "gwinf_*.err" -o -name "phroginf_*.err" \) -size +0c -printf "%s  %p\n" 2>/dev/null | sort -rn)
if [ -n "${ERRS}" ]; then
    echo "${ERRS}"
else
    echo "  (none)"
fi
