#!/bin/bash
#
# megaDNA LAMBDA replication — verify the RANDOM-EMBEDDING BASELINE was produced
# by Stage 1 (run_lambda_training.sh with INCLUDE_RANDOM_BASELINE=true).
#
# For every (length, variant, seed) cell it reports, from
#   <OUTPUT_DIR>/<LEN>/finetune/<variant>/seed-<N>/:
#   NPZ      embeddings_random.npz present (random embeddings were extracted)
#   FLAG     include_random_baseline in embedding_analysis_results.json
#   rand_LP  random linear-probe test MCC   (random_linear_probe_mcc)
#   rand_NN  random 3-layer NN test MCC     (random_nn_mcc)
#   pre_NN   pretrained NN test MCC         (pretrained_nn_mcc, for reference)
#   power_NN embedding power = pretrained - random (embedding_power_nn_mcc)
# A cell is OK when the npz exists AND both random metrics are present.
#
# Reads the same lambda_replication.conf as the launcher. Safe to run anytime
# after Stage 1 (e.g. while inference is running).
#
# Usage:
#   bash slurm_scripts/lambda_replication/check_random_baseline.sh

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

echo "============================================================"
echo "megaDNA LAMBDA replication — random-baseline check"
echo "============================================================"
echo "  OUTPUT_DIR: ${OUTPUT_DIR}"
echo "============================================================"

python - "${OUTPUT_DIR}" <<'PY'
import os, json, glob, sys
root = sys.argv[1]
rows = []
for jf in sorted(glob.glob(f"{root}/*/finetune/*/seed-*/embedding_analysis_results.json")):
    d = os.path.dirname(jf)
    parts = d.split("/")
    length, variant, seed = parts[-4], parts[-2], parts[-1]
    try:
        j = json.load(open(jf))
    except Exception as e:
        rows.append((length, variant, seed, "BADJSON", "-", "-", None, None, None, None))
        continue
    has_npz = os.path.exists(os.path.join(d, "embeddings_random.npz"))
    flag = j.get("include_random_baseline")
    r_lp = j.get("random_linear_probe_mcc")
    r_nn = j.get("random_nn_mcc")
    p_nn = j.get("pretrained_nn_mcc")
    pw   = j.get("embedding_power_nn_mcc")
    ok = has_npz and (r_lp is not None) and (r_nn is not None)
    rows.append((length, variant, seed, "OK" if ok else "MISSING",
                 str(flag), str(has_npz), r_lp, r_nn, p_nn, pw))

def f(x):
    return f"{x:.4f}" if isinstance(x, (int, float)) else "-"

hdr = f"{'LEN':4} {'VARIANT':9} {'SEED':7} {'STATUS':8} {'flag':6} {'npz':6} {'rand_LP':8} {'rand_NN':8} {'pre_NN':8} {'power_NN':8}"
print(hdr)
print("-" * len(hdr))
for length, variant, seed, status, flag, npz, r_lp, r_nn, p_nn, pw in rows:
    print(f"{length:4} {variant:9} {seed:7} {status:8} {flag:6} {npz:6} {f(r_lp):8} {f(r_nn):8} {f(p_nn):8} {f(pw):8}")

n_ok = sum(1 for r in rows if r[3] == "OK")
print()
print(f"Cells with random baseline complete: {n_ok} / {len(rows)}")
PY
