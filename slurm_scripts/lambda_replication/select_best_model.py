#!/usr/bin/env python3
"""
Per-variant, pick the embedding-analysis seed with the highest TEST-set MCC.

megaDNA does not finetune: the "trainable" per-seed unit is the embedding
analysis (a 3-layer NN trained on frozen backbone embeddings). This selects the
best-of-N seed for each variant, where the test MCC of the trained NN is read
from each seed's embedding_analysis_results.json.

Writes <output_dir>/winners.json:
    {
      "megadna": {
        "type": "embedding_nn",
        "seed": 3,
        "test_mcc": 0.85,
        "path": "<absolute path to the seed dir>",
        "all_candidates": [{type, seed, test_mcc}, ...]
      }
    }
The seed dir at "path" contains three_layer_nn_pretrained.pt and
three_layer_nn_pretrained_scaler.pkl, which lambda_inference_job.sh feeds to
inference_megadna.py.

Reads:
  <output_dir>/finetune/<variant>/seed-<N>/embedding_analysis_results.json
      (written DIRECTLY by embedding_analysis_megadna.py — it computes the
       3-layer NN test metrics, prefixes them with "pretrained_", and json.dumps
       to embedding_analysis_results.json. So the trained-NN TEST MCC lives under
       "pretrained_nn_mcc". The other keys below are accepted as fallbacks.)
"""

import argparse
import glob
import json
import os
import sys


# TEST-MCC key candidates in order of preference. embedding_analysis_megadna.py
# emits "nn_mcc" from the trained 3-layer NN, namespaced as "pretrained_nn_mcc".
# (The linear-probe MCC "pretrained_linear_probe_mcc" is a weaker fallback only.)
MCC_KEYS = (
    "pretrained_nn_mcc",
    "nn_mcc",
    "pretrained_linear_probe_mcc",
    "linear_probe_mcc",
)


def _read_mcc(metrics):
    for k in MCC_KEYS:
        if k in metrics and metrics[k] is not None:
            return float(metrics[k]), k
    return None, None


def collect_candidates(variant_dir):
    out = []
    for seed_dir in sorted(glob.glob(os.path.join(variant_dir, "seed-*"))):
        results_path = os.path.join(seed_dir, "embedding_analysis_results.json")
        if not os.path.isfile(results_path):
            print(f"  WARN: missing {results_path}, skipping", file=sys.stderr)
            continue
        with open(results_path) as f:
            metrics = json.load(f)
        mcc, key = _read_mcc(metrics)
        if mcc is None:
            print(f"  WARN: no MCC key {MCC_KEYS} in {results_path}, skipping",
                  file=sys.stderr)
            continue
        seed = int(os.path.basename(seed_dir).split("-")[1])
        out.append({
            "type": "embedding_nn",
            "seed": seed,
            "test_mcc": float(mcc),
            "mcc_key": key,
            "path": os.path.abspath(seed_dir),
        })
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output_dir", required=True,
                        help="Per-length replication output dir (contains finetune/)")
    parser.add_argument("--variants", nargs="+", required=True,
                        help="Variants to select for (e.g. megadna)")
    parser.add_argument("--allow-partial", action="store_true",
                        help="Skip variants with no candidates instead of aborting. "
                             "Useful for in-progress dev runs; do NOT use for the "
                             "reviewer-facing pipeline — a missing variant there means "
                             "a real run failure that should fail loudly.")
    args = parser.parse_args()

    winners = {}
    skipped = []
    for variant in args.variants:
        print(f"\n=== {variant} ===")
        variant_dir = os.path.join(args.output_dir, "finetune", variant)
        candidates = collect_candidates(variant_dir)
        if not candidates:
            if not args.allow_partial:
                print(f"  ERROR: no candidates found for {variant} "
                      f"(missing seed-*/embedding_analysis_results.json). "
                      f"Re-run with --allow-partial to skip and continue.",
                      file=sys.stderr)
                sys.exit(1)
            print(f"  SKIP: no candidates found for {variant}", file=sys.stderr)
            skipped.append(variant)
            continue

        for c in sorted(candidates, key=lambda c: c["test_mcc"], reverse=True):
            print(f"  test_mcc={c['test_mcc']:.4f}  ({c['mcc_key']})  finetune/seed-{c['seed']}")

        winner = max(candidates, key=lambda c: c["test_mcc"])
        winner["all_candidates"] = [
            {k: v for k, v in c.items() if k in ("type", "seed", "test_mcc")}
            for c in candidates
        ]
        winners[variant] = winner
        print(f"  WINNER: seed-{winner['seed']} (test_mcc={winner['test_mcc']:.4f})")

    out_path = os.path.join(args.output_dir, "winners.json")
    with open(out_path, "w") as f:
        json.dump(winners, f, indent=2)
    print(f"\nWrote {out_path}  ({len(winners)} variant(s) with winners"
          f"{'; skipped: ' + ','.join(skipped) if skipped else ''})")

    if not winners:
        print("\nERROR: no variant produced any candidates; nothing to write.",
              file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
