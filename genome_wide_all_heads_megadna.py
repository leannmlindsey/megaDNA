#!/usr/bin/env python3
"""
Collect GENOME-WIDE predictions for BOTH probe heads (linear probe + 3-layer NN)
in a single embedding pass — megaDNA, for the fragment-vs-genome-wide transfer
analysis (best-head in the main table, full per-head in the appendix).

megaDNA is an embedding-probe model (no fine-tuning): the canonical pipeline
deploys only the NN head (three_layer_nn_pretrained.pt) genome-wide via
inference_megadna.py. This script additionally applies the LINEAR probe
(linear_probe_pretrained.pkl, produced by the embedding_analysis_megadna.py
probe-save fix) so both heads have genome-wide coverage.

For each genome-wide CSV: extract megaDNA embeddings ONCE (backbone, layer,
pooling must match what trained the probes), then apply the saved linear probe
AND 3-layer NN from --embedding_dir. Writes:
    <output_dir>/lp/genome_wide_<stem>_predictions.csv   (+ _metrics.json)
    <output_dir>/nn/genome_wide_<stem>_predictions.csv   (+ _metrics.json)

--embedding_dir is the dir that holds the saved probe artifacts (for the
lambda_replication layout that is the WINNER seed dir, whose
three_layer_nn_pretrained.pt was deployed genome-wide; after the probe-save fix
+ an embedding re-run it also holds linear_probe_pretrained.pkl).

Usage (run from the repo root, so `import inference_megadna` resolves):
    python genome_wide_all_heads_megadna.py \
        --model_path <megaDNA_phage_145M.pt> \
        --embedding_dir <winner_seed_dir> \
        --input_dir <LAMBDA_BASE>/genome_wide/<w> \
        --output_dir <OUTPUT_DIR>/<w>/genome_wide_heads/megadna \
        --layer middle --pooling mean --max_length 8200 --save_metrics
"""

import argparse
import glob
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inference_megadna import ThreeLayerNN, extract_embeddings

LP_FILE = "linear_probe_pretrained.pkl"
NN_FILE = "three_layer_nn_pretrained.pt"
NN_SCALER_FILE = "three_layer_nn_pretrained_scaler.pkl"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_path", required=True, help="pretrained megaDNA backbone (.pt)")
    p.add_argument("--embedding_dir", required=True,
                   help="dir with the saved probe artifacts (LP pkl, NN pt + scaler)")
    p.add_argument("--input_dir", required=True, help="dir of genome-wide segment CSVs")
    p.add_argument("--output_dir", required=True, help="writes lp/ and nn/ subdirs")
    p.add_argument("--pattern", default="*.csv")
    p.add_argument("--layer", default="middle", choices=["global", "middle", "local", "all"],
                   help="must match what trained the probes")
    p.add_argument("--pooling", default="mean", choices=["mean", "max", "cls"])
    p.add_argument("--max_length", type=int, default=8200)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--save_metrics", action="store_true")
    return p.parse_args()


def load_heads(embedding_dir, device):
    with open(os.path.join(embedding_dir, LP_FILE), "rb") as f:
        bundle = pickle.load(f)
    lp = (bundle["classifier"], bundle["scaler"])
    ckpt = torch.load(os.path.join(embedding_dir, NN_FILE), map_location=device, weights_only=True)
    nn_model = ThreeLayerNN(ckpt["input_dim"], ckpt["hidden_dim"]).to(device)
    nn_model.load_state_dict(ckpt["model_state_dict"]); nn_model.eval()
    with open(os.path.join(embedding_dir, NN_SCALER_FILE), "rb") as f:
        nn_scaler = pickle.load(f)
    return lp, (nn_model, nn_scaler)


def apply_lp(lp, emb):
    clf, scaler = lp
    return clf.predict_proba(scaler.transform(emb))


def apply_nn(nn, emb, device):
    model, scaler = nn
    X = torch.FloatTensor(scaler.transform(emb)).to(device)
    with torch.no_grad():
        return torch.softmax(model(X), dim=1).cpu().numpy()


def write_out(df, probs, threshold, out_csv, save_metrics):
    preds = (probs[:, 1] >= threshold).astype(int)
    out = df.copy()
    out["prob_0"] = probs[:, 0]; out["prob_1"] = probs[:, 1]; out["pred_label"] = preds
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    out.to_csv(out_csv, index=False)
    if save_metrics and "label" in df.columns:
        from sklearn.metrics import matthews_corrcoef
        y = df["label"].astype(int).values
        m = {"mcc": float(matthews_corrcoef(y, preds)) if len(np.unique(y)) > 1 else 0.0}
        with open(out_csv.replace(".csv", "_metrics.json"), "w") as f:
            json.dump(m, f)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  embedding_dir={args.embedding_dir}  layer={args.layer}")
    backbone = torch.load(args.model_path, map_location=device, weights_only=False)
    backbone.eval(); backbone.to(device)
    lp, nn = load_heads(args.embedding_dir, device)

    csvs = sorted(glob.glob(os.path.join(args.input_dir, args.pattern)))
    print(f"{len(csvs)} genome CSV(s)")
    for i, csv in enumerate(csvs):
        df = pd.read_csv(csv)
        if "sequence" not in df.columns:
            print(f"  [skip] {os.path.basename(csv)} (no sequence)"); continue
        emb = extract_embeddings(backbone, df["sequence"].astype(str).tolist(),
                                 args.batch_size, args.max_length, args.pooling,
                                 args.layer, device)
        stem = os.path.basename(csv).replace(".csv", "")
        name = f"genome_wide_{stem}_predictions.csv" if not stem.startswith("genome_wide_") else f"{stem}_predictions.csv"
        write_out(df, apply_lp(lp, emb), args.threshold,
                  os.path.join(args.output_dir, "lp", name), args.save_metrics)
        write_out(df, apply_nn(nn, emb, device), args.threshold,
                  os.path.join(args.output_dir, "nn", name), args.save_metrics)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(csvs)} done")
    print(f"Done -> {args.output_dir}/{{lp,nn}}/")


if __name__ == "__main__":
    main()
