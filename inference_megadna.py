#!/usr/bin/env python3
"""
Inference Script for megaDNA

This script performs inference on a CSV file using a pretrained megaDNA model
and a trained classifier (3-layer NN checkpoint from embedding_analysis_megadna.py).

Workflow:
    1. Load the megaDNA backbone and extract embeddings
    2. Load the trained 3-layer NN classifier and scaler
    3. Standardize embeddings with the saved scaler
    4. Classify with the NN and output predictions

Input CSV format:
    - sequence: DNA sequence
    - label: Ground truth label (optional, used for metrics)

Output CSV format:
    - sequence: Original sequence
    - label: Original label (if present)
    - prob_0: Probability of class 0
    - prob_1: Probability of class 1
    - pred_label: Predicted label (argmax or thresholded)

Usage:
    python inference_megadna.py \
        --input_csv /path/to/test.csv \
        --model_path /path/to/megaDNA_model.pt \
        --classifier_path /path/to/three_layer_nn_pretrained.pt \
        --scaler_path /path/to/three_layer_nn_pretrained_scaler.pkl \
        --output_csv /path/to/predictions.csv
"""

import argparse
import json
import os
import pickle
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix,
)


# DNA vocabulary for megaDNA
NT_VOCAB = ['**', 'A', 'T', 'C', 'G', '#']  # Start, A, T, C, G, End


class ThreeLayerNN(nn.Module):
    """Simple 3-layer neural network for binary classification."""

    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2),
        )

    def forward(self, x):
        return self.network(x)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run inference with megaDNA embeddings + trained classifier"
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="Path to input CSV file with 'sequence' column (and optionally 'label')",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to pretrained megaDNA backbone checkpoint (.pt file)",
    )
    parser.add_argument(
        "--classifier_path",
        type=str,
        required=True,
        help="Path to trained 3-layer NN checkpoint (three_layer_nn_pretrained.pt)",
    )
    parser.add_argument(
        "--scaler_path",
        type=str,
        required=True,
        help="Path to saved StandardScaler (three_layer_nn_pretrained_scaler.pkl)",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Path to output CSV file (default: input_csv with _predictions suffix)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for embedding extraction",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=96000,
        help="Maximum sequence length in base pairs (megaDNA supports up to 96kb)",
    )
    parser.add_argument(
        "--pooling",
        type=str,
        default="mean",
        choices=["mean", "max", "cls"],
        help="Pooling strategy (must match what was used during training)",
    )
    parser.add_argument(
        "--layer",
        type=str,
        default="middle",
        choices=["local", "middle", "global", "all"],
        help="Which embedding layer to use (must match what was used during training)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Classification threshold for prob_1 (default: 0.5)",
    )
    parser.add_argument(
        "--save_metrics",
        action="store_true",
        help="If labels are present, calculate and save metrics to JSON",
    )
    return parser.parse_args()


def encode_sequence(sequence: str) -> List[int]:
    """Encode a DNA sequence to token IDs for megaDNA."""
    encoded = [0]  # Start token '**'
    for nt in sequence.upper():
        if nt in NT_VOCAB:
            encoded.append(NT_VOCAB.index(nt))
        else:
            encoded.append(1)  # Unknown -> 'A'
    encoded.append(5)  # End token '#'
    return encoded


def extract_embeddings(
    model,
    sequences: List[str],
    batch_size: int,
    max_length: int,
    pooling: str,
    layer: str,
    device: torch.device,
) -> np.ndarray:
    """
    Extract embeddings from megaDNA model.

    Returns:
        embeddings array of shape (n_sequences, embedding_dim)
    """
    model.eval()
    all_embeddings = []

    layer_map = {'global': 0, 'middle': 1, 'local': 2}

    for i in tqdm(range(0, len(sequences), batch_size), desc="Extracting embeddings"):
        batch_seqs = sequences[i:i + batch_size]

        encoded_seqs = []
        for seq in batch_seqs:
            if len(seq) > max_length:
                seq = seq[:max_length]
            encoded_seqs.append(encode_sequence(seq))

        max_len = max(len(seq) for seq in encoded_seqs)
        padded_seqs = [seq + [0] * (max_len - len(seq)) for seq in encoded_seqs]

        input_tensor = torch.tensor(padded_seqs, dtype=torch.long).to(device)

        with torch.no_grad():
            embeddings_list = model(input_tensor, return_value='embedding')

            def pool_layer(layer_emb, pooling_method):
                batch_sz = layer_emb.shape[0]
                embed_dim = layer_emb.shape[-1]
                layer_emb_flat = layer_emb.reshape(batch_sz, -1, embed_dim)
                if pooling_method == 'mean':
                    return layer_emb_flat.mean(dim=1)
                elif pooling_method == 'max':
                    return layer_emb_flat.max(dim=1)[0]
                elif pooling_method == 'cls':
                    return layer_emb_flat[:, 0, :]

            if layer == 'all':
                pooled_layers = []
                for layer_emb in embeddings_list:
                    pooled = pool_layer(layer_emb, pooling)
                    pooled_layers.append(pooled)
                embeddings = torch.cat(pooled_layers, dim=-1)
            else:
                layer_idx = layer_map[layer]
                layer_emb = embeddings_list[layer_idx]
                embeddings = pool_layer(layer_emb, pooling)

            all_embeddings.append(embeddings.cpu().numpy())

    return np.vstack(all_embeddings)


def calculate_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
) -> Dict[str, float]:
    """Calculate comprehensive metrics."""
    metrics = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "mcc": float(matthews_corrcoef(labels, predictions)),
    }

    try:
        metrics["auc"] = float(roc_auc_score(labels, probabilities[:, 1]))
    except ValueError:
        metrics["auc"] = 0.0

    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    metrics["sensitivity"] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    metrics["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    metrics["true_negatives"] = int(tn)
    metrics["false_positives"] = int(fp)
    metrics["false_negatives"] = int(fn)
    metrics["true_positives"] = int(tp)

    return metrics


def main():
    """Main function to run inference."""
    args = parse_arguments()

    print("\n" + "=" * 60)
    print("megaDNA Inference")
    print("=" * 60)

    start_time = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load input CSV
    print(f"\nLoading input CSV: {args.input_csv}")
    df = pd.read_csv(args.input_csv)

    if "sequence" not in df.columns:
        raise ValueError("Input CSV must have a 'sequence' column")

    has_labels = "label" in df.columns
    print(f"  Samples: {len(df)}")
    print(f"  Has labels: {has_labels}")

    # Step 1: Load megaDNA backbone
    print(f"\n1. Loading megaDNA backbone from: {args.model_path}")
    backbone = torch.load(args.model_path, map_location=device, weights_only=False)
    backbone.eval()
    backbone.to(device)
    print("  Backbone loaded")

    # Step 2: Extract embeddings
    print(f"\n2. Extracting embeddings (layer={args.layer}, pooling={args.pooling})...")
    sequences = df["sequence"].tolist()
    embeddings = extract_embeddings(
        backbone, sequences,
        args.batch_size, args.max_length,
        args.pooling, args.layer, device,
    )
    print(f"  Embeddings shape: {embeddings.shape}")

    # Free backbone memory
    del backbone
    torch.cuda.empty_cache()

    # Step 3: Load scaler and standardize embeddings
    print(f"\n3. Loading scaler from: {args.scaler_path}")
    with open(args.scaler_path, "rb") as f:
        scaler = pickle.load(f)
    embeddings_scaled = scaler.transform(embeddings)
    print(f"  Embeddings standardized")

    # Step 4: Load 3-layer NN classifier
    print(f"\n4. Loading classifier from: {args.classifier_path}")
    checkpoint = torch.load(args.classifier_path, map_location=device, weights_only=True)
    input_dim = checkpoint["input_dim"]
    hidden_dim = checkpoint["hidden_dim"]

    if input_dim != embeddings.shape[1]:
        raise ValueError(
            f"Classifier input_dim ({input_dim}) does not match embedding dim ({embeddings.shape[1]}). "
            f"Make sure --layer and --pooling match what was used during training."
        )

    classifier = ThreeLayerNN(input_dim, hidden_dim).to(device)
    classifier.load_state_dict(checkpoint["model_state_dict"])
    classifier.eval()
    print(f"  Classifier loaded (input_dim={input_dim}, hidden_dim={hidden_dim})")

    # Step 5: Run classification
    print(f"\n5. Running classification...")
    embeddings_tensor = torch.FloatTensor(embeddings_scaled).to(device)

    with torch.no_grad():
        logits = classifier(embeddings_tensor)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        preds = torch.argmax(logits, dim=-1).cpu().numpy()

    # Apply custom threshold
    if args.threshold != 0.5:
        print(f"  Applying custom threshold: {args.threshold}")
        preds = (probs[:, 1] >= args.threshold).astype(int)

    # Create output dataframe
    output_df = df.copy()
    output_df["prob_0"] = probs[:, 0]
    output_df["prob_1"] = probs[:, 1]
    output_df["pred_label"] = preds

    # Set output path
    if args.output_csv is None:
        base, ext = os.path.splitext(args.input_csv)
        args.output_csv = f"{base}_predictions{ext}"

    # Save predictions
    output_df.to_csv(args.output_csv, index=False)
    print(f"\n6. Saved predictions to: {args.output_csv}")

    # Calculate and print metrics if labels present
    if has_labels and args.save_metrics:
        labels = df["label"].values
        metrics = calculate_metrics(labels, preds, probs)

        metrics["model_path"] = args.model_path
        metrics["classifier_path"] = args.classifier_path
        metrics["input_csv"] = args.input_csv
        metrics["threshold"] = args.threshold
        metrics["layer"] = args.layer
        metrics["pooling"] = args.pooling
        metrics["num_samples"] = len(df)

        metrics_path = args.output_csv.replace(".csv", "_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved metrics to: {metrics_path}")

        print("\n" + "=" * 60)
        print("METRICS (threshold = {:.2f})".format(args.threshold))
        print("=" * 60)
        print(f"  Accuracy:    {metrics['accuracy']:.4f}")
        print(f"  Precision:   {metrics['precision']:.4f}")
        print(f"  Recall:      {metrics['recall']:.4f}")
        print(f"  F1 Score:    {metrics['f1']:.4f}")
        print(f"  MCC:         {metrics['mcc']:.4f}")
        print(f"  AUC:         {metrics['auc']:.4f}")
        print(f"  Sensitivity: {metrics['sensitivity']:.4f}")
        print(f"  Specificity: {metrics['specificity']:.4f}")
        print("=" * 60)

    elif has_labels:
        labels = df["label"].values
        acc = accuracy_score(labels, preds)
        print(f"\nAccuracy: {acc:.4f}")

    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.2f} seconds")
    print(f"Throughput: {len(df) / elapsed:.1f} sequences/second")


if __name__ == "__main__":
    main()
