#!/usr/bin/env python3
"""
Inference Script for megaDNA

This script performs inference on a CSV file using a fine-tuned megaDNA classifier.
It outputs predictions with probability scores for threshold analysis.

Input CSV format:
    - sequence: DNA sequence
    - label: Ground truth label (optional, used for comparison)

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
        --classifier_path /path/to/classifier.pt \
        --output_csv /path/to/predictions.csv
"""

import argparse
import json
import os
import pickle
import time
from typing import Dict, List

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
from sklearn.preprocessing import StandardScaler


# DNA vocabulary for megaDNA
NT_VOCAB = ['**', 'A', 'T', 'C', 'G', '#']  # Start, A, T, C, G, End


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run inference on CSV file with megaDNA model and classifier"
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
        help="Path to megaDNA model checkpoint (.pt file)",
    )
    parser.add_argument(
        "--classifier_path",
        type=str,
        required=True,
        help="Path to trained classifier (.pt for neural, .pkl for logistic)",
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
        help="Pooling strategy for embeddings",
    )
    parser.add_argument(
        "--layer",
        type=str,
        default="middle",
        choices=["local", "middle", "global", "all"],
        help="Which embedding layer to use",
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
    """
    Encode a DNA sequence to token IDs for megaDNA.

    Args:
        sequence: DNA sequence string

    Returns:
        List of token IDs
    """
    encoded = [0]  # Start token '**'
    for nt in sequence.upper():
        if nt in NT_VOCAB:
            encoded.append(NT_VOCAB.index(nt))
        else:
            # Unknown nucleotide -> map to 'A' (index 1)
            encoded.append(1)
    encoded.append(5)  # End token '#'
    return encoded


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
    Extract embeddings from megaDNA model for given sequences.

    Args:
        model: The megaDNA model
        sequences: List of DNA sequences
        batch_size: Batch size for processing
        max_length: Maximum sequence length
        pooling: Pooling strategy ('mean', 'max', 'cls')
        layer: Which layer to use ('local', 'middle', 'global', 'all')
        device: Device to run on

    Returns:
        Embeddings array
    """
    model.eval()
    all_embeddings = []

    # Layer index mapping
    layer_map = {'global': 0, 'middle': 1, 'local': 2}

    # Process in batches
    for i in tqdm(range(0, len(sequences), batch_size), desc="Extracting embeddings"):
        batch_seqs = sequences[i:i + batch_size]

        # Encode sequences
        encoded_seqs = []
        for seq in batch_seqs:
            # Truncate if necessary
            if len(seq) > max_length:
                seq = seq[:max_length]
            encoded_seqs.append(encode_sequence(seq))

        # Pad sequences to same length
        max_len = max(len(seq) for seq in encoded_seqs)
        padded_seqs = [seq + [0] * (max_len - len(seq)) for seq in encoded_seqs]

        input_tensor = torch.tensor(padded_seqs, dtype=torch.long).to(device)

        with torch.no_grad():
            # Get embeddings - returns list of 3 layers
            embeddings_list = model(input_tensor, return_value='embedding')

            if layer == 'all':
                # Concatenate all layers after pooling
                pooled_layers = []
                for layer_emb in embeddings_list:
                    if pooling == 'mean':
                        pooled = layer_emb.mean(dim=1)
                    elif pooling == 'max':
                        pooled = layer_emb.max(dim=1)[0]
                    elif pooling == 'cls':
                        pooled = layer_emb[:, 0, :]
                    pooled_layers.append(pooled)
                embeddings = torch.cat(pooled_layers, dim=-1)
            else:
                layer_idx = layer_map[layer]
                layer_emb = embeddings_list[layer_idx]

                if pooling == 'mean':
                    embeddings = layer_emb.mean(dim=1)
                elif pooling == 'max':
                    embeddings = layer_emb.max(dim=1)[0]
                elif pooling == 'cls':
                    embeddings = layer_emb[:, 0, :]

            all_embeddings.append(embeddings.cpu().numpy())

    embeddings_array = np.vstack(all_embeddings)
    return embeddings_array


def run_inference(
    embeddings: np.ndarray,
    classifier,
    classifier_type: str,
    scaler: StandardScaler,
    device: torch.device,
) -> tuple:
    """
    Run inference using classifier on embeddings.

    Args:
        embeddings: Extracted embeddings
        classifier: Trained classifier (neural network or logistic regression)
        classifier_type: 'neural' or 'logistic'
        scaler: StandardScaler for normalizing embeddings
        device: Device to run on

    Returns:
        Tuple of (probabilities array shape (n, 2), predictions array)
    """
    # Scale embeddings
    scaled_embeddings = scaler.transform(embeddings)

    if classifier_type == 'neural':
        classifier.eval()
        with torch.no_grad():
            X = torch.FloatTensor(scaled_embeddings).to(device)
            outputs = classifier(X)
            probs = torch.softmax(outputs, dim=-1).cpu().numpy()
            preds = torch.argmax(outputs, dim=-1).cpu().numpy()
    else:  # logistic
        preds = classifier.predict(scaled_embeddings)
        probs = classifier.predict_proba(scaled_embeddings)

    return probs, preds


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

    # AUC
    try:
        metrics["auc"] = float(roc_auc_score(labels, probabilities[:, 1]))
    except ValueError:
        metrics["auc"] = 0.0

    # Sensitivity and Specificity
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    metrics["sensitivity"] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    metrics["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    # Confusion matrix values
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

    # Set device
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

    # Load megaDNA model
    print(f"\nLoading megaDNA model from: {args.model_path}")
    model = torch.load(args.model_path, map_location=device, weights_only=False)
    model.eval()
    model.to(device)

    # Load classifier
    print(f"Loading classifier from: {args.classifier_path}")
    if args.classifier_path.endswith('.pkl'):
        # Logistic regression
        with open(args.classifier_path, 'rb') as f:
            classifier_data = pickle.load(f)
        classifier = classifier_data['classifier']
        scaler = classifier_data['scaler']
        classifier_type = 'logistic'
    else:
        # Neural network
        checkpoint = torch.load(args.classifier_path, map_location=device)
        input_dim = checkpoint['input_dim']
        hidden_dim = checkpoint.get('hidden_dim', 256)
        classifier = ThreeLayerNN(input_dim, hidden_dim).to(device)
        classifier.load_state_dict(checkpoint['model_state_dict'])
        classifier.eval()
        # Load scaler - should be saved alongside or we need to fit one
        scaler_path = args.classifier_path.replace('.pt', '_scaler.pkl')
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
        else:
            print("  Warning: Scaler not found, using StandardScaler with default params")
            scaler = StandardScaler()
            # This is not ideal - scaler should be fitted on training data
            # For now, we'll fit on the input data (not recommended for production)
        classifier_type = 'neural'

    # Extract embeddings
    sequences = df["sequence"].tolist()
    print(f"\nExtracting embeddings (layer={args.layer}, pooling={args.pooling})...")
    embeddings = extract_embeddings(
        model, sequences,
        args.batch_size, args.max_length, args.pooling, args.layer, device,
    )
    print(f"  Embedding shape: {embeddings.shape}")

    # Fit scaler if needed (for neural network without saved scaler)
    if classifier_type == 'neural' and not hasattr(scaler, 'mean_'):
        print("  Fitting scaler on input data (not recommended for production)")
        scaler.fit(embeddings)

    # Run inference
    print("\nRunning inference...")
    probs, preds = run_inference(
        embeddings, classifier, classifier_type, scaler, device,
    )

    # Apply custom threshold if specified
    if args.threshold != 0.5:
        print(f"Applying custom threshold: {args.threshold}")
        preds_thresholded = (probs[:, 1] >= args.threshold).astype(int)
    else:
        preds_thresholded = preds

    # Create output dataframe
    output_df = df.copy()
    output_df["prob_0"] = probs[:, 0]
    output_df["prob_1"] = probs[:, 1]
    output_df["pred_label"] = preds_thresholded

    # Set output path
    if args.output_csv is None:
        base, ext = os.path.splitext(args.input_csv)
        args.output_csv = f"{base}_predictions{ext}"

    # Save predictions
    output_df.to_csv(args.output_csv, index=False)
    print(f"\nSaved predictions to: {args.output_csv}")

    # Calculate and save metrics if labels present
    if has_labels and args.save_metrics:
        labels = df["label"].values
        metrics = calculate_metrics(labels, preds_thresholded, probs)

        # Add metadata
        metrics["model_path"] = args.model_path
        metrics["classifier_path"] = args.classifier_path
        metrics["input_csv"] = args.input_csv
        metrics["threshold"] = args.threshold
        metrics["num_samples"] = len(df)
        metrics["layer"] = args.layer
        metrics["pooling"] = args.pooling

        # Save metrics
        metrics_path = args.output_csv.replace(".csv", "_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved metrics to: {metrics_path}")

        # Print metrics
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
        # Just print basic accuracy even if not saving
        labels = df["label"].values
        acc = accuracy_score(labels, preds_thresholded)
        print(f"\nAccuracy: {acc:.4f}")

    # Print timing
    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.2f} seconds")
    print(f"Throughput: {len(df) / elapsed:.1f} sequences/second")


if __name__ == "__main__":
    main()
