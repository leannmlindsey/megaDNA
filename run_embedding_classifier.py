"""
Ready-to-run script for phage/bacteria classification using Approach 1 (Embeddings)
Customized for your CSV data format: sequence,label

Usage:
    python run_embedding_classifier.py --data_dir /data/lindseylm/GLM_EVALUATIONS/LAMBDA/CLEANED_DATA \
                                        --model_path megaDNA_phage_145M.pt \
                                        --classifier_type logistic

Requirements:
    - train.csv, dev.csv, test.csv in data_dir
    - CSV format: sequence,label
    - MegaDNA model file
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import pickle
import argparse
import random
import time
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, roc_auc_score, confusion_matrix,
    silhouette_score, silhouette_samples, matthews_corrcoef,
    precision_recall_fscore_support, log_loss
)
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from tqdm import tqdm
import json
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server use
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def encode_sequence(sequence, nt_vocab=['**', 'A', 'T', 'C', 'G', '#']):
    """Encode a DNA sequence to its numerical representation."""
    return [0] + [nt_vocab.index(nt) if nt in nt_vocab else 1 for nt in sequence.upper()] + [5]


def load_data(data_dir):
    """
    Load train, dev, and test data from CSV files.
    Expected format: sequence,label

    Returns:
        Dictionary with train/dev/test splits
    """
    data_dir = Path(data_dir)

    print("Loading data from CSV files...")

    # Load training data
    train_df = pd.read_csv(data_dir / 'train.csv')
    print(f"  Train: {len(train_df)} sequences")

    # Load dev (validation) data
    dev_df = pd.read_csv(data_dir / 'dev.csv')
    print(f"  Dev:   {len(dev_df)} sequences")

    # Load test data
    test_df = pd.read_csv(data_dir / 'test.csv')
    print(f"  Test:  {len(test_df)} sequences")

    # Convert to lists
    data = {
        'train': {
            'sequences': train_df['sequence'].tolist(),
            'labels': train_df['label'].tolist()
        },
        'dev': {
            'sequences': dev_df['sequence'].tolist(),
            'labels': dev_df['label'].tolist()
        },
        'test': {
            'sequences': test_df['sequence'].tolist(),
            'labels': test_df['label'].tolist()
        }
    }

    # Print class distribution
    for split_name in ['train', 'dev', 'test']:
        labels = data[split_name]['labels']
        n_phage = sum(labels)
        n_bacteria = len(labels) - n_phage
        print(f"  {split_name.capitalize()}: {n_phage} phage, {n_bacteria} bacteria")

    return data


def extract_embeddings(model, sequences, device='cpu', batch_size=8, pooling='mean', layer='middle'):
    """
    Extract embeddings from MegaDNA model for a list of sequences.

    Args:
        model: MegaDNA model
        sequences: List of DNA sequences (strings)
        device: 'cpu' or 'cuda'
        batch_size: Number of sequences to process at once
        pooling: How to aggregate sequence embeddings ('mean', 'max', 'cls')
        layer: Which layer to use ('local'=0, 'middle'=1, 'global'=2, 'all'=concatenate all)

    Returns:
        numpy array of embeddings [num_sequences, embedding_dim]
    """
    # Map layer names to indices (based on actual output dimensions)
    # Layer 0: 512 dim (global), Layer 1: 256 dim (middle), Layer 2: 196 dim (local)
    layer_map = {'global': 0, 'middle': 1, 'local': 2}

    print(f"Using layer: {layer}, pooling: {pooling}")
    model.eval()
    model.to(device)

    all_embeddings = []

    print(f"Extracting embeddings from {len(sequences)} sequences...")

    with torch.no_grad():
        for i in tqdm(range(0, len(sequences), batch_size)):
            batch_seqs = sequences[i:i+batch_size]

            # Encode sequences
            encoded_seqs = [encode_sequence(seq) for seq in batch_seqs]

            # Pad to same length in batch
            max_len = max(len(seq) for seq in encoded_seqs)
            padded_seqs = [seq + [0] * (max_len - len(seq)) for seq in encoded_seqs]

            # Convert to tensor
            input_tensor = torch.tensor(padded_seqs).to(device)

            try:
                # Get embeddings from all three transformer stages
                embeddings = model(input_tensor, return_value='embedding')

                # Debug: Check shapes
                if i == 0:  # Only print for first batch
                    print(f"Debug - Raw embedding shapes:")
                    for idx, emb in enumerate(embeddings):
                        print(f"  Layer {idx}: {emb.shape}")

                # Select layer(s) based on configuration
                if layer == 'all':
                    # Concatenate all three layers
                    actual_batch_size = len(batch_seqs)
                    pooled_layers = []

                    for layer_idx, emb in enumerate(embeddings):
                        # Ensure emb is 3D: [expanded_batch, seq_len, hidden_dim]
                        if len(emb.shape) == 2:
                            # Already pooled, just use as-is
                            pooled_layers.append(emb)
                        else:
                            # Reshape if batch dimension was expanded
                            expanded_batch_size = emb.shape[0]
                            if expanded_batch_size != actual_batch_size:
                                expansion_factor = expanded_batch_size // actual_batch_size
                                seq_len = emb.shape[1]
                                hidden_dim = emb.shape[2]
                                emb = emb.reshape(actual_batch_size, expansion_factor, seq_len, hidden_dim)

                                # Pool across expansion and sequence dimensions
                                if pooling == 'mean':
                                    pooled_layers.append(emb.mean(dim=(1, 2)))
                                elif pooling == 'max':
                                    pooled_layers.append(emb.reshape(actual_batch_size, -1, hidden_dim).max(dim=1)[0])
                                elif pooling == 'cls':
                                    pooled_layers.append(emb[:, 0, 0, :])
                            else:
                                # No expansion, normal pooling
                                if pooling == 'mean':
                                    pooled_layers.append(emb.mean(dim=1))
                                elif pooling == 'max':
                                    pooled_layers.append(emb.max(dim=1)[0])
                                elif pooling == 'cls':
                                    pooled_layers.append(emb[:, 0, :])

                    pooled = torch.cat(pooled_layers, dim=1)
                else:
                    # Use single layer
                    layer_idx = layer_map[layer]
                    final_embeddings = embeddings[layer_idx]

                    if i == 0:  # Debug first batch
                        print(f"Selected layer {layer} (idx={layer_idx}), shape: {final_embeddings.shape}")
                        print(f"Expected batch size: {len(batch_seqs)}")

                    # Check if embeddings are already 2D or need pooling
                    if len(final_embeddings.shape) == 2:
                        # Already [batch, hidden_dim], no pooling needed
                        pooled = final_embeddings
                        if i == 0:
                            print(f"  Already 2D, using as-is")
                    else:
                        # 3D: [expanded_batch, seq_len, hidden_dim] - need to reshape and pool
                        actual_batch_size = len(batch_seqs)
                        expanded_batch_size = final_embeddings.shape[0]

                        if i == 0:
                            print(f"  Pooling from 3D -> 2D using {pooling} pooling")
                            print(f"  Batch expansion factor: {expanded_batch_size // actual_batch_size}")

                        # Reshape to [actual_batch, expansion_factor, seq_len, hidden_dim]
                        expansion_factor = expanded_batch_size // actual_batch_size
                        seq_len = final_embeddings.shape[1]
                        hidden_dim = final_embeddings.shape[2]

                        reshaped = final_embeddings.reshape(
                            actual_batch_size, expansion_factor, seq_len, hidden_dim
                        )

                        if i == 0:
                            print(f"  Reshaped to: {reshaped.shape}")

                        # Pool across both expansion and sequence dimensions
                        if pooling == 'mean':
                            pooled = reshaped.mean(dim=(1, 2))  # Pool over expansion and seq_len
                        elif pooling == 'max':
                            pooled = reshaped.reshape(actual_batch_size, -1, hidden_dim).max(dim=1)[0]
                        elif pooling == 'cls':
                            pooled = reshaped[:, 0, 0, :]  # First position of first expansion

                    if i == 0:
                        print(f"  Final pooled shape: {pooled.shape}")
                        print(f"  ✓ Correct batch size: {pooled.shape[0] == len(batch_seqs)}")

                all_embeddings.append(pooled.cpu().numpy())

            except Exception as e:
                print(f"Error processing batch {i}-{i+len(batch_seqs)}: {e}")
                import traceback
                traceback.print_exc()
                # Add zero embeddings for failed sequences with correct dimension
                # Layer 0: global (512), Layer 1: middle (256), Layer 2: local (196)
                layer_dims = {'global': 512, 'middle': 256, 'local': 196}
                if layer == 'all':
                    emb_dim = sum(layer_dims.values())
                else:
                    emb_dim = layer_dims[layer]
                zero_emb = np.zeros((len(batch_seqs), emb_dim))
                all_embeddings.append(zero_emb)

    return np.vstack(all_embeddings)


class ImprovedNNClassifier(nn.Module):
    """
    Improved 3-layer neural network classifier with batch normalization.
    Architecture: input -> hidden1 -> hidden2 -> hidden3 -> output
    """

    def __init__(self, input_dim, hidden_dim1=512, hidden_dim2=256, hidden_dim3=128, dropout=0.4):
        super().__init__()

        self.network = nn.Sequential(
            # Layer 1
            nn.Linear(input_dim, hidden_dim1),
            nn.BatchNorm1d(hidden_dim1),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Layer 2
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.BatchNorm1d(hidden_dim2),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Layer 3
            nn.Linear(hidden_dim2, hidden_dim3),
            nn.BatchNorm1d(hidden_dim3),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Output layer
            nn.Linear(hidden_dim3, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)


def compute_and_plot_silhouette(X, y, output_dir, split_name='test', class_names=['Bacteria', 'Phage']):
    """
    Compute silhouette scores and create silhouette plot.

    Args:
        X: Embeddings array
        y: True labels
        output_dir: Directory to save plots
        split_name: Name of the split (e.g., 'test', 'dev')
        class_names: Names of the classes

    Returns:
        Dictionary with silhouette metrics
    """
    print(f"\nComputing silhouette scores for {split_name} set...")

    # Compute overall silhouette score
    silhouette_avg = silhouette_score(X, y)
    print(f"Average silhouette score: {silhouette_avg:.4f}")

    # Compute per-sample silhouette scores
    sample_silhouette_values = silhouette_samples(X, y)

    # Create silhouette plot
    fig, ax = plt.subplots(figsize=(10, 6))

    y_lower = 10
    unique_labels = np.unique(y)

    for i, label in enumerate(unique_labels):
        # Get silhouette scores for samples in this class
        ith_cluster_silhouette_values = sample_silhouette_values[y == label]
        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / len(unique_labels))
        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
            label=f'{class_names[label]} (n={size_cluster_i})'
        )

        # Label the silhouette plots with their cluster numbers at the middle
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(class_names[label]))

        y_lower = y_upper + 10

    ax.set_xlabel("Silhouette Coefficient")
    ax.set_ylabel("Sample Index")
    ax.set_title(f"Silhouette Plot for {split_name.capitalize()} Set\n"
                 f"Average Score: {silhouette_avg:.4f}")

    # Add vertical line for average silhouette score
    ax.axvline(x=silhouette_avg, color="red", linestyle="--",
               label=f'Average: {silhouette_avg:.4f}')

    ax.set_yticks([])
    ax.set_xlim([-0.2, 1])
    ax.legend(loc='best')

    plt.tight_layout()
    plot_path = output_dir / f'silhouette_plot_{split_name}.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Silhouette plot saved to: {plot_path}")

    # Compute per-class statistics
    class_stats = {}
    for label in unique_labels:
        class_scores = sample_silhouette_values[y == label]
        class_stats[class_names[label]] = {
            'mean': float(np.mean(class_scores)),
            'std': float(np.std(class_scores)),
            'min': float(np.min(class_scores)),
            'max': float(np.max(class_scores))
        }

    return {
        'overall_score': float(silhouette_avg),
        'per_class_stats': class_stats,
        'sample_scores': sample_silhouette_values.tolist()
    }


def visualize_embeddings(X, y, output_dir, split_name='test', class_names=['Bacteria', 'Phage'], seed=42):
    """
    Create PCA and t-SNE visualizations of embeddings.

    Args:
        X: Embeddings array
        y: True labels
        output_dir: Directory to save plots
        split_name: Name of the split
        class_names: Names of the classes
        seed: Random seed for t-SNE reproducibility
    """
    print(f"\nCreating embedding visualizations for {split_name} set...")

    # PCA visualization
    print("Running PCA...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ['#1f77b4', '#ff7f0e']  # Blue for bacteria, orange for phage

    for label in np.unique(y):
        mask = y == label
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                  c=[colors[label]], label=class_names[label],
                  alpha=0.6, s=50, edgecolors='k', linewidth=0.5)

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
    ax.set_title(f'PCA Visualization - {split_name.capitalize()} Set')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    pca_path = output_dir / f'pca_visualization_{split_name}.png'
    plt.savefig(pca_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"PCA plot saved to: {pca_path}")

    # t-SNE visualization
    print("Running t-SNE (this may take a minute)...")
    tsne = TSNE(n_components=2, random_state=seed, perplexity=min(30, len(X)-1))
    X_tsne = tsne.fit_transform(X)

    fig, ax = plt.subplots(figsize=(10, 8))

    for label in np.unique(y):
        mask = y == label
        ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                  c=[colors[label]], label=class_names[label],
                  alpha=0.6, s=50, edgecolors='k', linewidth=0.5)

    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    ax.set_title(f't-SNE Visualization - {split_name.capitalize()} Set')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    tsne_path = output_dir / f'tsne_visualization_{split_name}.png'
    plt.savefig(tsne_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"t-SNE plot saved to: {tsne_path}")

    return {
        'pca_variance_explained': [float(x) for x in pca.explained_variance_ratio_],
        'pca_components': X_pca.tolist(),
        'tsne_components': X_tsne.tolist()
    }


def train_neural_classifier(X_train, y_train, X_val, y_val,
                            epochs=200, lr=0.001, device='cpu',
                            hidden_dim1=512, hidden_dim2=256, hidden_dim3=128,
                            dropout=0.4, weight_decay=1e-4, batch_size=64):
    """
    Train an improved 3-layer neural network classifier.

    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        epochs: Maximum training epochs
        lr: Learning rate
        device: 'cpu' or 'cuda'
        hidden_dim1/2/3: Hidden layer dimensions
        dropout: Dropout rate
        weight_decay: L2 regularization
        batch_size: Mini-batch size for training
    """

    # Debug: Check shapes
    print(f"Debug - X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"Debug - X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")

    # Ensure labels match embeddings
    assert X_train.shape[0] == y_train.shape[0], f"X_train ({X_train.shape[0]}) and y_train ({y_train.shape[0]}) size mismatch!"
    assert X_val.shape[0] == y_val.shape[0], f"X_val ({X_val.shape[0]}) and y_val ({y_val.shape[0]}) size mismatch!"

    input_dim = X_train.shape[1]
    model = ImprovedNNClassifier(
        input_dim,
        hidden_dim1=hidden_dim1,
        hidden_dim2=hidden_dim2,
        hidden_dim3=hidden_dim3,
        dropout=dropout
    ).to(device)

    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Learning rate scheduler - reduce on plateau
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )

    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.FloatTensor(y_val).unsqueeze(1).to(device)

    print(f"Debug - X_train_t shape: {X_train_t.shape}, y_train_t shape: {y_train_t.shape}")

    # Create data loader for mini-batch training
    train_dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    patience_limit = 20

    print("\nTraining improved 3-layer neural network classifier...")
    print(f"Architecture: {input_dim} -> {hidden_dim1} -> {hidden_dim2} -> {hidden_dim3} -> 1")
    print(f"Training params: lr={lr}, dropout={dropout}, weight_decay={weight_decay}")
    print(f"Batch size: {batch_size}, Max epochs: {epochs}")

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation phase
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_t)
            val_loss = criterion(val_outputs, y_val_t)
            val_preds = (val_outputs > 0.5).float()
            val_acc = (val_preds == y_val_t).float().mean()

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}: Train Loss={train_loss:.4f}, "
                  f"Val Loss={val_loss.item():.4f}, Val Acc={val_acc.item():.4f}")

        # Early stopping
        if patience_counter >= patience_limit:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    print(f"\nBest validation loss: {best_val_loss:.4f}")
    model.load_state_dict(best_model_state)
    return model


def main(args):
    """Main pipeline."""

    print("="*70)
    print("Phage/Bacteria Classification - Approach 1: Embedding Classifier")
    print("="*70)

    # Set random seed for reproducibility
    set_seed(args.seed)
    print(f"\nRandom seed: {args.seed}")

    # Set device
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    print(f"Using device: {device}")

    # Load data
    print("\n" + "="*70)
    print("Step 1: Loading Data")
    print("="*70)
    data = load_data(args.data_dir)

    # Load MegaDNA model
    print("\n" + "="*70)
    print("Step 2: Loading MegaDNA Model")
    print("="*70)
    print(f"Loading model from: {args.model_path}")
    megadna_model = torch.load(args.model_path, map_location=torch.device(device), weights_only=False)
    megadna_model.eval()
    print("Model loaded successfully!")

    # Extract embeddings (with caching)
    print("\n" + "="*70)
    print("Step 3: Extracting Embeddings")
    print("="*70)

    # Create cache filename based on configuration
    cache_dir = Path(args.data_dir) / 'embedding_cache'
    cache_dir.mkdir(exist_ok=True)
    cache_filename = f"embeddings_{args.layer}_{args.pooling}_{args.batch_size}.npz"
    cache_path = cache_dir / cache_filename

    # Check if cached embeddings exist
    if cache_path.exists() and not args.no_cache:
        print(f"Found cached embeddings at: {cache_path}")
        print("Loading cached embeddings...")
        cached_data = np.load(cache_path)
        X_train = cached_data['X_train']
        y_train = cached_data['y_train']
        X_dev = cached_data['X_dev']
        y_dev = cached_data['y_dev']
        X_test = cached_data['X_test']
        y_test = cached_data['y_test']
        print("✓ Cached embeddings loaded successfully!")
    else:
        print("No cached embeddings found. Extracting from scratch...")
        print("This may take a while depending on sequence length and batch size")
        print(f"(Embeddings will be cached to: {cache_path})")

        X_train = extract_embeddings(
            megadna_model, data['train']['sequences'],
            device=device, batch_size=args.batch_size, pooling=args.pooling, layer=args.layer
        )
        y_train = np.array(data['train']['labels'])

        X_dev = extract_embeddings(
            megadna_model, data['dev']['sequences'],
            device=device, batch_size=args.batch_size, pooling=args.pooling, layer=args.layer
        )
        y_dev = np.array(data['dev']['labels'])

        X_test = extract_embeddings(
            megadna_model, data['test']['sequences'],
            device=device, batch_size=args.batch_size, pooling=args.pooling, layer=args.layer
        )
        y_test = np.array(data['test']['labels'])

        # Save embeddings to cache
        print(f"\nSaving embeddings to cache: {cache_path}")
        np.savez(
            cache_path,
            X_train=X_train, y_train=y_train,
            X_dev=X_dev, y_dev=y_dev,
            X_test=X_test, y_test=y_test
        )
        print("✓ Embeddings cached for future runs!")

    print(f"\nEmbedding shapes:")
    print(f"  Train: {X_train.shape}")
    print(f"  Dev:   {X_dev.shape}")
    print(f"  Test:  {X_test.shape}")

    # Train classifier
    print("\n" + "="*70)
    print("Step 4: Training Classifier")
    print("="*70)

    if args.classifier_type == 'logistic':
        print("Training Logistic Regression classifier...")
        classifier = LogisticRegression(
            max_iter=1000,
            random_state=args.seed,
            class_weight='balanced',
            verbose=1
        )
        classifier.fit(X_train, y_train)

        # Dev predictions
        y_dev_pred = classifier.predict(X_dev)
        y_dev_proba = classifier.predict_proba(X_dev)[:, 1]

        # Test predictions
        y_test_pred = classifier.predict(X_test)
        y_test_proba = classifier.predict_proba(X_test)[:, 1]

    elif args.classifier_type == 'neural':
        classifier = train_neural_classifier(
            X_train, y_train, X_dev, y_dev,
            epochs=args.epochs,
            lr=args.lr,
            device=device,
            hidden_dim1=args.hidden_dim1,
            hidden_dim2=args.hidden_dim2,
            hidden_dim3=args.hidden_dim3,
            dropout=args.dropout,
            weight_decay=args.weight_decay,
            batch_size=args.nn_batch_size
        )

        # Test predictions
        classifier.eval()
        with torch.no_grad():
            X_test_t = torch.FloatTensor(X_test).to(device)
            y_test_proba = classifier(X_test_t).cpu().numpy().squeeze()
            y_test_pred = (y_test_proba > 0.5).astype(int)

            X_dev_t = torch.FloatTensor(X_dev).to(device)
            y_dev_proba = classifier(X_dev_t).cpu().numpy().squeeze()
            y_dev_pred = (y_dev_proba > 0.5).astype(int)

    # Evaluate
    print("\n" + "="*70)
    print("Step 5: Evaluation Results")
    print("="*70)

    # Time the evaluation
    eval_start_time = time.time()

    print("\nDevelopment Set Results:")
    print("-" * 70)
    dev_acc = accuracy_score(y_dev, y_dev_pred)
    dev_auc = roc_auc_score(y_dev, y_dev_proba)
    print(f"Accuracy:  {dev_acc:.4f}")
    print(f"ROC-AUC:   {dev_auc:.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_dev, y_dev_pred))
    print("\nClassification Report:")
    print(classification_report(y_dev, y_dev_pred, target_names=['Bacteria', 'Phage']))

    print("\n" + "="*70)
    print("Test Set Results:")
    print("-" * 70)

    # Calculate comprehensive metrics for test set
    test_acc = accuracy_score(y_test, y_test_pred)
    test_auc = roc_auc_score(y_test, y_test_proba)
    test_mcc = matthews_corrcoef(y_test, y_test_pred)

    # Get precision, recall, f1 for the positive class (phage = 1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_test_pred, average='binary', pos_label=1
    )

    # Calculate sensitivity and specificity from confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Same as recall
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # Calculate loss (binary cross-entropy) using sklearn for numerical stability
    # Handle potential NaN/inf values
    y_test_proba_clean = np.nan_to_num(y_test_proba, nan=0.5, posinf=1.0, neginf=0.0)
    y_test_proba_clean = np.clip(y_test_proba_clean, 1e-7, 1 - 1e-7)
    test_loss = log_loss(y_test, y_test_proba_clean)

    # Warn if there were NaN values
    n_nan = np.sum(np.isnan(y_test_proba))
    if n_nan > 0:
        print(f"Warning: {n_nan} NaN values found in predictions, replaced with 0.5")

    eval_runtime = time.time() - eval_start_time
    eval_samples_per_second = len(y_test) / eval_runtime if eval_runtime > 0 else 0

    # For neural network, calculate steps per second based on batch size
    if args.classifier_type == 'neural':
        n_eval_steps = (len(y_test) + args.nn_batch_size - 1) // args.nn_batch_size
    else:
        n_eval_steps = 1
    eval_steps_per_second = n_eval_steps / eval_runtime if eval_runtime > 0 else 0

    print(f"Accuracy:    {test_acc:.4f}")
    print(f"Precision:   {precision:.4f}")
    print(f"Recall:      {recall:.4f}")
    print(f"F1 Score:    {f1:.4f}")
    print(f"MCC:         {test_mcc:.4f}")
    print(f"Sensitivity: {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"ROC-AUC:     {test_auc:.4f}")
    print(f"Loss:        {test_loss:.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_test_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_test_pred, target_names=['Bacteria', 'Phage']))

    # Compute silhouette scores and create visualizations
    print("\n" + "="*70)
    print("Step 6: Silhouette Analysis and Embedding Visualization")
    print("="*70)

    output_dir = Path(args.output_dir) if args.output_dir else Path('results')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute silhouette scores for test set
    silhouette_metrics = compute_and_plot_silhouette(
        X_test, y_test, output_dir, split_name='test'
    )

    # Create embedding visualizations for test set
    visualization_data = visualize_embeddings(
        X_test, y_test, output_dir, split_name='test', seed=args.seed
    )

    # Save results
    if args.output_dir:
        print("\n" + "="*70)
        print("Step 7: Saving Results")
        print("="*70)

        # Save classifier
        if args.classifier_type == 'logistic':
            classifier_path = output_dir / 'classifier.pkl'
            with open(classifier_path, 'wb') as f:
                pickle.dump(classifier, f)
            print(f"Classifier saved to: {classifier_path}")
        else:
            classifier_path = output_dir / 'classifier.pt'
            torch.save(classifier.state_dict(), classifier_path)
            print(f"Classifier saved to: {classifier_path}")

        # Save embeddings
        embeddings_path = output_dir / 'embeddings.npz'
        np.savez(
            embeddings_path,
            X_train=X_train, y_train=y_train,
            X_dev=X_dev, y_dev=y_dev,
            X_test=X_test, y_test=y_test
        )
        print(f"Embeddings saved to: {embeddings_path}")

        # Save predictions
        predictions_path = output_dir / 'predictions.npz'
        np.savez(
            predictions_path,
            y_test_pred=y_test_pred,
            y_test_proba=y_test_proba,
            y_test_true=y_test
        )
        print(f"Predictions saved to: {predictions_path}")

        # Save test_results.json in standard format (matching other models)
        test_results = {
            'eval_loss': float(test_loss),
            'eval_accuracy': float(test_acc),
            'eval_precision': float(precision),
            'eval_recall': float(recall),
            'eval_f1': float(f1),
            'eval_mcc': float(test_mcc),
            'eval_sensitivity': float(sensitivity),
            'eval_specificity': float(specificity),
            'eval_auc': float(test_auc),
            'eval_runtime': float(eval_runtime),
            'eval_samples_per_second': float(eval_samples_per_second),
            'eval_steps_per_second': float(eval_steps_per_second),
            'epoch': args.epochs if args.classifier_type == 'neural' else 1,
            'seed': args.seed,
            'silhouette_score': silhouette_metrics['overall_score'],
            'silhouette_per_class': silhouette_metrics['per_class_stats'],
            'image_files': {
                'silhouette_plot': str(output_dir / 'silhouette_plot_test.png'),
                'pca_visualization': str(output_dir / 'pca_visualization_test.png'),
                'tsne_visualization': str(output_dir / 'tsne_visualization_test.png')
            }
        }
        test_results_path = output_dir / 'test_results.json'
        with open(test_results_path, 'w') as f:
            json.dump(test_results, f, indent=2)
        print(f"Test results saved to: {test_results_path}")

        # Save detailed metrics (including dev set and config)
        config_dict = {
            'classifier_type': args.classifier_type,
            'pooling': args.pooling,
            'layer': args.layer,
            'batch_size': args.batch_size,
            'seed': args.seed
        }

        # Add neural network specific config if applicable
        if args.classifier_type == 'neural':
            config_dict.update({
                'epochs': args.epochs,
                'lr': args.lr,
                'hidden_dim1': args.hidden_dim1,
                'hidden_dim2': args.hidden_dim2,
                'hidden_dim3': args.hidden_dim3,
                'dropout': args.dropout,
                'weight_decay': args.weight_decay,
                'nn_batch_size': args.nn_batch_size
            })

        metrics = {
            'dev': {
                'accuracy': float(dev_acc),
                'roc_auc': float(dev_auc)
            },
            'test': {
                'accuracy': float(test_acc),
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'mcc': float(test_mcc),
                'sensitivity': float(sensitivity),
                'specificity': float(specificity),
                'roc_auc': float(test_auc),
                'loss': float(test_loss),
                'silhouette_score': silhouette_metrics['overall_score'],
                'silhouette_per_class': silhouette_metrics['per_class_stats']
            },
            'config': config_dict
        }
        metrics_path = output_dir / 'metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Detailed metrics saved to: {metrics_path}")

        # Save detailed silhouette analysis
        silhouette_path = output_dir / 'silhouette_analysis.json'
        with open(silhouette_path, 'w') as f:
            json.dump(silhouette_metrics, f, indent=2)
        print(f"Silhouette analysis saved to: {silhouette_path}")

        # Save visualization data
        viz_path = output_dir / 'visualization_data.json'
        with open(viz_path, 'w') as f:
            json.dump(visualization_data, f, indent=2)
        print(f"Visualization data saved to: {viz_path}")

    print("\n" + "="*70)
    print("Done!")
    print("="*70)
    print(f"\nRandom seed used: {args.seed}")
    print("\nGenerated files:")
    print(f"  - Test results: {output_dir / 'test_results.json'}")
    print(f"  - Detailed metrics: {output_dir / 'metrics.json'}")
    print(f"  - Silhouette plot: {output_dir / 'silhouette_plot_test.png'}")
    print(f"  - PCA visualization: {output_dir / 'pca_visualization_test.png'}")
    print(f"  - t-SNE visualization: {output_dir / 'tsne_visualization_test.png'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Phage/Bacteria Classification using MegaDNA Embeddings')

    # Data arguments
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing train.csv, dev.csv, test.csv')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to MegaDNA model file (.pt)')

    # Model arguments
    parser.add_argument('--classifier_type', type=str, default='logistic',
                       choices=['logistic', 'neural'],
                       help='Type of classifier to use (default: logistic)')
    parser.add_argument('--pooling', type=str, default='mean',
                       choices=['mean', 'max', 'cls'],
                       help='Pooling method for sequence embeddings (default: mean)')
    parser.add_argument('--layer', type=str, default='middle',
                       choices=['local', 'middle', 'global', 'all'],
                       help='Which MegaDNA layer to use: local (16bp), middle (1024bp), global (96K), or all (concatenate). Paper recommends middle for taxonomy (default: middle)')

    # Training arguments
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for embedding extraction (default: 8)')
    parser.add_argument('--epochs', type=int, default=200,
                       help='Number of epochs for neural classifier (default: 200)')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate for neural classifier (default: 0.001)')

    # Neural network architecture arguments
    parser.add_argument('--hidden_dim1', type=int, default=512,
                       help='First hidden layer size (default: 512)')
    parser.add_argument('--hidden_dim2', type=int, default=256,
                       help='Second hidden layer size (default: 256)')
    parser.add_argument('--hidden_dim3', type=int, default=128,
                       help='Third hidden layer size (default: 128)')
    parser.add_argument('--dropout', type=float, default=0.4,
                       help='Dropout rate (default: 0.4)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='L2 regularization weight decay (default: 1e-4)')
    parser.add_argument('--nn_batch_size', type=int, default=64,
                       help='Batch size for neural network training (default: 64)')

    # Output arguments
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Directory to save results (default: results)')

    # Device arguments
    parser.add_argument('--cpu', action='store_true',
                       help='Use CPU even if GPU is available')

    # Caching arguments
    parser.add_argument('--no_cache', action='store_true',
                       help='Disable embedding caching (force re-extraction)')

    # Reproducibility arguments
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')

    args = parser.parse_args()

    main(args)
