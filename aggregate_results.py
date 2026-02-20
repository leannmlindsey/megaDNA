"""
Aggregate test results across multiple seeds.

Supports two modes:
  - embedding_analysis: Aggregates embedding_analysis_results.json from seed_* dirs
  - finetuning: Aggregates test_results.json from seed_* dirs (original behavior)

Usage:
    # Embedding analysis (default)
    python aggregate_results.py --results_dir ./output/embedding_analysis/lambda_filtered_2k --pattern "seed_*" --mode embedding_analysis

    # Fine-tuning
    python aggregate_results.py --results_dir ./output/lambda_filtered/2k --pattern "seed_*" --mode finetuning
"""

import argparse
import json
import numpy as np
from pathlib import Path
from glob import glob


# Metrics for embedding_analysis mode
EMBEDDING_ANALYSIS_METRICS = [
    # Neural network classifier
    'pretrained_nn_accuracy',
    'pretrained_nn_precision',
    'pretrained_nn_recall',
    'pretrained_nn_f1',
    'pretrained_nn_mcc',
    'pretrained_nn_auc',
    'pretrained_nn_sensitivity',
    'pretrained_nn_specificity',
    # Linear probe
    'pretrained_linear_probe_accuracy',
    'pretrained_linear_probe_precision',
    'pretrained_linear_probe_recall',
    'pretrained_linear_probe_f1',
    'pretrained_linear_probe_mcc',
    'pretrained_linear_probe_auc',
    'pretrained_linear_probe_sensitivity',
    'pretrained_linear_probe_specificity',
    # Embedding quality
    'pretrained_silhouette_score',
    'pretrained_pca_explained_variance_pc1',
    'pretrained_pca_explained_variance_pc2',
    'pretrained_pca_total_explained_variance',
]

# Metrics for finetuning mode
FINETUNING_METRICS = [
    'eval_loss',
    'eval_accuracy',
    'eval_precision',
    'eval_recall',
    'eval_f1',
    'eval_mcc',
    'eval_sensitivity',
    'eval_specificity',
    'eval_auc',
]


def aggregate_results(results_dir, pattern="seed_*", mode="embedding_analysis"):
    """
    Find result JSON files across seed directories and aggregate metrics.

    Args:
        results_dir: Directory containing seed output folders
        pattern: Glob pattern to match output folders
        mode: 'embedding_analysis' or 'finetuning'
    """
    results_dir = Path(results_dir)

    if mode == "embedding_analysis":
        result_filename = "embedding_analysis_results.json"
        metrics = EMBEDDING_ANALYSIS_METRICS
        mcc_key = "pretrained_nn_mcc"
    else:
        result_filename = "test_results.json"
        metrics = FINETUNING_METRICS
        mcc_key = "eval_mcc"

    # Find all result files
    search_pattern = str(results_dir / pattern / result_filename)
    result_files = sorted(glob(search_pattern))

    if not result_files:
        print(f"No results found matching: {search_pattern}")
        return None

    print(f"Found {len(result_files)} result files:")
    for f in result_files:
        print(f"  - {f}")

    # Load all results
    all_results = []
    seeds = []
    seed_dirs = []
    for result_file in result_files:
        with open(result_file, 'r') as f:
            data = json.load(f)
            all_results.append(data)
            seeds.append(data.get('seed', 'unknown'))
            seed_dirs.append(str(Path(result_file).parent))

    # Calculate statistics
    aggregated = {
        'n_seeds': len(all_results),
        'seeds': seeds,
        'mode': mode,
        'metrics': {}
    }

    print(f"\n{'='*70}")
    print(f"Aggregated Results (n={len(all_results)} seeds, mode={mode})")
    print(f"{'='*70}")
    print(f"{'Metric':<45} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print(f"{'-'*70}")

    for metric in metrics:
        values = [r[metric] for r in all_results if metric in r]
        if values:
            mean_val = np.mean(values)
            std_val = np.std(values)
            min_val = np.min(values)
            max_val = np.max(values)

            aggregated['metrics'][metric] = {
                'mean': float(mean_val),
                'std': float(std_val),
                'min': float(min_val),
                'max': float(max_val),
                'values': [float(v) for v in values]
            }

            print(f"{metric:<45} {mean_val:>10.4f} {std_val:>10.4f} {min_val:>10.4f} {max_val:>10.4f}")

    print(f"{'='*70}")

    # Identify best seed by MCC
    mcc_values = [r.get(mcc_key) for r in all_results]
    if any(v is not None for v in mcc_values):
        valid_mcc = [(i, v) for i, v in enumerate(mcc_values) if v is not None]
        best_idx, best_mcc = max(valid_mcc, key=lambda x: x[1])
        best_seed = seeds[best_idx]
        best_seed_dir = seed_dirs[best_idx]

        aggregated['best_seed'] = {
            'seed': best_seed,
            'seed_dir': best_seed_dir,
            'mcc': float(best_mcc),
        }

        print(f"\nBest seed by MCC ({mcc_key}):")
        print(f"  Seed: {best_seed}")
        print(f"  MCC:  {best_mcc:.4f}")
        print(f"  Dir:  {best_seed_dir}")

    return aggregated


def main():
    parser = argparse.ArgumentParser(description='Aggregate test results across seeds')
    parser.add_argument('--results_dir', type=str, required=True,
                       help='Directory containing seed output folders')
    parser.add_argument('--pattern', type=str, default='seed_*',
                       help='Glob pattern to match output folders (default: seed_*)')
    parser.add_argument('--mode', type=str, default='embedding_analysis',
                       choices=['embedding_analysis', 'finetuning'],
                       help='Result type to aggregate (default: embedding_analysis)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file for aggregated results (default: aggregated_results.json in results_dir)')

    args = parser.parse_args()

    aggregated = aggregate_results(args.results_dir, args.pattern, args.mode)

    if aggregated:
        # Save aggregated results
        output_path = args.output or str(Path(args.results_dir) / 'aggregated_results.json')
        with open(output_path, 'w') as f:
            json.dump(aggregated, f, indent=2)
        print(f"\nAggregated results saved to: {output_path}")


if __name__ == "__main__":
    main()
