"""
Aggregate test results across multiple seeds.

Usage:
    python aggregate_results.py --results_dir ./output/lambda_filtered/2k --pattern "megaDNA_lambda*"
"""

import argparse
import json
import numpy as np
from pathlib import Path
from glob import glob


def aggregate_results(results_dir, pattern="megaDNA_lambda*"):
    """
    Find all test_results.json files and aggregate metrics.

    Args:
        results_dir: Directory containing run output folders
        pattern: Glob pattern to match output folders
    """
    results_dir = Path(results_dir)

    # Find all test_results.json files
    search_pattern = str(results_dir / pattern / "test_results.json")
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
    for result_file in result_files:
        with open(result_file, 'r') as f:
            data = json.load(f)
            all_results.append(data)
            seeds.append(data.get('seed', 'unknown'))

    # Metrics to aggregate
    metrics = [
        'eval_loss', 'eval_accuracy', 'eval_precision', 'eval_recall',
        'eval_f1', 'eval_mcc', 'eval_sensitivity', 'eval_specificity', 'eval_auc'
    ]

    # Calculate statistics
    aggregated = {
        'n_seeds': len(all_results),
        'seeds': seeds,
        'metrics': {}
    }

    print(f"\n{'='*70}")
    print(f"Aggregated Results (n={len(all_results)} seeds)")
    print(f"{'='*70}")
    print(f"{'Metric':<25} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
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

            print(f"{metric:<25} {mean_val:>10.4f} {std_val:>10.4f} {min_val:>10.4f} {max_val:>10.4f}")

    print(f"{'='*70}")

    # Also aggregate silhouette score if present
    if 'silhouette_score' in all_results[0]:
        sil_values = [r['silhouette_score'] for r in all_results]
        aggregated['metrics']['silhouette_score'] = {
            'mean': float(np.mean(sil_values)),
            'std': float(np.std(sil_values)),
            'min': float(np.min(sil_values)),
            'max': float(np.max(sil_values)),
            'values': [float(v) for v in sil_values]
        }
        print(f"{'silhouette_score':<25} {np.mean(sil_values):>10.4f} {np.std(sil_values):>10.4f}")

    return aggregated


def main():
    parser = argparse.ArgumentParser(description='Aggregate test results across seeds')
    parser.add_argument('--results_dir', type=str, required=True,
                       help='Directory containing run output folders')
    parser.add_argument('--pattern', type=str, default='megaDNA_lambda*',
                       help='Glob pattern to match output folders (default: megaDNA_lambda*)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file for aggregated results (default: aggregated_results.json in results_dir)')

    args = parser.parse_args()

    aggregated = aggregate_results(args.results_dir, args.pattern)

    if aggregated:
        # Save aggregated results
        output_path = args.output or str(Path(args.results_dir) / 'aggregated_results.json')
        with open(output_path, 'w') as f:
            json.dump(aggregated, f, indent=2)
        print(f"\nAggregated results saved to: {output_path}")


if __name__ == "__main__":
    main()
