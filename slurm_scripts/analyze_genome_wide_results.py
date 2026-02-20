import json
import glob
import os
import math
import re
import pandas as pd
import numpy as np
from pathlib import Path

def collapse_overlapping_windows(predictions_df, window_size=2000, step_size=1000):
    """
    Collapse overlapping window predictions using majority voting
    
    Parameters:
    -----------
    predictions_df : DataFrame
        DataFrame with columns: start, end, prediction, score (optional), label
    window_size : int
        Size of each window (default: 2000)
    step_size : int
        Step size between windows (default: 1000)
    
    Returns:
    --------
    collapsed_df : DataFrame
        Non-overlapping segments with majority vote predictions
    """
    
    # Determine the genome range
    min_pos = predictions_df['start'].min()
    max_pos = predictions_df['end'].max()
    
    # Create non-overlapping segments of step_size
    segments = []
    
    for pos in range(min_pos, max_pos, step_size):
        seg_start = pos
        seg_end = min(pos + step_size, max_pos)
        
        # Find all windows that cover this segment
        overlapping = predictions_df[
            (predictions_df['start'] <= seg_start) & 
            (predictions_df['end'] >= seg_end)
        ]
        
        if len(overlapping) == 0:
            continue
        
        # Majority vote for prediction
        votes = overlapping['prediction'].sum()
        total_votes = len(overlapping)
        majority_pred = 1 if votes > total_votes / 2 else 0
        
        # Average the scores if available
        avg_score = overlapping['score'].mean() if 'score' in overlapping.columns else 0.5
        
        # Get the true label (should be same for all overlapping windows)
        true_label = overlapping['label'].iloc[0] if 'label' in overlapping.columns else 0
        
        segments.append({
            'start': seg_start,
            'end': seg_end,
            'prediction': majority_pred,
            'score': avg_score,
            'label': true_label,
            'num_votes': total_votes,
            'votes_for_phage': votes
        })
    
    return pd.DataFrame(segments)

def apply_phage_clustering_filter(predictions_df, merge_gap=3000, min_cluster_size=1000, window_size=5):
    """
    Apply clustering filter to reduce false positives and merge nearby phage predictions

    Parameters:
    -----------
    predictions_df : DataFrame
        DataFrame with columns: start, end, prediction (0 or 1), score (optional)
    merge_gap : int
        Maximum gap (nt) between segments to merge into same cluster (default: 3000)
    min_cluster_size : int
        Minimum total size (nt) for a phage cluster to be kept (default: 1000)
    window_size : int
        Window size for bidirectional smoothing (default: 5)
    verbose : bool
        Print detailed processing information (default: False)

    Returns:
    --------
    filtered_df : DataFrame
        DataFrame with filtered predictions
    phage_regions : list
        List of dicts with predicted phage region info (start, end, size, avg_score)
    """

    # Sort by start position
    df = predictions_df.sort_values('start').copy()

    # Apply bidirectional smoothing to predictions if we have scores
    if 'score' in df.columns:
        # Forward pass (left to right)
        forward_smooth = df['score'].ewm(span=window_size, adjust=False).mean()

        # Backward pass (right to left) - reverse, smooth, reverse back
        backward_smooth = df['score'][::-1].ewm(span=window_size, adjust=False).mean()[::-1]

        # Average both directions for bidirectional smoothing
        df['smoothed_score'] = (forward_smooth + backward_smooth) / 2

        # Update predictions based on smoothed scores (threshold 0.5)
        df['prediction'] = (df['smoothed_score'] >= 0.5).astype(int)

    # Filter to only phage predictions
    phage_df = df[df['prediction'] == 1].copy()

    if len(phage_df) == 0:
        return df, []  # No phage predicted, return original with empty regions

    # Cluster nearby phage segments
    clusters = []
    current_cluster = [phage_df.iloc[0]]

    for idx in range(1, len(phage_df)):
        prev_segment = current_cluster[-1]
        curr_segment = phage_df.iloc[idx]

        # Check if gap between segments is less than merge_gap
        gap = curr_segment['start'] - prev_segment['end']

        if gap <= merge_gap:
            current_cluster.append(curr_segment)
        else:
            # Save current cluster and start new one
            clusters.append(current_cluster)
            current_cluster = [curr_segment]

    # Don't forget the last cluster
    clusters.append(current_cluster)

    # Filter clusters by minimum size and mark segments
    valid_indices = set()
    phage_regions = []

    for cluster_idx, cluster in enumerate(clusters):
        cluster_start = cluster[0]['start']
        cluster_end = cluster[-1]['end']
        cluster_size = cluster_end - cluster_start

        if cluster_size >= min_cluster_size:
            # Keep all segments in this cluster
            for segment in cluster:
                valid_indices.add(segment.name)

            # Calculate average score for this cluster
            if 'smoothed_score' in df.columns:
                cluster_indices = [seg.name for seg in cluster]
                avg_score = df.loc[cluster_indices, 'smoothed_score'].mean()
            elif 'score' in df.columns:
                cluster_indices = [seg.name for seg in cluster]
                avg_score = df.loc[cluster_indices, 'score'].mean()
            else:
                avg_score = 1.0

            phage_regions.append({
                'cluster_id': cluster_idx + 1,
                'start': cluster_start,
                'end': cluster_end,
                'size': cluster_size,
                'num_segments': len(cluster),
                'avg_score': avg_score
            })

    # Update predictions: set to 0 if not in valid cluster
    df['filtered_prediction'] = df['prediction'].copy()
    df.loc[~df.index.isin(valid_indices), 'filtered_prediction'] = 0

    return df, phage_regions

def calculate_mcc(tp, tn, fp, fn):
    """Calculate Matthews Correlation Coefficient"""
    numerator = (tp * tn) - (fp * fn)
    
    # Use float to avoid overflow
    denominator_val = float(tp + fp) * float(tp + fn) * float(tn + fp) * float(tn + fn)
    
    if denominator_val <= 0:
        return 0.0
    
    denominator = math.sqrt(denominator_val)
    
    if denominator == 0:
        return 0.0
    return numerator / denominator

def calculate_metrics(tp, tn, fp, fn):
    """Calculate all metrics from confusion matrix"""
    total = tp + tn + fp + fn
    
    metrics = {}
    metrics['accuracy'] = (tp + tn) / total if total > 0 else 0
    metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
    metrics['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    metrics['fnr'] = fn / (fn + tp) if (fn + tp) > 0 else 0
    metrics['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0
    metrics['mcc'] = calculate_mcc(tp, tn, fp, fn)
    
    if metrics['precision'] + metrics['recall'] > 0:
        metrics['f1'] = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall'])
    else:
        metrics['f1'] = 0
    
    return metrics

def summarize_genome_predictions(directory_path, model_name, output_dir='.',
                                output_individual=None,
                                output_summary=None,
                                output_predictions=None,
                                merge_gap=3000,
                                min_cluster_size=1000,
                                window_size=5,
                                verbose=False):
    """
    Read all JSON prediction files from a directory and summarize results.
    Always computes both raw and filtered metrics for comparison.

    Parameters:
    -----------
    directory_path : str
        Path to directory containing JSON files
    model_name : str
        Name/identifier for this model (for comparing multiple models)
    output_dir : str
        Directory to save output CSV files (default: current directory)
    output_individual : str
        Output CSV filename for individual genome results (default: {model_name}_individual.csv)
    output_summary : str
        Output CSV filename for summary metrics (default: {model_name}_summary.csv)
    output_predictions : str
        Output CSV filename for predicted phage regions (default: {model_name}_phage_predictions.csv)
    merge_gap : int
        Maximum gap (nt) between segments to merge (default: 3000)
    min_cluster_size : int
        Minimum cluster size (nt) to keep (default: 1000)
    window_size : int
        Window size for bidirectional smoothing (default: 5)

    Output files contain both RAW (unfiltered) and FILTERED metrics for comparison.
    - Aggregate metrics: calculated from summed TP/TN/FP/FN across all genomes
    - Average metrics: mean of per-genome metrics
    """

    # Set default output filenames with model name if not provided
    if output_individual is None:
        output_individual = f"{model_name}_individual.csv"
    if output_summary is None:
        output_summary = f"{model_name}_summary.csv"
    if output_predictions is None:
        output_predictions = f"{model_name}_phage_predictions.csv"

    # Add output directory to paths
    output_individual = os.path.join(output_dir, output_individual)
    output_summary = os.path.join(output_dir, output_summary)
    output_predictions = os.path.join(output_dir, output_predictions)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Find all JSON files, excluding summary/aggregate files
    all_json_files = glob.glob(os.path.join(directory_path, '*.json'))

    # Filter out summary files (files with 'summary' in name or other aggregate files)
    json_files = [f for f in all_json_files if not any(skip in os.path.basename(f).lower()
                  for skip in ['summary', 'aggregate', 'combined', 'total'])]

    if not json_files:
        print(f"No JSON files found in {directory_path}")
        print(f"(Found {len(all_json_files)} JSON files but all were filtered as summary files)")
        return None, None, None

    print(f"Found {len(json_files)} genome JSON files (skipped {len(all_json_files) - len(json_files)} summary files)")

    # Store individual file results
    results = []

    # Store all phage predictions
    all_phage_predictions = []

    # Accumulators for aggregate calculation (raw)
    raw_total_tp = 0
    raw_total_tn = 0
    raw_total_fp = 0
    raw_total_fn = 0

    # Accumulators for aggregate calculation (filtered)
    filt_total_tp = 0
    filt_total_tn = 0
    filt_total_fp = 0
    filt_total_fn = 0

    total_samples = 0

    # Track CSV file matching
    csv_found_count = 0
    csv_missing_count = 0
    csv_missing_genomes = []

    # Process each JSON file
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            # Extract filename (without _metrics.json suffix for cleaner display)
            filename = os.path.basename(json_file)
            genome_name = filename.replace('_metrics.json', '').replace('.json', '')

            # Get raw metrics from JSON
            raw_tp = data.get('true_positives', 0)
            raw_tn = data.get('true_negatives', 0)
            raw_fp = data.get('false_positives', 0)
            raw_fn = data.get('false_negatives', 0)

            raw_metrics = calculate_metrics(raw_tp, raw_tn, raw_fp, raw_fn)

            # Initialize filtered metrics as same as raw (will update if CSV exists)
            filt_tp, filt_tn, filt_fp, filt_fn = raw_tp, raw_tn, raw_fp, raw_fn
            filt_metrics = raw_metrics.copy()
            phage_regions = []

            # Find corresponding CSV file for filtering
            # Try multiple naming patterns
            csv_file = None
            json_dir = os.path.dirname(json_file)
            json_basename = os.path.basename(json_file)

            # Pattern 1: _metrics.json -> _predictions.csv (e.g., genome_segments_metrics.json -> genome_segments_predictions.csv)
            candidate1 = json_file.replace('_metrics.json', '_predictions.csv')
            # Pattern 2: direct replacement (e.g., NC_003197_metrics.json -> NC_003197.csv)
            candidate2 = json_file.replace('_metrics.json', '.csv')
            # Pattern 3: extract genome ID and search for matching CSV
            # Genome IDs typically look like NC_XXXXXX, NZ_XXXXXX, GCF_XXXXXX, GCA_XXXXXX, etc.
            genome_id_match = re.match(r'((?:NC|NZ|GCF|GCA)_[A-Z0-9.]+)', json_basename)

            for candidate in [candidate1, candidate2]:
                if verbose:
                    print(f"  Trying CSV: {candidate} -> {'FOUND' if os.path.exists(candidate) else 'NOT FOUND'}")
                if os.path.exists(candidate):
                    csv_file = candidate
                    break

            # If still not found, search for CSV with matching genome ID
            if csv_file is None and genome_id_match:
                genome_id = genome_id_match.group(1)
                matching_csvs = glob.glob(os.path.join(json_dir, f"{genome_id}*predictions.csv"))
                # Filter out any summary CSVs
                matching_csvs = [f for f in matching_csvs if 'summary' not in f.lower()]
                if matching_csvs:
                    csv_file = matching_csvs[0]  # Take first match

            if csv_file and os.path.exists(csv_file):
                csv_found_count += 1
                if verbose:
                    print(f"\n{'='*60}")
                    print(f"Genome: {genome_name}")

                # Read the CSV file with predictions
                pred_df = pd.read_csv(csv_file)

                if verbose:
                    print(f"Total windows: {len(pred_df)}")

                if 'pred_label' in pred_df.columns:
                    phage_before = pred_df['pred_label'].sum()
                    if verbose:
                        print(f"Phage predictions (raw): {phage_before}")

                    # Rename columns to match expected format
                    pred_df = pred_df.rename(columns={'pred_label': 'prediction'})
                    if 'prob_1' in pred_df.columns:
                        pred_df = pred_df.rename(columns={'prob_1': 'score'})

                    # First, collapse overlapping windows with majority voting
                    collapsed_df = collapse_overlapping_windows(pred_df, window_size=2000, step_size=1000)

                    # Apply clustering filter
                    filtered_df, phage_regions = apply_phage_clustering_filter(
                        collapsed_df,
                        merge_gap=merge_gap,
                        min_cluster_size=min_cluster_size,
                        window_size=window_size
                    )

                    phage_after = filtered_df['filtered_prediction'].sum() if 'filtered_prediction' in filtered_df.columns else 0
                    if verbose:
                        print(f"Phage predictions (filtered): {phage_after}")
                        print(f"Phage regions found: {len(phage_regions)}")

                    # Recalculate confusion matrix from filtered predictions
                    if 'label' in filtered_df.columns and 'filtered_prediction' in filtered_df.columns:
                        filt_tp = int(((filtered_df['label'] == 1) & (filtered_df['filtered_prediction'] == 1)).sum())
                        filt_tn = int(((filtered_df['label'] == 0) & (filtered_df['filtered_prediction'] == 0)).sum())
                        filt_fp = int(((filtered_df['label'] == 0) & (filtered_df['filtered_prediction'] == 1)).sum())
                        filt_fn = int(((filtered_df['label'] == 1) & (filtered_df['filtered_prediction'] == 0)).sum())

                        filt_metrics = calculate_metrics(filt_tp, filt_tn, filt_fp, filt_fn)

                        if verbose:
                            print(f"Raw:      MCC={raw_metrics['mcc']:.4f}, Recall={raw_metrics['recall']:.4f}, FPR={raw_metrics['fpr']:.4f}")
                            print(f"Filtered: MCC={filt_metrics['mcc']:.4f}, Recall={filt_metrics['recall']:.4f}, FPR={filt_metrics['fpr']:.4f}")
                    else:
                        if verbose:
                            print(f"WARNING: Could not find 'label' column, using raw metrics for filtered")
                else:
                    if verbose:
                        print(f"WARNING: CSV file missing 'pred_label' column")
            else:
                csv_missing_count += 1
                csv_missing_genomes.append(genome_name)
                if verbose:
                    print(f"WARNING: CSV file not found for {genome_name}, using raw metrics only")

            # Add phage regions to the collection
            for region in phage_regions:
                all_phage_predictions.append({
                    'genome': genome_name,
                    'cluster_id': region['cluster_id'],
                    'start': region['start'],
                    'end': region['end'],
                    'size': region['size'],
                    'num_segments': region['num_segments'],
                    'avg_score': region['avg_score']
                })

            # Store individual result with both raw and filtered metrics
            num_samples = data.get('num_samples', raw_tp + raw_tn + raw_fp + raw_fn)
            results.append({
                'genome': genome_name,
                'filename': filename,
                'samples': num_samples,
                # Raw metrics
                'raw_tp': raw_tp,
                'raw_tn': raw_tn,
                'raw_fp': raw_fp,
                'raw_fn': raw_fn,
                'raw_accuracy': raw_metrics['accuracy'],
                'raw_precision': raw_metrics['precision'],
                'raw_recall': raw_metrics['recall'],
                'raw_specificity': raw_metrics['specificity'],
                'raw_fnr': raw_metrics['fnr'],
                'raw_fpr': raw_metrics['fpr'],
                'raw_mcc': raw_metrics['mcc'],
                'raw_f1': raw_metrics['f1'],
                # Filtered metrics
                'filt_tp': filt_tp,
                'filt_tn': filt_tn,
                'filt_fp': filt_fp,
                'filt_fn': filt_fn,
                'filt_accuracy': filt_metrics['accuracy'],
                'filt_precision': filt_metrics['precision'],
                'filt_recall': filt_metrics['recall'],
                'filt_specificity': filt_metrics['specificity'],
                'filt_fnr': filt_metrics['fnr'],
                'filt_fpr': filt_metrics['fpr'],
                'filt_mcc': filt_metrics['mcc'],
                'filt_f1': filt_metrics['f1'],
                # Number of predicted phage regions
                'num_phage_regions': len(phage_regions)
            })

            # Accumulate totals
            raw_total_tp += raw_tp
            raw_total_tn += raw_tn
            raw_total_fp += raw_fp
            raw_total_fn += raw_fn

            filt_total_tp += filt_tp
            filt_total_tn += filt_tn
            filt_total_fp += filt_fp
            filt_total_fn += filt_fn

            total_samples += num_samples

        except Exception as e:
            print(f"Error processing {json_file}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Create DataFrame with individual results
    df_individual = pd.DataFrame(results)

    # Create DataFrame with phage predictions
    df_predictions = pd.DataFrame(all_phage_predictions)

    # Calculate aggregate metrics (from summed confusion matrix)
    raw_aggregate = calculate_metrics(raw_total_tp, raw_total_tn, raw_total_fp, raw_total_fn)
    filt_aggregate = calculate_metrics(filt_total_tp, filt_total_tn, filt_total_fp, filt_total_fn)

    # Calculate averaged metrics (mean across genomes)
    raw_average = {
        'accuracy': df_individual['raw_accuracy'].mean(),
        'precision': df_individual['raw_precision'].mean(),
        'recall': df_individual['raw_recall'].mean(),
        'specificity': df_individual['raw_specificity'].mean(),
        'fnr': df_individual['raw_fnr'].mean(),
        'fpr': df_individual['raw_fpr'].mean(),
        'mcc': df_individual['raw_mcc'].mean(),
        'f1': df_individual['raw_f1'].mean()
    }

    filt_average = {
        'accuracy': df_individual['filt_accuracy'].mean(),
        'precision': df_individual['filt_precision'].mean(),
        'recall': df_individual['filt_recall'].mean(),
        'specificity': df_individual['filt_specificity'].mean(),
        'fnr': df_individual['filt_fnr'].mean(),
        'fpr': df_individual['filt_fpr'].mean(),
        'mcc': df_individual['filt_mcc'].mean(),
        'f1': df_individual['filt_f1'].mean()
    }

    # Create summary DataFrame with all four rows
    summary_data = [
        {
            'model': model_name,
            'type': 'raw',
            'method': 'aggregate (summed TP/TN/FP/FN)',
            'num_files': len(results),
            'total_samples': total_samples,
            'total_tp': raw_total_tp,
            'total_tn': raw_total_tn,
            'total_fp': raw_total_fp,
            'total_fn': raw_total_fn,
            'accuracy': raw_aggregate['accuracy'],
            'precision': raw_aggregate['precision'],
            'recall': raw_aggregate['recall'],
            'specificity': raw_aggregate['specificity'],
            'fnr': raw_aggregate['fnr'],
            'fpr': raw_aggregate['fpr'],
            'mcc': raw_aggregate['mcc'],
            'f1': raw_aggregate['f1']
        },
        {
            'model': model_name,
            'type': 'raw',
            'method': 'average (mean per-genome)',
            'num_files': len(results),
            'total_samples': total_samples,
            'total_tp': '',
            'total_tn': '',
            'total_fp': '',
            'total_fn': '',
            'accuracy': raw_average['accuracy'],
            'precision': raw_average['precision'],
            'recall': raw_average['recall'],
            'specificity': raw_average['specificity'],
            'fnr': raw_average['fnr'],
            'fpr': raw_average['fpr'],
            'mcc': raw_average['mcc'],
            'f1': raw_average['f1']
        },
        {
            'model': model_name,
            'type': 'filtered',
            'method': 'aggregate (summed TP/TN/FP/FN)',
            'num_files': len(results),
            'total_samples': total_samples,
            'total_tp': filt_total_tp,
            'total_tn': filt_total_tn,
            'total_fp': filt_total_fp,
            'total_fn': filt_total_fn,
            'accuracy': filt_aggregate['accuracy'],
            'precision': filt_aggregate['precision'],
            'recall': filt_aggregate['recall'],
            'specificity': filt_aggregate['specificity'],
            'fnr': filt_aggregate['fnr'],
            'fpr': filt_aggregate['fpr'],
            'mcc': filt_aggregate['mcc'],
            'f1': filt_aggregate['f1']
        },
        {
            'model': model_name,
            'type': 'filtered',
            'method': 'average (mean per-genome)',
            'num_files': len(results),
            'total_samples': total_samples,
            'total_tp': '',
            'total_tn': '',
            'total_fp': '',
            'total_fn': '',
            'accuracy': filt_average['accuracy'],
            'precision': filt_average['precision'],
            'recall': filt_average['recall'],
            'specificity': filt_average['specificity'],
            'fnr': filt_average['fnr'],
            'fpr': filt_average['fpr'],
            'mcc': filt_average['mcc'],
            'f1': filt_average['f1']
        }
    ]

    df_summary = pd.DataFrame(summary_data)

    # Save to CSV files
    df_individual.to_csv(output_individual, index=False)
    df_summary.to_csv(output_summary, index=False)
    df_predictions.to_csv(output_predictions, index=False)

    print(f"\nIndividual results saved to {output_individual}")
    print(f"Summary metrics saved to {output_summary}")
    print(f"Phage predictions saved to {output_predictions}")
    print(f"\nProcessed {len(results)} genomes")
    print(f"Total samples: {total_samples}")
    print(f"Total phage regions predicted: {len(all_phage_predictions)}")

    # Print CSV file matching summary
    print(f"\nCSV file matching: {csv_found_count}/{len(results)} found")
    if csv_missing_count > 0:
        print(f"  WARNING: {csv_missing_count} genomes missing CSV files (raw metrics used for filtered)")
        if csv_missing_count <= 10:
            print(f"  Missing: {', '.join(csv_missing_genomes)}")
        else:
            print(f"  First 10 missing: {', '.join(csv_missing_genomes[:10])}...")

    print(f"\n{'='*70}")
    print("SUMMARY COMPARISON: RAW vs FILTERED")
    print(f"{'='*70}")
    print(f"\n{'Metric':<15} {'Raw Aggregate':>15} {'Filt Aggregate':>15} {'Raw Average':>15} {'Filt Average':>15}")
    print(f"{'-'*75}")
    print(f"{'MCC':<15} {raw_aggregate['mcc']:>15.4f} {filt_aggregate['mcc']:>15.4f} {raw_average['mcc']:>15.4f} {filt_average['mcc']:>15.4f}")
    print(f"{'Recall':<15} {raw_aggregate['recall']:>15.4f} {filt_aggregate['recall']:>15.4f} {raw_average['recall']:>15.4f} {filt_average['recall']:>15.4f}")
    print(f"{'Precision':<15} {raw_aggregate['precision']:>15.4f} {filt_aggregate['precision']:>15.4f} {raw_average['precision']:>15.4f} {filt_average['precision']:>15.4f}")
    print(f"{'FPR':<15} {raw_aggregate['fpr']:>15.4f} {filt_aggregate['fpr']:>15.4f} {raw_average['fpr']:>15.4f} {filt_average['fpr']:>15.4f}")
    print(f"{'FNR':<15} {raw_aggregate['fnr']:>15.4f} {filt_aggregate['fnr']:>15.4f} {raw_average['fnr']:>15.4f} {filt_average['fnr']:>15.4f}")
    print(f"{'F1':<15} {raw_aggregate['f1']:>15.4f} {filt_aggregate['f1']:>15.4f} {raw_average['f1']:>15.4f} {filt_average['f1']:>15.4f}")

    return df_individual, df_summary, df_predictions

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Summarize genome-wide prediction metrics from JSON files. Always computes both raw and filtered metrics.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Output files:
  - {model_name}_individual.csv: Per-genome metrics (both raw and filtered)
  - {model_name}_summary.csv: Summary with 4 rows:
      * raw + aggregate (metrics from summed TP/TN/FP/FN across all genomes)
      * raw + average (mean of per-genome metrics)
      * filtered + aggregate
      * filtered + average
  - {model_name}_phage_predictions.csv: Predicted phage regions with start/end positions

Examples:
  # Analyze single model
  python analyze_genome_wide_results.py -d /path/to/json/files -m model1 -r results

  # Compare multiple models
  python analyze_genome_wide_results.py -d /path/to/model1 -m model1 -r results
  python analyze_genome_wide_results.py -d /path/to/model2 -m model2 -r results

  # Then combine summaries:
  python -c "import pandas as pd; pd.concat([pd.read_csv('results/model1_summary.csv'), pd.read_csv('results/model2_summary.csv')]).to_csv('results/comparison.csv', index=False)"
        """
    )

    parser.add_argument('-d', '--directory', required=True,
                        help='Directory containing JSON prediction files')
    parser.add_argument('-m', '--model-name', required=True,
                        help='Model identifier/name for this run')
    parser.add_argument('-r', '--output-dir', default='.',
                        help='Directory to save output CSV files (default: current directory)')
    parser.add_argument('-i', '--output-individual', default=None,
                        help='Output CSV filename for individual genome results (default: {model_name}_individual.csv)')
    parser.add_argument('-s', '--output-summary', default=None,
                        help='Output CSV filename for summary metrics (default: {model_name}_summary.csv)')
    parser.add_argument('-p', '--output-predictions', default=None,
                        help='Output CSV filename for phage predictions (default: {model_name}_phage_predictions.csv)')
    parser.add_argument('-o', '--output-prefix', default=None,
                        help='Prefix for all output files (overrides -i, -s, -p)')
    parser.add_argument('--merge-gap', type=int, default=3000,
                        help='Maximum gap (nt) between segments to merge into cluster (default: 3000)')
    parser.add_argument('--min-cluster-size', type=int, default=1000,
                        help='Minimum cluster size (nt) to keep (default: 1000)')
    parser.add_argument('--window-size', type=int, default=5,
                        help='Window size for bidirectional smoothing (default: 5)')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Print detailed processing information for each genome')

    args = parser.parse_args()

    # Handle output prefix if provided
    if args.output_prefix:
        output_individual = f"{args.output_prefix}_individual.csv"
        output_summary = f"{args.output_prefix}_summary.csv"
        output_predictions = f"{args.output_prefix}_phage_predictions.csv"
    else:
        output_individual = args.output_individual
        output_summary = args.output_summary
        output_predictions = args.output_predictions

    # Run the analysis
    df_individual, df_summary, df_predictions = summarize_genome_predictions(
        directory_path=args.directory,
        model_name=args.model_name,
        output_dir=args.output_dir,
        output_individual=output_individual,
        output_summary=output_summary,
        output_predictions=output_predictions,
        merge_gap=args.merge_gap,
        min_cluster_size=args.min_cluster_size,
        window_size=args.window_size,
        verbose=args.verbose
    )

    if df_individual is not None and df_summary is not None:
        # Display first few rows
        print("\n" + "="*70)
        print("INDIVIDUAL GENOME RESULTS (first 5):")
        print("="*70)
        # Show selected columns for readability
        display_cols = ['genome', 'samples', 'raw_mcc', 'raw_recall', 'raw_fpr',
                        'filt_mcc', 'filt_recall', 'filt_fpr', 'num_phage_regions']
        print(df_individual[display_cols].head().to_string(index=False))

        print("\n" + "="*70)
        print("SUMMARY METRICS:")
        print("="*70)
        print(df_summary.to_string(index=False))

        if df_predictions is not None and len(df_predictions) > 0:
            print("\n" + "="*70)
            print(f"PHAGE PREDICTIONS (first 10 of {len(df_predictions)}):")
            print("="*70)
            print(df_predictions.head(10).to_string(index=False))
