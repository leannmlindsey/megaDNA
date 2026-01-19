#!/usr/bin/env python
"""
Calculate performance metrics from ProkBERT predictions
Outputs per-genome metrics and overall averages
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    confusion_matrix
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate metrics from prediction results"
    )
    
    parser.add_argument(
        "--predictions_dir",
        type=str,
        required=True,
        help="Directory containing prediction CSV files"
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="metrics_per_genome.csv",
        help="Output CSV file for per-genome metrics"
    )
    parser.add_argument(
        "--summary_csv",
        type=str,
        default="metrics_summary.csv",
        help="Output CSV file for summary statistics"
    )
    
    return parser.parse_args()


def calculate_metrics(labels, predictions):
    """Calculate all metrics for a set of predictions"""
    
    # Basic metrics
    accuracy = accuracy_score(labels, predictions)
    mcc = matthews_corrcoef(labels, predictions)
    
    # Precision, recall, F1
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary', zero_division=0
    )
    
    # Confusion matrix for sensitivity/specificity
    tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'mcc': mcc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'total_sequences': len(labels),
        'total_phage': int(sum(labels)),
        'total_bacteria': int(len(labels) - sum(labels)),
        'predicted_phage': int(sum(predictions)),
        'predicted_bacteria': int(len(predictions) - sum(predictions))
    }


def process_prediction_file(csv_path):
    """Process a single prediction CSV file"""
    
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"  ✗ Error reading {csv_path.name}: {e}")
        return None
    
    # Validate columns
    if 'label' not in df.columns or 'predicted_label' not in df.columns:
        print(f"  ✗ Missing required columns in {csv_path.name}")
        return None
    
    # Extract genome ID from filename
    # Assumes format: GCF_000007565.2_ASM756v2_genomic.csv
    genome_id = csv_path.stem  # Remove .csv extension
    
    # Calculate metrics
    metrics = calculate_metrics(df['label'].values, df['predicted_label'].values)
    metrics['genome_id'] = genome_id
    metrics['filename'] = csv_path.name
    
    return metrics


def main():
    args = parse_args()
    
    print("="*80)
    print("ProkBERT Metrics Calculator")
    print("="*80)
    print(f"Predictions directory: {args.predictions_dir}")
    print(f"Output CSV:            {args.output_csv}")
    print(f"Summary CSV:           {args.summary_csv}")
    print("="*80)
    print()
    
    # Validate paths
    pred_dir = Path(args.predictions_dir)
    
    if not pred_dir.exists():
        print(f"Error: Predictions directory does not exist: {pred_dir}")
        sys.exit(1)
    
    # Find all CSV files
    csv_files = sorted(pred_dir.glob("*.csv"))
    
    if not csv_files:
        print(f"Error: No CSV files found in {pred_dir}")
        sys.exit(1)
    
    print(f"Found {len(csv_files)} CSV files to process")
    print()
    
    # Process each file
    all_metrics = []
    all_labels = []
    all_predictions = []
    
    print("Processing files...")
    for csv_path in csv_files:
        metrics = process_prediction_file(csv_path)
        if metrics:
            all_metrics.append(metrics)
            
            # Also collect for overall calculation
            df = pd.read_csv(csv_path)
            all_labels.extend(df['label'].values)
            all_predictions.extend(df['predicted_label'].values)
            
            print(f"  ✓ {csv_path.name:50s} | MCC: {metrics['mcc']:.4f} | Acc: {metrics['accuracy']:.4f}")
    
    print()
    print(f"Successfully processed {len(all_metrics)} files")
    print()
    
    if not all_metrics:
        print("Error: No valid metrics calculated")
        sys.exit(1)
    
    # Create per-genome dataframe
    df_metrics = pd.DataFrame(all_metrics)
    
    # Reorder columns for better readability
    column_order = [
        'genome_id',
        'filename',
        'total_sequences',
        'total_phage',
        'total_bacteria',
        'predicted_phage',
        'predicted_bacteria',
        'accuracy',
        'mcc',
        'precision',
        'recall',
        'f1',
        'sensitivity',
        'specificity',
        'true_positives',
        'true_negatives',
        'false_positives',
        'false_negatives'
    ]
    
    df_metrics = df_metrics[column_order]
    
    # Save per-genome metrics
    df_metrics.to_csv(args.output_csv, index=False)
    print(f"✓ Per-genome metrics saved to: {args.output_csv}")
    
    # Calculate overall metrics (micro-average - all sequences together)
    overall_metrics = calculate_metrics(
        np.array(all_labels), 
        np.array(all_predictions)
    )
    
    # Calculate average metrics (macro-average - average of per-genome metrics)
    avg_metrics = {
        'metric': ['accuracy', 'mcc', 'precision', 'recall', 'f1', 'sensitivity', 'specificity'],
        'overall': [
            overall_metrics['accuracy'],
            overall_metrics['mcc'],
            overall_metrics['precision'],
            overall_metrics['recall'],
            overall_metrics['f1'],
            overall_metrics['sensitivity'],
            overall_metrics['specificity']
        ],
        'average_per_genome': [
            df_metrics['accuracy'].mean(),
            df_metrics['mcc'].mean(),
            df_metrics['precision'].mean(),
            df_metrics['recall'].mean(),
            df_metrics['f1'].mean(),
            df_metrics['sensitivity'].mean(),
            df_metrics['specificity'].mean()
        ],
        'std_per_genome': [
            df_metrics['accuracy'].std(),
            df_metrics['mcc'].std(),
            df_metrics['precision'].std(),
            df_metrics['recall'].std(),
            df_metrics['f1'].std(),
            df_metrics['sensitivity'].std(),
            df_metrics['specificity'].std()
        ],
        'min_per_genome': [
            df_metrics['accuracy'].min(),
            df_metrics['mcc'].min(),
            df_metrics['precision'].min(),
            df_metrics['recall'].min(),
            df_metrics['f1'].min(),
            df_metrics['sensitivity'].min(),
            df_metrics['specificity'].min()
        ],
        'max_per_genome': [
            df_metrics['accuracy'].max(),
            df_metrics['mcc'].max(),
            df_metrics['precision'].max(),
            df_metrics['recall'].max(),
            df_metrics['f1'].max(),
            df_metrics['sensitivity'].max(),
            df_metrics['specificity'].max()
        ]
    }
    
    df_summary = pd.DataFrame(avg_metrics)
    
    # Add total counts
    summary_info = pd.DataFrame([{
        'metric': 'total_genomes',
        'overall': len(all_metrics),
        'average_per_genome': len(all_metrics),
        'std_per_genome': 0,
        'min_per_genome': 0,
        'max_per_genome': 0
    }, {
        'metric': 'total_sequences',
        'overall': overall_metrics['total_sequences'],
        'average_per_genome': df_metrics['total_sequences'].mean(),
        'std_per_genome': df_metrics['total_sequences'].std(),
        'min_per_genome': df_metrics['total_sequences'].min(),
        'max_per_genome': df_metrics['total_sequences'].max()
    }, {
        'metric': 'total_phage',
        'overall': overall_metrics['total_phage'],
        'average_per_genome': df_metrics['total_phage'].mean(),
        'std_per_genome': df_metrics['total_phage'].std(),
        'min_per_genome': df_metrics['total_phage'].min(),
        'max_per_genome': df_metrics['total_phage'].max()
    }, {
        'metric': 'total_bacteria',
        'overall': overall_metrics['total_bacteria'],
        'average_per_genome': df_metrics['total_bacteria'].mean(),
        'std_per_genome': df_metrics['total_bacteria'].std(),
        'min_per_genome': df_metrics['total_bacteria'].min(),
        'max_per_genome': df_metrics['total_bacteria'].max()
    }])
    
    df_summary = pd.concat([summary_info, df_summary], ignore_index=True)
    
    # Save summary
    df_summary.to_csv(args.summary_csv, index=False)
    print(f"✓ Summary metrics saved to:    {args.summary_csv}")
    print()
    
    # Print summary to console
    print("="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print()
    
    print(f"Total genomes processed:  {len(all_metrics)}")
    print(f"Total sequences:          {overall_metrics['total_sequences']:,}")
    print(f"  Phage:                  {overall_metrics['total_phage']:,} ({overall_metrics['total_phage']/overall_metrics['total_sequences']*100:.1f}%)")
    print(f"  Bacteria:               {overall_metrics['total_bacteria']:,} ({overall_metrics['total_bacteria']/overall_metrics['total_sequences']*100:.1f}%)")
    print()
    
    print("OVERALL METRICS (all sequences combined):")
    print("-"*80)
    print(f"  Accuracy:    {overall_metrics['accuracy']:.4f}")
    print(f"  MCC:         {overall_metrics['mcc']:.4f}")
    print(f"  Precision:   {overall_metrics['precision']:.4f}")
    print(f"  Recall:      {overall_metrics['recall']:.4f}")
    print(f"  F1:          {overall_metrics['f1']:.4f}")
    print(f"  Sensitivity: {overall_metrics['sensitivity']:.4f}")
    print(f"  Specificity: {overall_metrics['specificity']:.4f}")
    print()
    
    print("AVERAGE PER-GENOME METRICS (mean ± std):")
    print("-"*80)
    print(f"  Accuracy:    {df_metrics['accuracy'].mean():.4f} ± {df_metrics['accuracy'].std():.4f}")
    print(f"  MCC:         {df_metrics['mcc'].mean():.4f} ± {df_metrics['mcc'].std():.4f}")
    print(f"  Precision:   {df_metrics['precision'].mean():.4f} ± {df_metrics['precision'].std():.4f}")
    print(f"  Recall:      {df_metrics['recall'].mean():.4f} ± {df_metrics['recall'].std():.4f}")
    print(f"  F1:          {df_metrics['f1'].mean():.4f} ± {df_metrics['f1'].std():.4f}")
    print(f"  Sensitivity: {df_metrics['sensitivity'].mean():.4f} ± {df_metrics['sensitivity'].std():.4f}")
    print(f"  Specificity: {df_metrics['specificity'].mean():.4f} ± {df_metrics['specificity'].std():.4f}")
    print()
    
    print("FOR YOUR PAPER:")
    print("-"*80)
    print("Overall (all sequences):")
    print(f"  MCC:       {overall_metrics['mcc']:.4f}")
    print(f"  Precision: {overall_metrics['precision']:.4f}")
    print(f"  Recall:    {overall_metrics['recall']:.4f}")
    print()
    print("Average per genome:")
    print(f"  MCC:       {df_metrics['mcc'].mean():.4f} ± {df_metrics['mcc'].std():.4f}")
    print(f"  Precision: {df_metrics['precision'].mean():.4f} ± {df_metrics['precision'].std():.4f}")
    print(f"  Recall:    {df_metrics['recall'].mean():.4f} ± {df_metrics['recall'].std():.4f}")
    print()
    
    print("="*80)
    print()


if __name__ == "__main__":
    main()
