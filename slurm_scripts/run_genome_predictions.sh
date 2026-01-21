#!/bin/bash
#SBATCH --job-name=megadna_genome_pred
#SBATCH --time=8:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu

# Biowulf SLURM script for running megaDNA genome window predictions

# Load modules
#module load python/3.9
#module load cuda/11.3
source activate megadna

# Set paths
MEGADNA_MODEL="megaDNA_phage_145M.pt"
CLASSIFIER_MODEL="/data/lindseylm/GLM_EVALUATIONS/MODELS/MEGADNA/megaDNA/classification_results_20251118_113839/classifier.pt" 
CLASSIFIER_TYPE="neural"  # or "logistic"
INPUT_DIR="/data/lindseylm/gLMs/lambda/data/CSV/lambda_labeled_20251117_100334" 
OUTPUT_DIR="genome_predictions_$(date +%Y%m%d_%H%M%S)"

# Create output directory
mkdir -p $OUTPUT_DIR

echo "========================================================================"
echo "megaDNA Genome Window Prediction"
echo "========================================================================"
echo "megaDNA model:    $MEGADNA_MODEL"
echo "Classifier:       $CLASSIFIER_MODEL"
echo "Input directory:  $INPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "========================================================================"
echo ""

# Run predictions
python genome_predictions_megadna.py \
    --megadna_model $MEGADNA_MODEL \
    --classifier_model $CLASSIFIER_MODEL \
    --classifier_type $CLASSIFIER_TYPE \
    --input_dir $INPUT_DIR \
    --output_dir $OUTPUT_DIR \
    --batch_size 8 \
    --pooling mean \
    --cache_embeddings \
    --device cuda \
    2>&1 | tee $OUTPUT_DIR/prediction.log

echo ""
echo "Done! Predictions saved to: $OUTPUT_DIR"

# To run this script on Biowulf:
# 1. Update CLASSIFIER_MODEL path (from your training results)
# 2. Update INPUT_DIR path (where your genome CSVs are)
# 3. sbatch run_genome_predictions.sh
