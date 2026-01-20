#!/bin/bash
#SBATCH --job-name=megadna_linear_probe
#SBATCH --time=2:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu

# SLURM script for running linear probe (logistic regression) on MegaDNA embeddings
# This demonstrates embedding quality - high accuracy with linear classifier = strong embeddings

# Parse command line arguments
SEED=${1:-42}  # Default seed is 42 if not provided

# Load modules
#module load python/3.9
#module load cuda/11.3

source activate megadna

# Set paths
DATA_DIR="/home/lindseylm/lindseylm/lambda_final/merged_datasets_filtered/2k"
MODEL_PATH="megaDNA_phage_145M.pt"
f="filtered"
len="2k"
OUTPUT_DIR="./output/lambda_filtered/2k/megaDNA_linear_probe_${f}_${len}_seed${SEED}_$(date +%Y%m%d_%H%M%S)"

# Create output directory
mkdir -p $OUTPUT_DIR

# Run linear probe (logistic regression) classifier
echo "Running Linear Probe (Logistic Regression) on MegaDNA Embeddings..."
echo "Using seed: $SEED"
python run_embedding_classifier.py \
    --data_dir $DATA_DIR \
    --model_path $MODEL_PATH \
    --classifier_type logistic \
    --layer middle \
    --pooling mean \
    --batch_size 8 \
    --output_dir $OUTPUT_DIR \
    --seed $SEED \
    2>&1 | tee $OUTPUT_DIR/run.log

echo "Done! Results saved to: $OUTPUT_DIR"

# Usage: bash run_linear_probe_megaDNA_lambda.sh [SEED]
# Examples:
#   bash run_linear_probe_megaDNA_lambda.sh        # Uses default seed 42
#   bash run_linear_probe_megaDNA_lambda.sh 123    # Uses seed 123
#
# To run with multiple seeds:
#   for seed in 1 2 3 4 5 6 7 8 9 10; do
#       sbatch run_linear_probe_megaDNA_lambda.sh $seed
#   done
#
# To run interactively (for testing):
# 1. sinteractive --gres=gpu:v100:1 --mem=32G
# 2. bash run_linear_probe_megaDNA_lambda.sh [SEED]
