#!/bin/bash
#SBATCH --job-name=megadna_classify
#SBATCH --time=4:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu

# Biowulf SLURM script for running phage/bacteria classification

# Load modules
#module load python/3.9
#module load cuda/11.3

source activate megadna

# Set paths
DATA_DIR="/data/lindseylm/GLM_EVALUATIONS/LAMBDA/CLEANED_DATA"
MODEL_PATH="megaDNA_phage_145M.pt"  # Update this path to where you put the model
OUTPUT_DIR="classification_results_$(date +%Y%m%d_%H%M%S)"

# Create output directory
mkdir -p $OUTPUT_DIR

# Run classification with improved 3-layer neural network
echo "Running embedding classifier with Improved 3-Layer Neural Network..."
python run_embedding_classifier.py \
    --data_dir $DATA_DIR \
    --model_path $MODEL_PATH \
    --classifier_type neural \
    --layer middle \
    --pooling mean \
    --batch_size 8 \
    --epochs 200 \
    --lr 0.001 \
    --hidden_dim1 512 \
    --hidden_dim2 256 \
    --hidden_dim3 128 \
    --dropout 0.4 \
    --weight_decay 1e-4 \
    --nn_batch_size 64 \
    --output_dir $OUTPUT_DIR \
    2>&1 | tee $OUTPUT_DIR/run.log

echo "Done! Results saved to: $OUTPUT_DIR"

# To run this script on Biowulf:
# 1. sbatch run_on_biowulf.sh
#
# To run interactively (for testing):
# 1. sinteractive --gres=gpu:v100:1 --mem=32G
# 2. bash run_on_biowulf.sh
