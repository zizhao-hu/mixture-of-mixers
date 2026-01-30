#!/bin/bash
#SBATCH --job-name=vit-benchmark
#SBATCH --output=logs/cls_%j.out
#SBATCH --error=logs/cls_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --account=jessetho_1732

# ==============================================================================
# ViT vs ViT-MoM Classification Benchmark
# ==============================================================================

# Configuration
DATASET=${1:-cifar10}  # cifar10 or cifar100
MODEL=${2:-ViT-S/4}    # Model to run

# Setup
cd /project2/jessetho_1732/zizhaoh/mixture-of-mixers
mkdir -p logs results_cls

export CUDA_VISIBLE_DEVICES=0

module load conda
source activate dit

echo "============================================"
echo "ViT vs ViT-MoM Benchmark"
echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""
echo "Dataset: $DATASET"
echo "Model: $MODEL"
echo "============================================"

# Run training
python train_cls.py \
    --dataset $DATASET \
    --model "$MODEL" \
    --epochs 200 \
    --batch-size 128 \
    --lr 1e-3 \
    --num-workers 4 \
    --results-dir ./results_cls

echo "============================================"
echo "Training Complete: $(date)"
echo "============================================"
