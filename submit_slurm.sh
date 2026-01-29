#!/bin/bash
#SBATCH --job-name=dit-s4-imagenet
#SBATCH --output=logs/dit_%j.out
#SBATCH --error=logs/dit_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=a100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --account=jessetho_1732

# ==============================================================================
# DiT-S/4 Training on ImageNet
# Official settings from: https://github.com/facebookresearch/DiT
# ==============================================================================

# Configuration
DATA_PATH="/project2/jessetho_1732/zizhaoh/mixture-of-mixers/data/imagenet/train"
NUM_CLASSES=1000
MODEL="DiT-S/4"
IMAGE_SIZE=256
GLOBAL_BATCH_SIZE=256
CKPT_EVERY=50000
LOG_EVERY=1000

# Setup
cd /project2/jessetho_1732/zizhaoh/mixture-of-mixers
mkdir -p logs

# Fix for SLURM GPU visibility
export CUDA_VISIBLE_DEVICES=0,1
export OPENBLAS_NUM_THREADS=4
export OMP_NUM_THREADS=4

module load conda
source activate dit

echo "============================================"
echo "DiT Training - 1 Node x 2 A100"
echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""
echo "Model: $MODEL (${IMAGE_SIZE}x${IMAGE_SIZE})"
echo "Global batch size: $GLOBAL_BATCH_SIZE"
echo "============================================"

# Single-node DDP training
torchrun \
    --nproc_per_node=2 \
    --master_addr=localhost \
    --master_port=29500 \
    train_ddp.py \
    --data-path "$DATA_PATH" \
    --model $MODEL \
    --image-size $IMAGE_SIZE \
    --num-classes $NUM_CLASSES \
    --global-batch-size $GLOBAL_BATCH_SIZE \
    --epochs 80 \
    --ckpt-every $CKPT_EVERY \
    --log-every $LOG_EVERY \
    --num-workers 8 \
    --sample-steps 250 \
    --cfg-scale 1.0 \
    --num-sample-images 16

echo "============================================"
echo "Training Complete: $(date)"
echo "============================================"
