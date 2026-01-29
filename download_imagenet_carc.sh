#!/bin/bash
# =============================================================================
# ImageNet Download Script for CARC Transfer Node
# =============================================================================
# 
# IMPORTANT: Run this on the data transfer node, NOT compute nodes!
#
# Usage:
#   1. SSH to transfer node:  ssh hpc-transfer1.usc.edu
#   2. Run this script:       bash download_imagenet_carc.sh
#
# Prerequisites:
#   - Hugging Face account with ImageNet-1k access
#   - HF token (get from https://huggingface.co/settings/tokens)
# =============================================================================

set -e

# Configuration
SCRATCH_DIR="/scratch1/${USER}"
IMAGENET_DIR="${SCRATCH_DIR}/imagenet"
PROJECT_DIR="/project2/jessetho_1732/zizhaoh/mixture-of-mixers"

echo "============================================"
echo "ImageNet Download for DiT Training"
echo "============================================"
echo "User: ${USER}"
echo "Scratch: ${SCRATCH_DIR}"
echo "ImageNet will be saved to: ${IMAGENET_DIR}"
echo "============================================"
echo ""

# Check we're on transfer node
HOSTNAME=$(hostname)
if [[ ! "$HOSTNAME" =~ "transfer" ]]; then
    echo "WARNING: You don't appear to be on a transfer node (hostname: $HOSTNAME)"
    echo "For large downloads, please use: ssh hpc-transfer1.usc.edu"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Create directories
mkdir -p "${IMAGENET_DIR}"
mkdir -p "${SCRATCH_DIR}/logs"

# Load conda and activate environment
echo "Loading conda environment..."
module load conda
source activate dit 2>/dev/null || {
    echo "Creating dit environment..."
    conda create -n dit python=3.11 -y
    source activate dit
}

# Install required packages
echo "Installing required packages..."
pip install --quiet datasets huggingface_hub tqdm Pillow

# Check HuggingFace authentication
echo ""
echo "Checking Hugging Face authentication..."
if ! python -c "from huggingface_hub import HfApi; HfApi().whoami()" 2>/dev/null; then
    echo ""
    echo "============================================"
    echo "Hugging Face Login Required"
    echo "============================================"
    echo ""
    echo "1. Get ImageNet access at:"
    echo "   https://huggingface.co/datasets/imagenet-1k"
    echo ""
    echo "2. Create an access token at:"
    echo "   https://huggingface.co/settings/tokens"
    echo ""
    echo "3. Enter your token below:"
    echo ""
    huggingface-cli login
fi

# Verify authentication
echo ""
echo "Verifying authentication..."
python -c "from huggingface_hub import HfApi; print(f'Logged in as: {HfApi().whoami()[\"name\"]}')"

# Start download
echo ""
echo "============================================"
echo "Starting ImageNet Download"
echo "============================================"
echo "This will take several hours (~150GB download)"
echo "Output: ${IMAGENET_DIR}"
echo ""

cd "${PROJECT_DIR}"

# Download using the Python script with nohup for long-running process
nohup python download_imagenet.py \
    --output-dir "${IMAGENET_DIR}" \
    --method huggingface \
    --split all \
    > "${SCRATCH_DIR}/logs/imagenet_download.log" 2>&1 &

DOWNLOAD_PID=$!
echo "Download started in background (PID: ${DOWNLOAD_PID})"
echo "Log file: ${SCRATCH_DIR}/logs/imagenet_download.log"
echo ""
echo "To monitor progress:"
echo "  tail -f ${SCRATCH_DIR}/logs/imagenet_download.log"
echo ""
echo "To check if still running:"
echo "  ps aux | grep ${DOWNLOAD_PID}"
echo ""

# Create symlink
echo "Creating symlink to project directory..."
mkdir -p "${PROJECT_DIR}/data"
ln -sfn "${IMAGENET_DIR}" "${PROJECT_DIR}/data/imagenet"
echo "Symlink created: ${PROJECT_DIR}/data/imagenet -> ${IMAGENET_DIR}"

echo ""
echo "============================================"
echo "Download Started!"
echo "============================================"
echo ""
echo "Next steps after download completes:"
echo ""
echo "1. Verify the download:"
echo "   python download_imagenet.py --method verify --output-dir ${IMAGENET_DIR}"
echo ""
echo "2. Update submit_slurm.sh:"
echo "   DATA_PATH=\"${IMAGENET_DIR}/train\""
echo "   NUM_CLASSES=1000"
echo ""
echo "3. Submit training job:"
echo "   sbatch submit_slurm.sh"
echo ""
