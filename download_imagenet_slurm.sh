#!/bin/bash
#SBATCH --job-name=imagenet-download
#SBATCH --output=logs/imagenet_download_%j.out
#SBATCH --error=logs/imagenet_download_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --account=jessetho_1732

# ImageNet Download via SLURM Job
# This downloads ImageNet to scratch storage

SCRATCH_DIR="/scratch1/${USER}"
IMAGENET_DIR="${SCRATCH_DIR}/imagenet"
PROJECT_DIR="/project2/jessetho_1732/zizhaoh/mixture-of-mixers"

echo "============================================"
echo "ImageNet Download Job"
echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo "Output: ${IMAGENET_DIR}"
echo "============================================"

# Create directories
mkdir -p "${IMAGENET_DIR}"
mkdir -p "${PROJECT_DIR}/logs"

# Load environment
module load conda
source activate dit

# Install required packages
pip install --quiet datasets huggingface_hub tqdm Pillow

# Check authentication
echo "Checking Hugging Face authentication..."
if ! python -c "from huggingface_hub import HfApi; HfApi().whoami()" 2>/dev/null; then
    echo "ERROR: Not authenticated with Hugging Face!"
    echo "Please run: huggingface-cli login"
    echo "Before submitting this job."
    exit 1
fi

echo "Authentication OK"
python -c "from huggingface_hub import HfApi; print(f'Logged in as: {HfApi().whoami()[\"name\"]}')"

# Run download
cd "${PROJECT_DIR}"
python download_imagenet.py \
    --output-dir "${IMAGENET_DIR}" \
    --method huggingface \
    --split all

# Create symlink
echo "Creating symlink..."
mkdir -p "${PROJECT_DIR}/data"
ln -sfn "${IMAGENET_DIR}" "${PROJECT_DIR}/data/imagenet"

# Verify
echo "Verifying download..."
python download_imagenet.py --method verify --output-dir "${IMAGENET_DIR}"

echo "============================================"
echo "Download Complete!"
echo "End: $(date)"
echo "============================================"
