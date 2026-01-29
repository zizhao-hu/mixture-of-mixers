#!/bin/bash
# ============================================================================
# Setup script for DiT training environment
# Works on most HPC clusters and local machines
# ============================================================================

echo "Setting up DiT environment..."

# Optional: Load modules (uncomment and adjust for your cluster)
# module purge
# module load cuda/11.8
# module load anaconda3

# Create conda environment
echo "Creating conda environment 'dit'..."
conda create -n dit python=3.10 -y
source activate dit || conda activate dit

# Install PyTorch with CUDA support
echo "Installing PyTorch..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
echo "Installing dependencies..."
pip install diffusers timm tqdm accelerate

# Create necessary directories
mkdir -p logs
mkdir -p results

echo "============================================"
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. Prepare your dataset (COCO or ImageNet) in ImageFolder format"
echo "   python download_coco.py --output-dir ./data/coco --include-val"
echo ""
echo "2. Edit submit_slurm.sh:"
echo "   - Set DATA_PATH to your dataset location"
echo "   - Set NUM_CLASSES (80 for COCO, 1000 for ImageNet)"
echo "   - Adjust SLURM directives for your cluster"
echo ""
echo "3. Submit the job:"
echo "   sbatch submit_slurm.sh"
echo ""
echo "Or run locally with:"
echo "   torchrun --nproc_per_node=2 train_ddp.py --data-path ./data --model DiT-S/4"
echo "============================================"
