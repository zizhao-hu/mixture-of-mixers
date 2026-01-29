# Mixture of Mixers

This repository contains a PyTorch implementation of Diffusion Transformers (DiT) for training on personal devices.

Based on the [DiT paper](https://arxiv.org/abs/2212.09748) by Peebles & Xie.

## Model Variants

| Model | Depth | Hidden Size | Heads | Params |
|-------|-------|-------------|-------|--------|
| DiT-S/2 | 12 | 384 | 6 | ~33M |
| DiT-S/4 | 12 | 384 | 6 | ~33M |
| DiT-S/8 | 12 | 384 | 6 | ~33M |
| DiT-B/2 | 12 | 768 | 12 | ~130M |
| DiT-B/4 | 12 | 768 | 12 | ~130M |
| DiT-L/2 | 24 | 1024 | 16 | ~458M |
| DiT-XL/2 | 28 | 1152 | 16 | ~675M |

The `/2`, `/4`, `/8` suffix indicates patch size. Larger patch size = fewer tokens = faster training.

**Recommended for personal devices:** `DiT-S/4` or `DiT-S/8`

## Setup

```bash
# Create conda environment
conda env create -f environment.yml
conda activate DiT

# Or install with pip
pip install torch torchvision diffusers timm tqdm
```

## Datasets

### MS-COCO (Recommended for testing)

Download and prepare MS-COCO dataset:

```bash
# Download validation set only (5K images, ~1GB) - good for quick testing
python download_coco.py --output-dir ./data/coco --val-only

# Download full training set (~18GB) + validation
python download_coco.py --output-dir ./data/coco --include-val
```

### ImageNet

For ImageNet, organize your data in ImageFolder format:
```
imagenet/train/
├── n01440764/
│   ├── image1.JPEG
│   └── ...
└── ...
```

## Training (Single GPU)

### On MS-COCO (80 classes)

```bash
# Quick test with COCO validation set
python train_single_gpu.py --data-path ./data/coco/imagefolder/val --num-classes 80 --model DiT-S/4 --batch-size 16 --mixed-precision

# Full training on COCO train set
python train_single_gpu.py --data-path ./data/coco/imagefolder/train --num-classes 80 --model DiT-S/4 --batch-size 16 --epochs 100 --mixed-precision
```

### On ImageNet (1000 classes)

```bash
# Train DiT-S/4 (recommended for personal devices)
python train_single_gpu.py --data-path /path/to/imagenet/train --model DiT-S/4 --batch-size 16

# With mixed precision (faster, less memory)
python train_single_gpu.py --data-path /path/to/imagenet/train --model DiT-S/4 --batch-size 32 --mixed-precision

# Resume from checkpoint
python train_single_gpu.py --data-path /path/to/imagenet/train --resume results/000-DiT-S-4/checkpoints/0010000.pt
```

### Dataset Structure

Your dataset should follow ImageFolder structure:
```
data/
├── class1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── class2/
│   ├── image1.jpg
│   └── ...
└── ...
```

### Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data-path` | required | Path to ImageFolder dataset |
| `--model` | DiT-S/4 | Model variant |
| `--image-size` | 256 | Image resolution (256 or 512) |
| `--num-classes` | 1000 | Number of classes |
| `--batch-size` | 16 | Batch size |
| `--epochs` | 100 | Number of training epochs |
| `--lr` | 1e-4 | Learning rate |
| `--mixed-precision` | False | Use FP16 mixed precision |
| `--ckpt-every` | 10000 | Save checkpoint every N steps |

## Training (Multi-GPU with DDP)

For distributed training on multiple GPUs:

```bash
torchrun --nnodes=1 --nproc_per_node=N train.py --model DiT-S/4 --data-path /path/to/data
```

## Multi-GPU Training (SLURM / HPC Clusters)

### Setup

```bash
# Clone repo and setup environment
git clone <your-repo-url> mixture-of-mixers
cd mixture-of-mixers
bash setup_env.sh
```

### Prepare Dataset

```bash
# Download COCO (full train + val, ~20GB)
python download_coco.py --output-dir ./data/coco --include-val
```

### Submit Training Job

```bash
# Edit submit_slurm.sh first:
# 1. Set DATA_PATH to your dataset location
# 2. Set NUM_CLASSES (80 for COCO, 1000 for ImageNet)
# 3. Adjust SLURM directives for your cluster

# Submit job
sbatch submit_slurm.sh

# Monitor job
squeue -u $USER
tail -f logs/dit_<jobid>.out
```

### Or Run Locally with Multiple GPUs

```bash
# 2 GPUs
torchrun --nproc_per_node=2 train_ddp.py \
    --data-path ./data/coco/imagefolder/train \
    --model DiT-S/4 \
    --num-classes 80 \
    --global-batch-size 128
```

### Expected Training Time (2x A100)

| Dataset | Model | Epochs | Time |
|---------|-------|--------|------|
| COCO train (118K) | DiT-S/4 | 400 | ~16-24 hours |
| COCO train (118K) | DiT-S/2 | 400 | ~2-3 days |
| ImageNet (1.28M) | DiT-S/4 | 400 | ~4-5 days |

### Output Structure

Training automatically generates samples at every checkpoint:

```
results/000-DiT-S-4/
├── log.txt                     # Training log
├── checkpoints/
│   ├── 0010000.pt             # Checkpoint at step 10K
│   ├── 0020000.pt             # Checkpoint at step 20K
│   └── final.pt               # Final checkpoint
└── samples/
    ├── step_0010000.png       # Samples at step 10K
    ├── step_0020000.png       # Samples at step 20K
    └── final.png              # Final samples
```

## Sampling

Generate images from a trained model:

```bash
python sample.py --model DiT-S/4 --ckpt /path/to/checkpoint.pt --image-size 256
```

## References

```bibtex
@article{Peebles2022DiT,
  title={Scalable Diffusion Models with Transformers},
  author={William Peebles and Saining Xie},
  year={2022},
  journal={arXiv preprint arXiv:2212.09748},
}
```

## Acknowledgments

This codebase is based on [facebookresearch/DiT](https://github.com/facebookresearch/DiT).
