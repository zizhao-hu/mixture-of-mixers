# Mixture of Mixers (MoM)

A diffusion model architecture that replaces attention with **Mixture of Experts (MoE)** style token and channel mixers.

## Architecture

MoM replaces the attention mechanism in DiT with a mixture of two types of experts:

- **Token Mixers**: 2-layer FFN that mixes across the spatial/token dimension (N)
- **Channel Mixers**: 2-layer FFN that mixes across the feature/channel dimension (D)

A router selects top-k experts (default k=2) from the pool, and their outputs are combined via weighted sum.

```
Input: (B, N, D)
    │
    ├─► Router ──► Select top-k experts
    │
    ├─► Token Mixer 1 ──┐
    ├─► Token Mixer 2 ──┤
    ├─► Token Mixer 3 ──┼──► Weighted Sum ──► Output: (B, N, D)
    ├─► Token Mixer 4 ──┤
    ├─► Channel Mixer 1 ┤
    ├─► Channel Mixer 2 ┤
    ├─► Channel Mixer 3 ┤
    └─► Channel Mixer 4 ┘
```

## Installation

```bash
pip install torch torchvision timm diffusers
```

## Models

| Model | Depth | Hidden Size | Patch Size |
|-------|-------|-------------|------------|
| MoM-S/4 | 12 | 384 | 4 |
| MoM-B/4 | 12 | 768 | 4 |
| MoM-L/4 | 24 | 1024 | 4 |
| MoM-XL/4 | 28 | 1152 | 4 |

Also includes baseline DiT models for comparison.

## Training

### Single GPU
```bash
python train.py --data-path /path/to/imagenet/train --model MoM-S/4
```

### Multi-GPU (DDP)
```bash
torchrun --nproc_per_node=2 train.py \
    --data-path /path/to/imagenet/train \
    --model MoM-S/4 \
    --global-batch-size 256
```

### SLURM Cluster
```bash
sbatch submit_slurm.sh
```

## Sampling

```bash
python sample.py --model MoM-S/4 --ckpt results/000-MoM-S-4/checkpoints/final.pt
```

## Project Structure

```
mixture-of-mixers/
├── models/
│   ├── __init__.py     # Model registry
│   ├── common.py       # Shared components (embeddings, pos encoding)
│   ├── dit.py          # DiT baseline
│   └── mom.py          # Mixture of Mixers
├── diffusion/          # Diffusion utilities
├── train.py            # Unified training script
├── sample.py           # Sampling script
└── submit_slurm.sh     # SLURM job submission
```

## References

- [DiT: Scalable Diffusion Models with Transformers](https://github.com/facebookresearch/DiT)
- [MoE: Mixture of Experts](https://arxiv.org/abs/1701.06538)
- [MLP-Mixer](https://arxiv.org/abs/2105.01601)

## License

See [LICENSE](LICENSE) for details.
