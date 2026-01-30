# Sampling script for DiT and DiT-MoM models.
#
# Usage:
#   python sample.py --model DiT-S/2 --ckpt results/000-DiT-S-2/checkpoints/final.pt
#   python sample.py --model DiT-MoM-S/2 --ckpt results/001-DiT-MoM-S-2/checkpoints/final.pt

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from torchvision.utils import save_image
import argparse
import os

from models import all_models
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL


def main(args):
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    
    if device.type == "cpu":
        print("Warning: Running on CPU, sampling will be slow.")
    
    # Create model
    latent_size = args.image_size // 8
    model = all_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    ).to(device)
    
    # Load checkpoint
    if args.ckpt:
        print(f"Loading checkpoint: {args.ckpt}")
        ckpt = torch.load(args.ckpt, map_location=device)
        if args.use_ema and "ema" in ckpt:
            model.load_state_dict(ckpt["ema"])
            print("Using EMA weights")
        else:
            model.load_state_dict(ckpt["model"])
            print("Using model weights")
    
    model.eval()
    
    # Load VAE
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    
    # Create diffusion
    diffusion = create_diffusion(str(args.num_sampling_steps))
    
    # Setup sampling
    n = args.num_samples
    z = torch.randn(n, 4, latent_size, latent_size, device=device)
    
    # Class labels
    if args.class_labels:
        y = torch.tensor(args.class_labels, device=device)
        if len(y) < n:
            y = y.repeat(n // len(y) + 1)[:n]
    else:
        y = torch.randint(0, args.num_classes, (n,), device=device)
    
    # Classifier-free guidance setup
    z = torch.cat([z, z], 0)
    y_null = torch.tensor([args.num_classes] * n, device=device)
    y = torch.cat([y, y_null], 0)
    
    print(f"Sampling {n} images with {args.num_sampling_steps} steps, cfg_scale={args.cfg_scale}...")
    
    # Sample
    with torch.no_grad():
        samples = diffusion.p_sample_loop(
            model.forward_with_cfg,
            z.shape,
            z,
            clip_denoised=False,
            model_kwargs=dict(y=y, cfg_scale=args.cfg_scale),
            progress=True,
            device=device
        )
    
    samples, _ = samples.chunk(2, dim=0)
    
    # Decode
    with torch.no_grad():
        samples = vae.decode(samples / 0.18215).sample
    
    # Save
    samples = torch.clamp(samples * 0.5 + 0.5, 0, 1)
    
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    save_image(samples, args.output, nrow=int(args.num_samples ** 0.5), padding=2)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(all_models.keys()), default="DiT-S/4")
    parser.add_argument("--ckpt", type=str, default=None, help="Path to checkpoint")
    parser.add_argument("--use-ema", action="store_true", default=True, help="Use EMA weights")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--class-labels", type=int, nargs="+", default=None, help="Class labels to sample")
    parser.add_argument("--output", type=str, default="samples.png")
    args = parser.parse_args()
    main(args)
