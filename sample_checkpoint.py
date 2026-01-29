# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
Sample images from a DiT checkpoint.
Standalone script that doesn't require download.py.

Usage:
    python sample_checkpoint.py --ckpt /path/to/checkpoint.pt --model DiT-S/4 --num-classes 74
"""
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from models import DiT_models
import argparse
import os


def sample_from_checkpoint(
    ckpt_path,
    model_name="DiT-S/4",
    num_classes=74,
    image_size=256,
    num_sampling_steps=250,
    cfg_scale=4.0,
    num_samples=8,
    seed=0,
    output_path=None,
    device="cuda",
    vae_type="ema"
):
    """
    Generate samples from a DiT checkpoint.
    
    Args:
        ckpt_path: Path to the checkpoint file
        model_name: DiT model variant
        num_classes: Number of classes in the model
        image_size: Output image size
        num_sampling_steps: Number of DDPM sampling steps
        cfg_scale: Classifier-free guidance scale
        num_samples: Number of images to generate
        seed: Random seed
        output_path: Where to save the output image grid
        device: Device to run on
        vae_type: VAE variant ("ema" or "mse")
    
    Returns:
        samples: Generated image tensor
    """
    torch.manual_seed(seed)
    torch.set_grad_enabled(False)
    
    # Load model
    latent_size = image_size // 8
    model = DiT_models[model_name](
        input_size=latent_size,
        num_classes=num_classes
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(ckpt_path, map_location=device)
    if "ema" in checkpoint:
        # Use EMA weights for sampling (better quality)
        model.load_state_dict(checkpoint["ema"])
    elif "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    # Setup diffusion and VAE
    diffusion = create_diffusion(str(num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{vae_type}").to(device)
    
    # Generate random class labels (spread across available classes)
    class_labels = [i % num_classes for i in range(num_samples)]
    
    # Create sampling noise
    n = len(class_labels)
    z = torch.randn(n, 4, latent_size, latent_size, device=device)
    y = torch.tensor(class_labels, device=device)
    
    # Setup classifier-free guidance
    z = torch.cat([z, z], 0)
    y_null = torch.tensor([num_classes] * n, device=device)  # null class for CFG
    y = torch.cat([y, y_null], 0)
    model_kwargs = dict(y=y, cfg_scale=cfg_scale)
    
    # Sample images
    print(f"Sampling {num_samples} images with {num_sampling_steps} steps...")
    samples = diffusion.p_sample_loop(
        model.forward_with_cfg, z.shape, z, clip_denoised=False, 
        model_kwargs=model_kwargs, progress=True, device=device
    )
    samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
    samples = vae.decode(samples / 0.18215).sample
    
    # Save images
    if output_path is None:
        ckpt_name = os.path.basename(ckpt_path).replace(".pt", "")
        output_path = f"samples_{ckpt_name}.png"
    
    save_image(samples, output_path, nrow=4, normalize=True, value_range=(-1, 1))
    print(f"Saved samples to {output_path}")
    
    return samples


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    sample_from_checkpoint(
        ckpt_path=args.ckpt,
        model_name=args.model,
        num_classes=args.num_classes,
        image_size=args.image_size,
        num_sampling_steps=args.num_sampling_steps,
        cfg_scale=args.cfg_scale,
        num_samples=args.num_samples,
        seed=args.seed,
        output_path=args.output,
        device=device,
        vae_type=args.vae
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to DiT checkpoint")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-S/4")
    parser.add_argument("--num-classes", type=int, default=74)
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=str, default=None, help="Output image path")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    args = parser.parse_args()
    main(args)
