# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
Multi-GPU DiT training with PyTorch DDP and automatic sampling at checkpoints.

Usage:
    # Single node, 2 GPUs
    torchrun --nnodes=1 --nproc_per_node=2 train_ddp.py --data-path /path/to/data --model DiT-S/4
    
    # With SLURM (see submit_slurm.sh)
    sbatch submit_slurm.sh
"""
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os

from models import DiT_models
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL


#################################################################################
#                             Dataset with Error Handling                       #
#################################################################################

class SafeImageFolder(ImageFolder):
    """ImageFolder that skips corrupted images instead of crashing."""
    
    def __getitem__(self, index):
        try:
            return super().__getitem__(index)
        except Exception as e:
            return None


def collate_skip_none(batch):
    """Collate function that filters out None values from corrupted images."""
    batch = [x for x in batch if x is not None]
    if len(batch) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)


#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """Step the EMA model towards the current model."""
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """Set requires_grad flag for all parameters in a model."""
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """End DDP training."""
    dist.destroy_process_group()


def create_logger(logging_dir, rank):
    """Create a logger that writes to a log file and stdout."""
    if rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def center_crop_arr(pil_image, image_size):
    """Center cropping implementation from ADM."""
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )
    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


#################################################################################
#                              Sampling Function                                #
#################################################################################

@torch.no_grad()
def generate_samples(model, vae, diffusion, num_classes, latent_size, device, 
                     num_samples=8, cfg_scale=4.0, num_steps=250, seed=0):
    """Generate sample images from the model."""
    torch.manual_seed(seed)
    model.eval()
    
    # Generate class labels spread across classes
    class_labels = [i % num_classes for i in range(num_samples)]
    n = len(class_labels)
    
    # Create sampling noise
    z = torch.randn(n, 4, latent_size, latent_size, device=device)
    y = torch.tensor(class_labels, device=device)
    
    # Setup classifier-free guidance
    z = torch.cat([z, z], 0)
    y_null = torch.tensor([num_classes] * n, device=device)
    y = torch.cat([y, y_null], 0)
    model_kwargs = dict(y=y, cfg_scale=cfg_scale)
    
    # Sample
    samples = diffusion.p_sample_loop(
        model.forward_with_cfg, z.shape, z, clip_denoised=False,
        model_kwargs=model_kwargs, progress=False, device=device
    )
    samples, _ = samples.chunk(2, dim=0)
    samples = vae.decode(samples / 0.18215).sample
    
    model.train()
    return samples


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """Trains a DiT model with DDP and samples at each checkpoint."""
    assert torch.cuda.is_available(), "Training requires at least one GPU."

    # Setup DDP
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, \
        f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Setup experiment folder
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"
        checkpoint_dir = f"{experiment_dir}/checkpoints"
        samples_dir = f"{experiment_dir}/samples"
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(samples_dir, exist_ok=True)
        logger = create_logger(experiment_dir, rank)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None, rank)
        experiment_dir = None
        checkpoint_dir = None
        samples_dir = None

    # Broadcast experiment_dir to all ranks
    if rank == 0:
        objects = [experiment_dir, checkpoint_dir, samples_dir]
    else:
        objects = [None, None, None]
    dist.broadcast_object_list(objects, src=0)
    experiment_dir, checkpoint_dir, samples_dir = objects

    # Create model
    assert args.image_size % 8 == 0, "Image size must be divisible by 8."
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    )
    ema = deepcopy(model).to(device)
    requires_grad(ema, False)
    model = DDP(model.to(device), device_ids=[device])
    diffusion = create_diffusion(timestep_respacing="")
    
    # For sampling, create a separate diffusion with fewer steps
    sample_diffusion = create_diffusion(str(args.sample_steps))
    
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0)

    # Setup data
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    dataset = SafeImageFolder(args.data_path, transform=transform)
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_skip_none
    )
    logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")

    # Prepare models for training
    update_ema(ema, model.module, decay=0)
    model.train()
    ema.eval()

    # Training variables
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()

    logger.info(f"Training for {args.epochs} epochs...")
    logger.info(f"Global batch size: {args.global_batch_size}")
    logger.info(f"Sampling every {args.ckpt_every} steps with {args.sample_steps} diffusion steps")
    
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        
        for batch in loader:
            if batch is None:
                continue  # Skip if all images in batch were corrupted
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            
            with torch.no_grad():
                x = vae.encode(x).latent_dist.sample().mul_(0.18215)
            
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            model_kwargs = dict(y=y)
            loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
            loss = loss_dict["loss"].mean()
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            update_ema(ema, model.module)

            # Log loss
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            
            if train_steps % args.log_every == 0:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Steps/Sec: {steps_per_sec:.2f}")
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save checkpoint and generate samples
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args,
                        "train_steps": train_steps,
                        "epoch": epoch
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                    
                    # Generate and save samples using EMA model
                    logger.info(f"Generating samples at step {train_steps}...")
                    samples = generate_samples(
                        ema, vae, sample_diffusion, args.num_classes, latent_size, device,
                        num_samples=args.num_sample_images, cfg_scale=args.cfg_scale,
                        num_steps=args.sample_steps, seed=train_steps
                    )
                    sample_path = f"{samples_dir}/step_{train_steps:07d}.png"
                    save_image(samples, sample_path, nrow=4, normalize=True, value_range=(-1, 1))
                    logger.info(f"Saved samples to {sample_path}")
                
                dist.barrier()

    # Final checkpoint and samples
    if rank == 0:
        checkpoint = {
            "model": model.module.state_dict(),
            "ema": ema.state_dict(),
            "opt": opt.state_dict(),
            "args": args,
            "train_steps": train_steps,
            "epoch": args.epochs
        }
        torch.save(checkpoint, f"{checkpoint_dir}/final.pt")
        logger.info(f"Saved final checkpoint")
        
        # Final samples
        logger.info("Generating final samples...")
        samples = generate_samples(
            ema, vae, sample_diffusion, args.num_classes, latent_size, device,
            num_samples=args.num_sample_images, cfg_scale=args.cfg_scale,
            num_steps=args.sample_steps, seed=42
        )
        save_image(samples, f"{samples_dir}/final.png", nrow=4, normalize=True, value_range=(-1, 1))
        logger.info(f"Saved final samples")

    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="results")
    
    # Model
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-S/4")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    
    # Training
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--num-workers", type=int, default=4)
    
    # Logging & Checkpointing
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=10000)
    
    # Sampling at checkpoints
    parser.add_argument("--sample-steps", type=int, default=250, help="Diffusion steps for sampling")
    parser.add_argument("--cfg-scale", type=float, default=4.0, help="Classifier-free guidance scale")
    parser.add_argument("--num-sample-images", type=int, default=16, help="Number of images to sample at each checkpoint")
    
    args = parser.parse_args()
    main(args)
