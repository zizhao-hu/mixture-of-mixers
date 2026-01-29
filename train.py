# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Unified training script for DiT and MoM models.
# Supports single-GPU and multi-GPU (DDP) training.
#
# Usage:
#   Single GPU:  python train.py --data-path /path/to/data --model DiT-S/4
#   Multi-GPU:   torchrun --nproc_per_node=2 train.py --data-path /path/to/data --model MoM-S/4

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

from models import all_models
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL


#################################################################################
#                             Dataset with Error Handling                        #
#################################################################################

class SafeImageFolder(ImageFolder):
    """ImageFolder that skips corrupted images."""
    def __getitem__(self, index):
        try:
            return super().__getitem__(index)
        except Exception:
            return None


def collate_skip_none(batch):
    """Collate function that filters out None values."""
    batch = [x for x in batch if x is not None]
    if len(batch) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)


#################################################################################
#                             Training Helper Functions                          #
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


def create_logger(logging_dir, rank=0):
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
#                              Sampling Function                                 #
#################################################################################

@torch.no_grad()
def generate_samples(model, vae, diffusion, num_classes, latent_size, device,
                     num_samples=16, cfg_scale=4.0, num_steps=250, seed=0):
    """Generate samples from the model for visualization."""
    model.eval()
    torch.manual_seed(seed)
    
    n = num_samples
    z = torch.randn(n, 4, latent_size, latent_size, device=device)
    y = torch.randint(0, num_classes, (n,), device=device)
    
    # Setup classifier-free guidance
    z = torch.cat([z, z], 0)
    y_null = torch.tensor([num_classes] * n, device=device)
    y = torch.cat([y, y_null], 0)
    
    # Create sampling diffusion
    sample_diffusion = create_diffusion(str(num_steps))
    
    # Sample
    samples = sample_diffusion.p_sample_loop(
        model.forward_with_cfg,
        z.shape,
        z,
        clip_denoised=False,
        model_kwargs=dict(y=y, cfg_scale=cfg_scale),
        progress=False,
        device=device
    )
    samples, _ = samples.chunk(2, dim=0)
    
    # Decode
    samples = vae.decode(samples / 0.18215).sample
    samples = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
    samples = torch.from_numpy(samples).permute(0, 3, 1, 2).float() / 255.0
    
    model.train()
    return samples


#################################################################################
#                                  Training Loop                                 #
#################################################################################

def main(args):
    # Check if distributed
    distributed = 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE']) > 1
    
    if distributed:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = rank % torch.cuda.device_count()
        seed = args.global_seed * world_size + rank
        torch.cuda.set_device(device)
        print(f"Starting rank={rank}, seed={seed}, world_size={world_size}.")
    else:
        rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        seed = args.global_seed
        print(f"Running on {device}")
    
    torch.manual_seed(seed)
    
    # Setup experiment directory
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"
        checkpoint_dir = f"{experiment_dir}/checkpoints"
        sample_dir = f"{experiment_dir}/samples"
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(sample_dir, exist_ok=True)
        logger = create_logger(experiment_dir, rank)
        logger.info(f"Experiment directory: {experiment_dir}")
    else:
        logger = create_logger(None, rank)
        experiment_dir = checkpoint_dir = sample_dir = None
    
    if distributed:
        dist.barrier()
    
    # Create model
    assert args.image_size % 8 == 0, "Image size must be divisible by 8"
    latent_size = args.image_size // 8
    
    is_mom = args.model.startswith('MoM')
    model = all_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    ).to(device)
    
    ema = deepcopy(model).to(device)
    requires_grad(ema, False)
    
    if distributed:
        model = DDP(model, device_ids=[device])
    
    diffusion = create_diffusion(timestep_respacing="")
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    
    if rank == 0:
        num_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model: {args.model}")
        logger.info(f"Parameters: {num_params:,}")
    
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
    
    if distributed:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=args.global_seed)
        batch_size = args.global_batch_size // world_size
    else:
        sampler = None
        batch_size = args.global_batch_size
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_skip_none
    )
    
    if rank == 0:
        logger.info(f"Dataset: {len(dataset):,} images ({args.data_path})")
        logger.info(f"Batch size: {args.global_batch_size} (per GPU: {batch_size})")
    
    # Training
    update_ema(ema, model.module if distributed else model, decay=0)
    model.train()
    ema.eval()
    
    train_steps = 0
    log_steps = 0
    running_loss = 0
    running_aux_loss = 0
    start_time = time()
    
    if rank == 0:
        logger.info(f"Training for {args.epochs} epochs...")
    
    for epoch in range(args.epochs):
        if distributed:
            sampler.set_epoch(epoch)
        
        if rank == 0:
            logger.info(f"Epoch {epoch}...")
        
        for batch in loader:
            if batch is None:
                continue
            
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            
            with torch.no_grad():
                x = vae.encode(x).latent_dist.sample().mul_(0.18215)
            
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            model_kwargs = dict(y=y)
            
            # Forward pass
            if is_mom:
                # MoM returns (output, aux_loss)
                loss_dict = diffusion.training_losses(
                    lambda *args, **kwargs: (model.module if distributed else model)(*args, **kwargs)[0],
                    x, t, model_kwargs
                )
                # Get aux loss separately
                with torch.no_grad():
                    _, aux_loss = (model.module if distributed else model)(x, t, y)
            else:
                loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
                aux_loss = torch.tensor(0.0, device=device)
            
            loss = loss_dict["loss"].mean()
            
            # Add auxiliary loss for MoM
            if is_mom:
                loss = loss + args.aux_loss_weight * aux_loss
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            update_ema(ema, model.module if distributed else model)
            
            # Logging
            running_loss += loss_dict["loss"].mean().item()
            running_aux_loss += aux_loss.item() if is_mom else 0
            log_steps += 1
            train_steps += 1
            
            if train_steps % args.log_every == 0:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                avg_loss = running_loss / log_steps
                avg_aux = running_aux_loss / log_steps if is_mom else 0
                steps_per_sec = log_steps / (time() - start_time)
                
                if rank == 0:
                    if is_mom:
                        logger.info(f"(step={train_steps:07d}) Loss: {avg_loss:.4f}, Aux: {avg_aux:.4f}, Steps/Sec: {steps_per_sec:.2f}")
                    else:
                        logger.info(f"(step={train_steps:07d}) Loss: {avg_loss:.4f}, Steps/Sec: {steps_per_sec:.2f}")
                
                running_loss = 0
                running_aux_loss = 0
                log_steps = 0
                start_time = time()
            
            # Checkpointing
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": (model.module if distributed else model).state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args,
                        "train_steps": train_steps,
                        "epoch": epoch
                    }
                    ckpt_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, ckpt_path)
                    logger.info(f"Saved checkpoint: {ckpt_path}")
                    
                    # Generate samples
                    samples = generate_samples(
                        ema, vae, diffusion, args.num_classes, latent_size, device,
                        num_samples=args.num_sample_images,
                        cfg_scale=args.cfg_scale,
                        num_steps=args.sample_steps
                    )
                    sample_path = f"{sample_dir}/{train_steps:07d}.png"
                    save_image(samples, sample_path, nrow=int(np.sqrt(args.num_sample_images)), padding=2)
                    logger.info(f"Saved samples: {sample_path}")
                
                if distributed:
                    dist.barrier()
    
    # Final checkpoint
    if rank == 0:
        checkpoint = {
            "model": (model.module if distributed else model).state_dict(),
            "ema": ema.state_dict(),
            "opt": opt.state_dict(),
            "args": args,
            "train_steps": train_steps,
            "epoch": args.epochs
        }
        torch.save(checkpoint, f"{checkpoint_dir}/final.pt")
        logger.info("Saved final checkpoint")
        
        samples = generate_samples(
            ema, vae, diffusion, args.num_classes, latent_size, device,
            num_samples=args.num_sample_images,
            cfg_scale=args.cfg_scale,
            num_steps=args.sample_steps
        )
        save_image(samples, f"{sample_dir}/final.png", nrow=int(np.sqrt(args.num_sample_images)), padding=2)
        logger.info("Done!")
    
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="results")
    
    # Model
    parser.add_argument("--model", type=str, choices=list(all_models.keys()), default="DiT-S/4")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    
    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--aux-loss-weight", type=float, default=0.01, help="Weight for MoM auxiliary loss")
    
    # Logging & Checkpointing
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=50000)
    
    # Sampling
    parser.add_argument("--sample-steps", type=int, default=250)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sample-images", type=int, default=16)
    
    args = parser.parse_args()
    main(args)
