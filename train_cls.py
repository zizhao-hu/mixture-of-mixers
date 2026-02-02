# Classification training script for ViT and ViT-MoM models.
#
# Implements best practices for training ViT on small datasets:
#   - RandAugment/AutoAugment
#   - Mixup/CutMix
#   - Label smoothing
#   - Warmup + Cosine LR schedule
#   - Gradient clipping
#
# References:
#   - https://github.com/omihub777/ViT-CIFAR (90%+ on CIFAR-10)
#   - https://github.com/kentaroy47/vision-transformers-cifar10
#
# Usage:
#   python train_cls.py --dataset cifar10 --model ViT-S/4 --epochs 200
#   python train_cls.py --dataset cifar10 --model ViT-MoM-S/4 --epochs 200

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder

import os
import time
import math
import argparse
import json
import random
import numpy as np
from collections import OrderedDict

from models import ViT_models, ViT_MoM_models, ViT_LMoM_models, ViT_SMoM_models


#################################################################################
#                              Data Augmentation                                 #
#################################################################################

class RandAugment:
    """RandAugment data augmentation."""
    def __init__(self, n=2, m=10):
        self.n = n
        self.m = m
        self.augment_list = [
            ('AutoContrast', 0, 1),
            ('Equalize', 0, 1),
            ('Rotate', -30, 30),
            ('Posterize', 4, 8),
            ('Solarize', 0, 256),
            ('Color', 0.1, 1.9),
            ('Contrast', 0.1, 1.9),
            ('Brightness', 0.1, 1.9),
            ('Sharpness', 0.1, 1.9),
            ('ShearX', -0.3, 0.3),
            ('ShearY', -0.3, 0.3),
            ('TranslateX', -0.3, 0.3),
            ('TranslateY', -0.3, 0.3),
        ]

    def __call__(self, img):
        from PIL import Image, ImageOps, ImageEnhance
        
        ops = random.choices(self.augment_list, k=self.n)
        for op_name, min_val, max_val in ops:
            val = min_val + (max_val - min_val) * (self.m / 30.0)
            
            if op_name == 'AutoContrast':
                img = ImageOps.autocontrast(img)
            elif op_name == 'Equalize':
                img = ImageOps.equalize(img)
            elif op_name == 'Rotate':
                img = img.rotate(val)
            elif op_name == 'Posterize':
                img = ImageOps.posterize(img, int(val))
            elif op_name == 'Solarize':
                img = ImageOps.solarize(img, int(val))
            elif op_name == 'Color':
                img = ImageEnhance.Color(img).enhance(val)
            elif op_name == 'Contrast':
                img = ImageEnhance.Contrast(img).enhance(val)
            elif op_name == 'Brightness':
                img = ImageEnhance.Brightness(img).enhance(val)
            elif op_name == 'Sharpness':
                img = ImageEnhance.Sharpness(img).enhance(val)
            elif op_name == 'ShearX':
                img = img.transform(img.size, Image.AFFINE, (1, val, 0, 0, 1, 0))
            elif op_name == 'ShearY':
                img = img.transform(img.size, Image.AFFINE, (1, 0, 0, val, 1, 0))
            elif op_name == 'TranslateX':
                img = img.transform(img.size, Image.AFFINE, (1, 0, val * img.size[0], 0, 1, 0))
            elif op_name == 'TranslateY':
                img = img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, val * img.size[1]))
        
        return img


class Cutout:
    """Cutout augmentation."""
    def __init__(self, size=16):
        self.size = size

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = torch.ones(h, w)
        
        y = random.randint(0, h)
        x = random.randint(0, w)
        
        y1 = max(0, y - self.size // 2)
        y2 = min(h, y + self.size // 2)
        x1 = max(0, x - self.size // 2)
        x2 = min(w, x + self.size // 2)
        
        mask[y1:y2, x1:x2] = 0
        img = img * mask.unsqueeze(0)
        return img


def mixup_data(x, y, alpha=0.8):
    """Mixup augmentation."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup loss."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


#################################################################################
#                              Dataset Utilities                                 #
#################################################################################

def get_dataset(name, data_path, img_size, train=True, randaugment=True):
    """Get dataset with appropriate transforms."""
    
    if name in ['cifar10', 'cifar100']:
        # CIFAR mean/std
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        
        if train:
            transform_list = [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
            ]
            if randaugment:
                transform_list.append(RandAugment(n=2, m=14))
            transform_list.extend([
                transforms.Resize(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
            # Add cutout after ToTensor
            if randaugment:
                transform_list.append(Cutout(size=img_size // 4))
            
            transform = transforms.Compose(transform_list)
        else:
            transform = transforms.Compose([
                transforms.Resize(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        
        if name == 'cifar10':
            dataset = CIFAR10(root=data_path, train=train, download=True, transform=transform)
            num_classes = 10
        else:
            dataset = CIFAR100(root=data_path, train=train, download=True, transform=transform)
            num_classes = 100
            
    elif name == 'tiny-imagenet':
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        
        if train:
            transform_list = [
                transforms.RandomCrop(64, padding=8),
                transforms.RandomHorizontalFlip(),
            ]
            if randaugment:
                transform_list.append(RandAugment(n=2, m=14))
            transform_list.extend([
                transforms.Resize(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
            transform = transforms.Compose(transform_list)
            dataset = ImageFolder(os.path.join(data_path, 'train'), transform=transform)
        else:
            transform = transforms.Compose([
                transforms.Resize(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
            dataset = ImageFolder(os.path.join(data_path, 'val'), transform=transform)
        num_classes = 200
        
    elif name == 'imagenet':
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        
        if train:
            transform = transforms.Compose([
                transforms.RandomResizedCrop(img_size),
                transforms.RandomHorizontalFlip(),
                RandAugment(n=2, m=14) if randaugment else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
            dataset = ImageFolder(os.path.join(data_path, 'train'), transform=transform)
        else:
            transform = transforms.Compose([
                transforms.Resize(int(img_size * 256 / 224)),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
            dataset = ImageFolder(os.path.join(data_path, 'val'), transform=transform)
        num_classes = 1000
    else:
        raise ValueError(f"Unknown dataset: {name}")
    
    return dataset, num_classes


#################################################################################
#                              Training Utilities                                #
#################################################################################

class AverageMeter:
    """Computes and stores the average and current value."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Compute top-k accuracy."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class LabelSmoothingCrossEntropy(torch.nn.Module):
    """Cross entropy with label smoothing."""
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, x, target):
        confidence = 1. - self.smoothing
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


def get_cosine_schedule_with_warmup(optimizer, warmup_epochs, total_epochs, min_lr=1e-5):
    """Cosine schedule with warmup."""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            remaining = total_epochs - warmup_epochs
            if remaining <= 0:
                return 1.0
            progress = (epoch - warmup_epochs) / remaining
            return max(min_lr / optimizer.defaults['lr'], 0.5 * (1 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def create_logger(log_dir, rank=0, resume=False):
    """Create a logger that writes to a file and stdout."""
    import logging
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.handlers = []  # Clear existing handlers
    
    if rank == 0:
        os.makedirs(log_dir, exist_ok=True)
        mode = 'a' if resume else 'w'
        handler = logging.FileHandler(os.path.join(log_dir, 'log.txt'), mode=mode)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(formatter)
        logger.addHandler(console)
    
    return logger


#################################################################################
#                                   Training                                     #
#################################################################################

def train_one_epoch(model, loader, optimizer, scheduler, device, epoch, is_mom, 
                    aux_loss_weight, criterion, use_mixup, mixup_alpha, logger, rank):
    """Train for one epoch."""
    model.train()
    
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    aux_losses = AverageMeter()
    
    total_samples = 0
    start_time = time.time()
    
    for i, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        # Mixup
        if use_mixup:
            images, targets_a, targets_b, lam = mixup_data(images, targets, mixup_alpha)
        
        optimizer.zero_grad()
        
        if is_mom:
            logits, aux_loss = model(images)
            if use_mixup:
                ce_loss = mixup_criterion(criterion, logits, targets_a, targets_b, lam)
            else:
                ce_loss = criterion(logits, targets)
            loss = ce_loss + aux_loss_weight * aux_loss
            aux_losses.update(aux_loss.item(), images.size(0))
        else:
            logits = model(images)
            if use_mixup:
                loss = mixup_criterion(criterion, logits, targets_a, targets_b, lam)
            else:
                loss = criterion(logits, targets)
        
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Measure accuracy
        if use_mixup:
            # When using mixup, report accuracy against the dominant label
            # (the one with the higher mixing coefficient)
            targets_dominant = targets_a if lam > 0.5 else targets_b
            acc1, acc5 = accuracy(logits, targets_dominant, topk=(1, 5))
        else:
            acc1, acc5 = accuracy(logits, targets, topk=(1, 5))
        
        top1.update(acc1.item(), images.size(0))
        top5.update(acc5.item(), images.size(0))
        
        losses.update(loss.item(), images.size(0))
        total_samples += images.size(0)
    
    elapsed = time.time() - start_time
    samples_per_sec = total_samples / elapsed if elapsed > 0 else 0
    
    return losses.avg, top1.avg, top5.avg, aux_losses.avg if is_mom else 0, samples_per_sec


@torch.no_grad()
def evaluate(model, loader, device, is_mom, criterion):
    """Evaluate the model."""
    model.eval()
    
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        if is_mom:
            logits, _ = model(images)
        else:
            logits = model(images)
        
        loss = criterion(logits, targets)
        acc1, acc5 = accuracy(logits, targets, topk=(1, 5))
        
        losses.update(loss.item(), images.size(0))
        top1.update(acc1.item(), images.size(0))
        top5.update(acc5.item(), images.size(0))
    
    return losses.avg, top1.avg, top5.avg


def get_num_params(model):
    """Get total and active parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    
    # Active parameters calculation for MoM/MoE models
    active_params = 0
    
    # Simple recursive function to find MoM layers and calculate active params
    def _count_active(module):
        nonlocal active_params
        if hasattr(module, 'num_experts') and hasattr(module, 'top_k') and hasattr(module, 'experts'):
            # This is a MixtureOfMixers (top-k) layer
            # Inactive = (num_experts - top_k) experts
            if hasattr(module.experts, 'fc1_weight'):
                # MLP experts (old)
                expert_params = (module.experts.fc1_weight.numel() + 
                                 module.experts.fc1_bias.numel() + 
                                 module.experts.fc2_weight.numel() + 
                                 module.experts.fc2_bias.numel())
            elif hasattr(module.experts, 'fc1_tok_weight'):
                # Unified (Bilinear) experts
                expert_params = (module.experts.fc1_tok_weight.numel() + 
                                 module.experts.fc1_chan_weight.numel() +
                                 module.experts.fc1_bias.numel() +
                                 module.experts.fc2_tok_weight.numel() +
                                 module.experts.fc2_chan_weight.numel() +
                                 module.experts.fc2_bias.numel())
            elif hasattr(module.experts, 'weight'):
                # Linear experts (N x N)
                expert_params = (module.experts.weight.numel() + 
                                 module.experts.bias.numel())
            else:
                expert_params = sum(p.numel() for p in module.experts.parameters())
            
            per_expert = expert_params // module.num_experts
            inactive_per_layer = (module.num_experts - module.top_k) * per_expert
            return inactive_per_layer
        
        elif hasattr(module, 'num_experts') and hasattr(module, 'slots_per_expert'):
            # This is a SoftMixtureOfMixers layer
            # All experts are technically active because it's soft, 
            # but usually we report the "bottleneck" as active compute.
            # However, for total parameter efficiency, all parameters are technically used.
            # We'll treat all as active for Soft MoE unless otherwise specified.
            return 0
        
        inactive = 0
        for child in module.children():
            inactive += _count_active(child)
        return inactive

    inactive_params = _count_active(model)
    active_params = total_params - inactive_params
    
    return total_params, active_params


#################################################################################
#                                     Main                                       #
#################################################################################

def main(args):
    # Setup distributed training
    distributed = 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE']) > 1
    
    if distributed:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = rank % torch.cuda.device_count()
        torch.cuda.set_device(device)
    else:
        rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set seed
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    random.seed(args.seed + rank)
    
    # Create experiment directory
    model_name = args.model.replace('/', '-')
    # Simplify model name as requested: no mixup, no ls
    simple_model_name = f"{model_name}_e{args.epochs}"
    
    # Path: results/cls/<dataset>/<model>
    dataset_name = args.dataset
    if dataset_name.startswith('cifar'):
        dataset_name = 'cifar'
    exp_dir = os.path.join(args.results_dir, dataset_name, simple_model_name)
    
    if rank == 0:
        os.makedirs(exp_dir, exist_ok=True)
        os.makedirs(os.path.join(exp_dir, 'checkpoints'), exist_ok=True)
    
    if distributed:
        dist.barrier()
    
    logger = create_logger(exp_dir, rank, resume=args.resume is not None)
    
    # Get dataset
    if args.dataset in ['cifar10', 'cifar100']:
        img_size = args.img_size if args.img_size else 32
    elif args.dataset == 'tiny-imagenet':
        img_size = args.img_size if args.img_size else 64
    else:
        img_size = args.img_size if args.img_size else 224
    
    train_dataset, num_classes = get_dataset(
        args.dataset, args.data_path, img_size, train=True, 
        randaugment=args.randaugment
    )
    val_dataset, _ = get_dataset(
        args.dataset, args.data_path, img_size, train=False, 
        randaugment=False
    )
    
    # Create data loaders
    if distributed:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    # Create model
    is_mom = 'MoM' in args.model
    all_cls_models = {**ViT_models, **ViT_MoM_models, **ViT_LMoM_models, **ViT_SMoM_models}
    
    model = all_cls_models[args.model](
        img_size=img_size,
        num_classes=num_classes,
    ).to(device)
    
    if distributed:
        model = DDP(model, device_ids=[device])
    
    # Log model info
    if rank == 0:
        total_params, active_params = get_num_params(model)
        logger.info(f"="*60)
        logger.info(f"Experiment: {exp_dir}")
        logger.info(f"Model: {args.model}")
        logger.info(f"Total Parameters: {total_params:,}")
        logger.info(f"Active Parameters: {active_params:,}")
        logger.info(f"Dataset: {args.dataset} ({num_classes} classes)")
        logger.info(f"Image size: {img_size}")
        logger.info(f"Batch size: {args.batch_size} x {world_size} = {args.batch_size * world_size}")
        logger.info(f"Epochs: {args.epochs}")
        logger.info(f"Learning rate: {args.lr}")
        logger.info(f"Warmup epochs: {args.warmup_epochs}")
        logger.info(f"Label smoothing: {args.label_smoothing}")
        logger.info(f"Mixup: {args.mixup} (alpha={args.mixup_alpha})")
        logger.info(f"RandAugment: {args.randaugment}")
        logger.info(f"="*60)
    
    # Loss function
    if args.label_smoothing > 0:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.label_smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()
    
    # Create optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )
    
    # Cosine schedule with warmup
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        warmup_epochs=args.warmup_epochs,
        total_epochs=args.epochs,
        min_lr=args.min_lr,
    )
    
    # Resume from checkpoint
    start_epoch = 0
    best_acc1 = 0
    if args.resume:
        if os.path.isfile(args.resume):
            if rank == 0:
                logger.info(f"Loading checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location='cpu')
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint.get('best_acc1', 0)
            if rank == 0:
                logger.info(f"Resumed from epoch {start_epoch} with best Acc@1 {best_acc1:.2f}%")
        else:
            if rank == 0:
                logger.info(f"No checkpoint found at {args.resume}")
    
    # Training loop
    results = []
    
    for epoch in range(start_epoch, args.epochs):
        if distributed:
            train_sampler.set_epoch(epoch)
        
        # Train
        train_loss, train_acc1, train_acc5, aux_loss, samples_per_sec = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, epoch,
            is_mom, args.aux_loss_weight, criterion, 
            args.mixup, args.mixup_alpha, logger, rank
        )
        
        scheduler.step()
        
        # Evaluate
        val_loss, val_acc1, val_acc5 = evaluate(model, val_loader, device, is_mom, criterion)
        
        # All-reduce for distributed
        if distributed:
            val_acc1_tensor = torch.tensor(val_acc1).to(device)
            dist.all_reduce(val_acc1_tensor, op=dist.ReduceOp.AVG)
            val_acc1 = val_acc1_tensor.item()
            
            val_acc5_tensor = torch.tensor(val_acc5).to(device)
            dist.all_reduce(val_acc5_tensor, op=dist.ReduceOp.AVG)
            val_acc5 = val_acc5_tensor.item()
        
        # Track best accuracy
        is_best = val_acc1 > best_acc1
        best_acc1 = max(val_acc1, best_acc1)
        
        # Log
        if rank == 0:
            result = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_acc1': train_acc1,
                'train_acc5': train_acc5,
                'val_loss': val_loss,
                'val_acc1': val_acc1,
                'val_acc5': val_acc5,
                'lr': optimizer.param_groups[0]['lr'],
            }
            if is_mom:
                result['aux_loss'] = aux_loss
            results.append(result)
            
            log_msg = (
                f"Epoch [{epoch+1}/{args.epochs}] "
                f"LR={optimizer.param_groups[0]['lr']:.6f} | "
                f"Train Loss={train_loss:.4f}, Acc@1={train_acc1:.2f}% | "
                f"Val: Loss={val_loss:.4f}, Acc@1={val_acc1:.2f}%, Acc@5={val_acc5:.2f}% | "
                f"Best={best_acc1:.2f}% | "
                f"Speed={samples_per_sec:.1f} samples/s"
            )
            if is_mom:
                log_msg += f" | Aux={aux_loss:.4f}"
            logger.info(log_msg)
            
            # Save checkpoint
            if is_best or (epoch + 1) % args.save_every == 0 or (epoch + 1) == args.epochs:
                checkpoint = {
                    'epoch': epoch + 1,
                    'model': model.module.state_dict() if distributed else model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'best_acc1': best_acc1,
                    'args': vars(args),
                }
                
                if is_best:
                    torch.save(checkpoint, os.path.join(exp_dir, 'checkpoints', 'best.pt'))
                if (epoch + 1) % args.save_every == 0:
                    torch.save(checkpoint, os.path.join(exp_dir, 'checkpoints', f'epoch_{epoch+1}.pt'))
    
    # Save final results
    if rank == 0:
        # Save final checkpoint
        checkpoint = {
            'epoch': args.epochs,
            'model': model.module.state_dict() if distributed else model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_acc1': best_acc1,
            'args': vars(args),
        }
        torch.save(checkpoint, os.path.join(exp_dir, 'checkpoints', 'final.pt'))
        
        # Save results JSON
        with open(os.path.join(exp_dir, 'results.json'), 'w') as f:
            json.dump({
                'args': vars(args),
                'best_acc1': best_acc1,
                'final_acc1': val_acc1,
                'results': results,
            }, f, indent=2)
        
        logger.info(f"="*60)
        logger.info(f"Training complete!")
        logger.info(f"Best Acc@1: {best_acc1:.2f}%")
        logger.info(f"Results saved to: {exp_dir}")
        logger.info(f"="*60)
    
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classification training for ViT and ViT-MoM")
    
    # Dataset
    parser.add_argument("--dataset", type=str, default="cifar10", 
                        choices=['cifar10', 'cifar100', 'tiny-imagenet', 'imagenet'],
                        help="Dataset to use")
    parser.add_argument("--data-path", type=str, default="./data", help="Path to dataset")
    parser.add_argument("--img-size", type=int, default=None, help="Image size (default: dataset native)")
    
    # Model
    parser.add_argument("--model", type=str, default="ViT-S/4", help="Model architecture")
    
    # Training
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size per GPU")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--min-lr", type=float, default=1e-5, help="Minimum learning rate")
    parser.add_argument("--weight-decay", type=float, default=5e-5, help="Weight decay")
    parser.add_argument("--warmup-epochs", type=int, default=5, help="Warmup epochs")
    
    # Augmentation & Regularization
    parser.add_argument("--label-smoothing", type=float, default=0.1, help="Label smoothing")
    parser.add_argument("--mixup", action="store_true", help="Use mixup")
    parser.add_argument("--mixup-alpha", type=float, default=0.8, help="Mixup alpha")
    parser.add_argument("--randaugment", action="store_true", default=True, help="Use RandAugment")
    parser.add_argument("--no-randaugment", action="store_false", dest="randaugment", help="Disable RandAugment")
    
    # MoM specific
    parser.add_argument("--aux-loss-weight", type=float, default=0.01, help="MoM auxiliary loss weight")
    
    # Misc
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of data workers")
    parser.add_argument("--results-dir", type=str, default="./results/cls", help="Results directory")
    parser.add_argument("--save-every", type=int, default=50, help="Save checkpoint every N epochs")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    
    args = parser.parse_args()
    main(args)
