#!/usr/bin/env python3
"""
Download and prepare ImageNet for DiT training on HPC clusters.

ImageNet requires authorization. Options:
1. Hugging Face (recommended): Get access at https://huggingface.co/datasets/imagenet-1k
2. Academic Torrents: https://academictorrents.com/details/a306397ccf9c2ead27155983c254227c0fd938e2
3. Official: https://image-net.org/download.php

Usage on CARC transfer node:
    ssh hpc-transfer1.usc.edu
    cd /scratch1/$USER
    module load conda
    source activate dit
    huggingface-cli login  # Enter your HF token
    python download_imagenet.py --output-dir /scratch1/$USER/imagenet

This script downloads ImageNet-1K and organizes it into ImageFolder format:
    imagenet/
    ├── train/
    │   ├── n01440764/
    │   │   ├── n01440764_10026.JPEG
    │   │   └── ...
    │   └── ... (1000 classes)
    └── val/
        ├── n01440764/
        └── ... (1000 classes)
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path


def check_huggingface_auth():
    """Check if user is authenticated with Hugging Face."""
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        api.whoami()
        return True
    except Exception:
        return False


def download_from_huggingface(output_dir: str, split: str = "all"):
    """Download ImageNet from Hugging Face datasets."""
    try:
        from datasets import load_dataset
        from PIL import Image
        from tqdm import tqdm
    except ImportError:
        print("Installing required packages...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", 
                               "datasets", "Pillow", "tqdm", "huggingface_hub"])
        from datasets import load_dataset
        from PIL import Image
        from tqdm import tqdm
    
    output_path = Path(output_dir)
    
    splits_to_download = ["train", "validation"] if split == "all" else [split]
    
    for split_name in splits_to_download:
        print(f"\n{'='*60}")
        print(f"Downloading ImageNet {split_name} split...")
        print(f"{'='*60}")
        
        # Map validation -> val for output folder
        out_split = "val" if split_name == "validation" else split_name
        split_dir = output_path / out_split
        split_dir.mkdir(parents=True, exist_ok=True)
        
        # Load dataset with streaming to avoid memory issues
        dataset = load_dataset(
            "imagenet-1k",
            split=split_name,
            trust_remote_code=True
        )
        
        print(f"Found {len(dataset)} images in {split_name} split")
        
        # Get class names mapping
        class_names = dataset.features["label"].names
        
        # Create class directories
        for class_name in class_names:
            (split_dir / class_name).mkdir(exist_ok=True)
        
        # Save images
        for idx, example in enumerate(tqdm(dataset, desc=f"Saving {split_name}")):
            image = example["image"]
            label = example["label"]
            class_name = class_names[label]
            
            # Save image
            img_path = split_dir / class_name / f"{class_name}_{idx:08d}.JPEG"
            if not img_path.exists():
                if image.mode != "RGB":
                    image = image.convert("RGB")
                image.save(img_path, "JPEG", quality=95)
        
        print(f"Saved {split_name} split to {split_dir}")
    
    print(f"\n{'='*60}")
    print(f"ImageNet download complete!")
    print(f"Location: {output_path}")
    print(f"{'='*60}")


def download_from_torrent(output_dir: str):
    """Download ImageNet using Academic Torrents (aria2)."""
    print("Downloading ImageNet from Academic Torrents...")
    print("This requires aria2 to be installed.")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Academic Torrents magnet links for ImageNet
    train_magnet = "magnet:?xt=urn:btih:a306397ccf9c2ead27155983c254227c0fd938e2"
    val_magnet = "magnet:?xt=urn:btih:5d6d0df7ed81efd49ca99ea4737e0ae5e3a5f2e5"
    
    print("\nTo download via torrent, run:")
    print(f"  aria2c --seed-time=0 -d {output_path} '{train_magnet}'")
    print(f"  aria2c --seed-time=0 -d {output_path} '{val_magnet}'")
    print("\nThen extract and organize the data.")


def create_symlink_to_project(scratch_path: str, project_path: str):
    """Create symlink from project dir to scratch for easy access."""
    scratch = Path(scratch_path)
    project = Path(project_path)
    
    link_path = project / "data" / "imagenet"
    link_path.parent.mkdir(parents=True, exist_ok=True)
    
    if link_path.exists() or link_path.is_symlink():
        print(f"Removing existing link: {link_path}")
        link_path.unlink()
    
    link_path.symlink_to(scratch)
    print(f"Created symlink: {link_path} -> {scratch}")


def verify_imagenet(data_dir: str):
    """Verify ImageNet dataset structure and counts."""
    data_path = Path(data_dir)
    
    print(f"\nVerifying ImageNet at {data_path}...")
    
    expected = {
        "train": 1281167,
        "val": 50000,
    }
    
    for split, expected_count in expected.items():
        split_dir = data_path / split
        if not split_dir.exists():
            print(f"  {split}: MISSING")
            continue
        
        classes = list(split_dir.iterdir())
        num_classes = len([c for c in classes if c.is_dir()])
        
        total_images = sum(
            len(list(c.glob("*.JPEG"))) + len(list(c.glob("*.jpeg"))) + len(list(c.glob("*.png")))
            for c in classes if c.is_dir()
        )
        
        status = "OK" if total_images >= expected_count * 0.99 else "INCOMPLETE"
        print(f"  {split}: {num_classes} classes, {total_images:,} images [{status}]")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Download ImageNet for DiT training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="/scratch1/" + os.environ.get("USER", "user") + "/imagenet",
        help="Output directory for ImageNet (default: /scratch1/$USER/imagenet)"
    )
    parser.add_argument(
        "--method",
        choices=["huggingface", "torrent", "verify"],
        default="huggingface",
        help="Download method (default: huggingface)"
    )
    parser.add_argument(
        "--split",
        choices=["train", "validation", "all"],
        default="all",
        help="Which split to download (default: all)"
    )
    parser.add_argument(
        "--create-symlink",
        type=str,
        default=None,
        help="Create symlink from this project path to the dataset"
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("ImageNet Download for DiT Training")
    print("="*60)
    print(f"Output directory: {args.output_dir}")
    print(f"Method: {args.method}")
    print()
    
    if args.method == "verify":
        verify_imagenet(args.output_dir)
        return
    
    if args.method == "huggingface":
        if not check_huggingface_auth():
            print("ERROR: Not authenticated with Hugging Face!")
            print("\nTo authenticate:")
            print("  1. Get access to ImageNet-1k at: https://huggingface.co/datasets/imagenet-1k")
            print("  2. Create access token at: https://huggingface.co/settings/tokens")
            print("  3. Run: huggingface-cli login")
            print("  4. Enter your token when prompted")
            sys.exit(1)
        
        download_from_huggingface(args.output_dir, args.split)
    
    elif args.method == "torrent":
        download_from_torrent(args.output_dir)
    
    # Verify download
    verify_imagenet(args.output_dir)
    
    # Create symlink if requested
    if args.create_symlink:
        create_symlink_to_project(args.output_dir, args.create_symlink)
    
    print("\n" + "="*60)
    print("Next steps:")
    print("="*60)
    print(f"1. Create symlink from your project:")
    print(f"   ln -s {args.output_dir} /project2/jessetho_1732/zizhaoh/mixture-of-mixers/data/imagenet")
    print()
    print("2. Update submit_slurm.sh DATA_PATH:")
    print(f"   DATA_PATH=\"{args.output_dir}/train\"")
    print()
    print("3. Submit training job:")
    print("   sbatch submit_slurm.sh")


if __name__ == "__main__":
    main()
