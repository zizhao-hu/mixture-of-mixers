"""
Download and prepare MS-COCO dataset for DiT training.

This script downloads COCO 2017 train images and organizes them by category
in ImageFolder format for class-conditional training.

Usage:
    python download_coco.py --output-dir ./data/coco
"""

import os
import json
import argparse
import urllib.request
import zipfile
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import shutil


COCO_URLS = {
    "train_images": "http://images.cocodataset.org/zips/train2017.zip",
    "val_images": "http://images.cocodataset.org/zips/val2017.zip",
    "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
}

# COCO 80 categories
COCO_CATEGORIES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url, output_path):
    """Download a file with progress bar."""
    if os.path.exists(output_path):
        print(f"File already exists: {output_path}")
        return
    
    print(f"Downloading {url}...")
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=output_path) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def extract_zip(zip_path, extract_to):
    """Extract a zip file."""
    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)


def organize_by_category(coco_dir, output_dir, split="train"):
    """
    Organize COCO images by category in ImageFolder format.
    Each image is assigned to its primary (largest area) object category.
    """
    ann_file = os.path.join(coco_dir, "annotations", f"instances_{split}2017.json")
    img_dir = os.path.join(coco_dir, f"{split}2017")
    
    print(f"Loading annotations from {ann_file}...")
    with open(ann_file, 'r') as f:
        coco_ann = json.load(f)
    
    # Build category id to name mapping
    cat_id_to_name = {cat['id']: cat['name'] for cat in coco_ann['categories']}
    cat_id_to_idx = {cat['id']: idx for idx, cat in enumerate(coco_ann['categories'])}
    
    # Find primary category for each image (largest bounding box area)
    img_to_category = {}
    img_to_max_area = defaultdict(float)
    
    for ann in tqdm(coco_ann['annotations'], desc="Processing annotations"):
        img_id = ann['image_id']
        cat_id = ann['category_id']
        area = ann['area']
        
        if area > img_to_max_area[img_id]:
            img_to_max_area[img_id] = area
            img_to_category[img_id] = cat_id
    
    # Build image id to filename mapping
    img_id_to_filename = {img['id']: img['file_name'] for img in coco_ann['images']}
    
    # Create output directories
    output_split_dir = os.path.join(output_dir, split)
    for cat in coco_ann['categories']:
        cat_dir = os.path.join(output_split_dir, f"{cat_id_to_idx[cat['id']]:03d}_{cat['name'].replace(' ', '_')}")
        os.makedirs(cat_dir, exist_ok=True)
    
    # Copy/link images to category folders
    print(f"Organizing {len(img_to_category)} images by category...")
    for img_id, cat_id in tqdm(img_to_category.items(), desc="Organizing images"):
        src_path = os.path.join(img_dir, img_id_to_filename[img_id])
        cat_idx = cat_id_to_idx[cat_id]
        cat_name = cat_id_to_name[cat_id].replace(' ', '_')
        dst_dir = os.path.join(output_split_dir, f"{cat_idx:03d}_{cat_name}")
        dst_path = os.path.join(dst_dir, img_id_to_filename[img_id])
        
        if not os.path.exists(dst_path):
            # Use symlink on Unix, copy on Windows
            try:
                os.symlink(os.path.abspath(src_path), dst_path)
            except (OSError, NotImplementedError):
                shutil.copy2(src_path, dst_path)
    
    # Save category mapping
    mapping_file = os.path.join(output_dir, "category_mapping.json")
    mapping = {
        "idx_to_name": {cat_id_to_idx[cat['id']]: cat['name'] for cat in coco_ann['categories']},
        "name_to_idx": {cat['name']: cat_id_to_idx[cat['id']] for cat in coco_ann['categories']},
        "num_classes": len(coco_ann['categories'])
    }
    with open(mapping_file, 'w') as f:
        json.dump(mapping, f, indent=2)
    
    print(f"Category mapping saved to {mapping_file}")
    print(f"Dataset organized at {output_split_dir}")
    print(f"Number of classes: {len(coco_ann['categories'])}")
    
    return len(coco_ann['categories'])


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    download_dir = os.path.join(args.output_dir, "downloads")
    os.makedirs(download_dir, exist_ok=True)
    
    # Download files
    if not args.skip_download:
        for name, url in COCO_URLS.items():
            if args.val_only and name == "train_images":
                continue
            zip_path = os.path.join(download_dir, os.path.basename(url))
            download_file(url, zip_path)
            
            # Extract
            extract_to = os.path.join(args.output_dir, "raw")
            os.makedirs(extract_to, exist_ok=True)
            if not os.path.exists(os.path.join(extract_to, name.replace("_images", "2017").replace("annotations", "annotations"))):
                extract_zip(zip_path, extract_to)
    
    raw_dir = os.path.join(args.output_dir, "raw")
    organized_dir = os.path.join(args.output_dir, "imagefolder")
    
    # Organize by category
    if not args.val_only:
        num_classes = organize_by_category(raw_dir, organized_dir, split="train")
    
    if args.include_val or args.val_only:
        num_classes = organize_by_category(raw_dir, organized_dir, split="val")
    
    print("\n" + "="*60)
    print("COCO dataset preparation complete!")
    print("="*60)
    print(f"\nTo train DiT on COCO, run:")
    print(f"  python train_single_gpu.py --data-path {os.path.join(organized_dir, 'train')} --num-classes 80 --model DiT-S/4")
    print(f"\nOr for a quick test with validation set:")
    print(f"  python train_single_gpu.py --data-path {os.path.join(organized_dir, 'val')} --num-classes 80 --model DiT-S/4 --batch-size 8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and prepare MS-COCO for DiT training")
    parser.add_argument("--output-dir", type=str, default="./data/coco", help="Output directory")
    parser.add_argument("--skip-download", action="store_true", help="Skip downloading (use existing files)")
    parser.add_argument("--val-only", action="store_true", help="Only download and process validation set (smaller, for testing)")
    parser.add_argument("--include-val", action="store_true", help="Also organize validation set")
    args = parser.parse_args()
    main(args)
