#!/usr/bin/env python3
"""
FLIR ADAS v2 Dataset Preparation Script for IMDN Fine-tuning

This script prepares the FLIR ADAS v2 thermal dataset for fine-tuning IMDN.
It organizes the thermal images into train/val HR folders and generates corresponding LR images.

Features:
- Handles FLIR dataset structure automatically
- Converts thermal images to proper format for training
- Generates LR images using bicubic downsampling
- Creates train/val splits with proper organization
- Validates image quality and format
- Provides detailed progress information

Usage:
    python prepare_flir_data.py --flir_root /path/to/FLIR-dataset --output_dir /path/to/prepared_dataset --scale 2
"""

import os
import cv2
import numpy as np
import argparse
from pathlib import Path
import shutil
from tqdm import tqdm
import random

def ensure_dir(directory):
    """Create directory if it doesn't exist"""
    Path(directory).mkdir(parents=True, exist_ok=True)

def load_thermal_image(image_path):
    """
    Load thermal image and ensure proper format
    
    Args:
        image_path: Path to thermal image
        
    Returns:
        Loaded image in uint8 format or None if failed
    """
    try:
        # Load image - thermal images might be in different formats
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            print(f"Warning: Could not load {image_path}")
            return None
            
        # Ensure uint8 format
        if img.dtype != np.uint8:
            # Normalize to 0-255 range if needed
            if img.max() > 255:
                img = (img / img.max() * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)
                
        return img
        
    except Exception as e:
        print(f"Error loading {image_path}: {e}")
        return None

def generate_lr_image(hr_image, scale_factor):
    """
    Generate LR image from HR image using bicubic downsampling
    
    Args:
        hr_image: High resolution image
        scale_factor: Downsampling factor (2, 3, or 4)
        
    Returns:
        Low resolution image
    """
    h, w = hr_image.shape[:2]
    lr_h, lr_w = h // scale_factor, w // scale_factor
    
    # Downsample using bicubic interpolation
    lr_image = cv2.resize(hr_image, (lr_w, lr_h), interpolation=cv2.INTER_CUBIC)
    
    return lr_image

def is_valid_thermal_image(image_path, min_size=64):
    """
    Check if thermal image is valid for training
    
    Args:
        image_path: Path to image
        min_size: Minimum image dimension
        
    Returns:
        Boolean indicating if image is valid
    """
    try:
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return False
            
        h, w = img.shape
        
        # Check minimum size
        if h < min_size or w < min_size:
            return False
            
        # Check if image has content (not all zeros or uniform)
        if img.std() < 5:  # Very low variance indicates poor quality
            return False
            
        return True
        
    except:
        return False

def prepare_flir_dataset(flir_root, output_dir, scale_factor=2, train_ratio=0.9, max_samples=None, min_size=128):
    """
    Prepare FLIR ADAS v2 dataset for IMDN training
    
    Args:
        flir_root: Root directory of FLIR dataset
        output_dir: Output directory for prepared dataset
        scale_factor: Super-resolution scale factor
        train_ratio: Ratio of data to use for training (vs validation)
        max_samples: Maximum number of samples to process (None for all)
        min_size: Minimum image size to accept
    """
    
    print("🔥 FLIR ADAS v2 Dataset Preparation for IMDN Fine-tuning")
    print("=" * 60)
    
    # Define paths
    flir_thermal_train = Path(flir_root) / "FLIR_ADAS_v2" / "THERMAL" / "images_thermal_train" / "data"
    flir_thermal_val = Path(flir_root) / "FLIR_ADAS_v2" / "THERMAL" / "images_thermal_val" / "data"
    
    # Check if FLIR dataset exists
    if not flir_thermal_train.exists():
        raise FileNotFoundError(f"FLIR thermal training data not found at {flir_thermal_train}")
    
    if not flir_thermal_val.exists():
        raise FileNotFoundError(f"FLIR thermal validation data not found at {flir_thermal_val}")
    
    # Create output directories
    output_path = Path(output_dir)
    train_hr_dir = output_path / "train" / "HR"
    train_lr_dir = output_path / "train" / f"LR_bicubic" / f"X{scale_factor}"
    val_hr_dir = output_path / "val" / "HR"
    val_lr_dir = output_path / "val" / f"LR_bicubic" / f"X{scale_factor}"
    
    for dir_path in [train_hr_dir, train_lr_dir, val_hr_dir, val_lr_dir]:
        ensure_dir(dir_path)
    
    # Collect all thermal images
    print("📂 Scanning FLIR thermal images...")
    train_images = list(flir_thermal_train.glob("*.jpg"))
    val_images = list(flir_thermal_val.glob("*.jpg"))
    
    print(f"Found {len(train_images)} training images")
    print(f"Found {len(val_images)} validation images")
    
    # Combine and shuffle for better distribution
    all_images = [(img, 'train') for img in train_images] + [(img, 'val') for img in val_images]
    random.shuffle(all_images)
    
    # Limit samples if specified
    if max_samples and len(all_images) > max_samples:
        all_images = all_images[:max_samples]
        print(f"Limited to {max_samples} samples")
    
    # Split into train/val based on ratio
    split_idx = int(len(all_images) * train_ratio)
    final_train = all_images[:split_idx]
    final_val = all_images[split_idx:]
    
    print(f"Final split: {len(final_train)} train, {len(final_val)} val")
    print()
    
    # Process training images
    print("🚀 Processing training images...")
    successful_train = 0
    for img_path, _ in tqdm(final_train, desc="Train"):
        if not is_valid_thermal_image(img_path, min_size):
            continue
            
        # Load thermal image
        hr_img = load_thermal_image(img_path)
        if hr_img is None:
            continue
        
        # Ensure image size is divisible by scale factor
        h, w = hr_img.shape
        h = (h // scale_factor) * scale_factor
        w = (w // scale_factor) * scale_factor
        hr_img = hr_img[:h, :w]
        
        if h < min_size or w < min_size:
            continue
        
        # Generate LR image
        lr_img = generate_lr_image(hr_img, scale_factor)
        
        # Save with sanitized filename
        img_name = img_path.stem + ".png"  # Convert to PNG for consistency
        hr_path = train_hr_dir / img_name
        lr_path = train_lr_dir / img_name
        
        try:
            cv2.imwrite(str(hr_path), hr_img)
            cv2.imwrite(str(lr_path), lr_img)
            successful_train += 1
        except Exception as e:
            print(f"Error saving {img_name}: {e}")
    
    # Process validation images
    print("🔍 Processing validation images...")
    successful_val = 0
    for img_path, _ in tqdm(final_val, desc="Val"):
        if not is_valid_thermal_image(img_path, min_size):
            continue
            
        # Load thermal image
        hr_img = load_thermal_image(img_path)
        if hr_img is None:
            continue
        
        # Ensure image size is divisible by scale factor
        h, w = hr_img.shape
        h = (h // scale_factor) * scale_factor
        w = (w // scale_factor) * scale_factor
        hr_img = hr_img[:h, :w]
        
        if h < min_size or w < min_size:
            continue
        
        # Generate LR image
        lr_img = generate_lr_image(hr_img, scale_factor)
        
        # Save with sanitized filename
        img_name = img_path.stem + ".png"
        hr_path = val_hr_dir / img_name
        lr_path = val_lr_dir / img_name
        
        try:
            cv2.imwrite(str(hr_path), hr_img)
            cv2.imwrite(str(lr_path), lr_img)
            successful_val += 1
        except Exception as e:
            print(f"Error saving {img_name}: {e}")
    
    # Print summary
    print()
    print("✅ Dataset preparation complete!")
    print("=" * 60)
    print(f"📊 Summary:")
    print(f"   • Training images: {successful_train}")
    print(f"   • Validation images: {successful_val}")
    print(f"   • Scale factor: {scale_factor}x")
    print(f"   • Output directory: {output_dir}")
    print()
    print(f"📁 Directory structure:")
    print(f"   {output_dir}/")
    print(f"   ├── train/")
    print(f"   │   ├── HR/           ({successful_train} images)")
    print(f"   │   └── LR_bicubic/X{scale_factor}/  ({successful_train} images)")
    print(f"   └── val/")
    print(f"       ├── HR/           ({successful_val} images)")
    print(f"       └── LR_bicubic/X{scale_factor}/    ({successful_val} images)")
    print()
    print("🎯 Ready for IMDN fine-tuning!")
    print("   Next steps:")
    print(f"   1. Run: python finetune_thermal.py --hr_dir {output_dir}/train/HR --lr_dir {output_dir}/train/LR_bicubic/X{scale_factor} --val_hr_dir {output_dir}/val/HR --val_lr_dir {output_dir}/val/LR_bicubic/X{scale_factor} --pretrained checkpoints/IMDN_x{scale_factor}.pth --scale {scale_factor}")
    
    # Validate a few samples
    print()
    print("🔍 Validation check:")
    sample_hr = train_hr_dir / os.listdir(train_hr_dir)[0] if os.listdir(train_hr_dir) else None
    sample_lr = train_lr_dir / os.listdir(train_lr_dir)[0] if os.listdir(train_lr_dir) else None
    
    if sample_hr and sample_lr:
        hr_img = cv2.imread(str(sample_hr), cv2.IMREAD_GRAYSCALE)
        lr_img = cv2.imread(str(sample_lr), cv2.IMREAD_GRAYSCALE)
        
        print(f"   • Sample HR image: {hr_img.shape} (min: {hr_img.min()}, max: {hr_img.max()})")
        print(f"   • Sample LR image: {lr_img.shape} (min: {lr_img.min()}, max: {lr_img.max()})")
        print(f"   • Scale ratio: {hr_img.shape[0] / lr_img.shape[0]:.1f}x")
        print("   ✅ Images look good!")
    
    return successful_train, successful_val

def main():
    parser = argparse.ArgumentParser(description='Prepare FLIR ADAS v2 thermal dataset for IMDN fine-tuning')
    parser.add_argument('--flir_root', type=str, required=True,
                        help='Root directory of FLIR-dataset')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for prepared dataset')
    parser.add_argument('--scale', type=int, default=2, choices=[2, 3, 4],
                        help='Super-resolution scale factor')
    parser.add_argument('--train_ratio', type=float, default=0.9,
                        help='Ratio of data to use for training (default: 0.9)')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to process (default: all)')
    parser.add_argument('--min_size', type=int, default=128,
                        help='Minimum image size to accept (default: 128)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    try:
        prepare_flir_dataset(
            flir_root=args.flir_root,
            output_dir=args.output_dir,
            scale_factor=args.scale,
            train_ratio=args.train_ratio,
            max_samples=args.max_samples,
            min_size=args.min_size
        )
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())