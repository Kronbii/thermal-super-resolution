#!/usr/bin/env python3
"""
IMDN Thermal Fine-tuning Script - FLIR ADAS v2 Optimized

Fine-tune IMDN for thermal super-resolution using the FLIR ADAS v2 dataset.
This script is optimized for thermal imagery characteristics and FLIR dataset structure.

Features:
- Thermal-specific data loading and preprocessing
- Gradual unfreezing for better transfer learning
- FLIR dataset integration
- Advanced thermal image augmentations
- Comprehensive logging and monitoring
- Automatic mixed precision training
- Thermal-specific loss functions

Usage:
    python finetune_flir.py --flir_dataset_dir /path/to/prepared_flir_data --pretrained checkpoints/IMDN_x2.pth --scale 2
"""

import os
import sys
import argparse
import time
import math
import random
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

# Add the current directory to path to import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model.architecture import IMDN
from data.custom_dataset import ThermalDataset

def set_random_seed(seed=42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class ThermalLoss(nn.Module):
    """
    Custom loss function optimized for thermal images
    Combines L1 loss with thermal-specific perceptual components
    """
    def __init__(self, l1_weight=1.0, gradient_weight=0.1, thermal_weight=0.05):
        super(ThermalLoss, self).__init__()
        self.l1_weight = l1_weight
        self.gradient_weight = gradient_weight
        self.thermal_weight = thermal_weight
        self.l1_loss = nn.L1Loss()
        
    def gradient_loss(self, pred, target):
        """Calculate gradient loss to preserve thermal edges"""
        # Sobel operators for gradient calculation
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(pred.device)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(pred.device)
        
        # Calculate gradients
        pred_grad_x = F.conv2d(pred, sobel_x, padding=1)
        pred_grad_y = F.conv2d(pred, sobel_y, padding=1)
        target_grad_x = F.conv2d(target, sobel_x, padding=1)
        target_grad_y = F.conv2d(target, sobel_y, padding=1)
        
        # L1 loss on gradients
        grad_loss = self.l1_loss(pred_grad_x, target_grad_x) + self.l1_loss(pred_grad_y, target_grad_y)
        return grad_loss
    
    def thermal_contrast_loss(self, pred, target):
        """Loss to preserve thermal contrast characteristics"""
        # Calculate local variance to preserve thermal texture
        kernel = torch.ones(1, 1, 3, 3).to(pred.device) / 9.0
        
        pred_mean = F.conv2d(pred, kernel, padding=1)
        target_mean = F.conv2d(target, kernel, padding=1)
        
        pred_var = F.conv2d((pred - pred_mean)**2, kernel, padding=1)
        target_var = F.conv2d((target - target_mean)**2, kernel, padding=1)
        
        contrast_loss = self.l1_loss(pred_var, target_var)
        return contrast_loss
    
    def forward(self, pred, target):
        # Main L1 loss
        l1 = self.l1_loss(pred, target)
        
        # Gradient preservation loss
        grad = self.gradient_loss(pred, target)
        
        # Thermal contrast loss
        thermal = self.thermal_contrast_loss(pred, target)
        
        total_loss = (self.l1_weight * l1 + 
                     self.gradient_weight * grad + 
                     self.thermal_weight * thermal)
        
        return total_loss, {'l1': l1.item(), 'gradient': grad.item(), 'thermal': thermal.item()}

def freeze_layers(model, freeze_backbone=True):
    """Freeze/unfreeze model layers for gradual training"""
    for name, param in model.named_parameters():
        if freeze_backbone and not any(layer in name.lower() for layer in ['upsampler', 'lr_conv', 'fea_conv']):
            param.requires_grad = False
        else:
            param.requires_grad = True

def calculate_psnr(img1, img2, max_val=1.0):
    """Calculate PSNR between two images"""
    mse = torch.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(max_val / torch.sqrt(mse))

def validate_model(model, val_loader, criterion, device, max_batches=50):
    """Validate the model on validation set"""
    model.eval()
    total_loss = 0
    total_psnr = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch_idx, (lr, hr) in enumerate(val_loader):
            if batch_idx >= max_batches:
                break
                
            lr, hr = lr.to(device), hr.to(device)
            
            # Forward pass
            with autocast():
                sr = model(lr)
                loss, loss_components = criterion(sr, hr)
            
            # Calculate PSNR
            psnr = calculate_psnr(sr, hr)
            
            total_loss += loss.item()
            total_psnr += psnr.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    avg_psnr = total_psnr / num_batches
    
    return avg_loss, avg_psnr

def save_checkpoint(model, optimizer, epoch, loss, psnr, checkpoint_dir, is_best=False):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'psnr': psnr,
    }
    
    # Save regular checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f'thermal_epoch_{epoch}.pth')
    torch.save(checkpoint, checkpoint_path)
    
    # Save best model
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'thermal_best.pth')
        torch.save(checkpoint, best_path)
        print(f"💫 New best model saved! PSNR: {psnr:.2f}")

def print_model_info(model, sample_input):
    """Print model information"""
    model.eval()
    with torch.no_grad():
        output = model(sample_input)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"📊 Model Information:")
    print(f"   • Total parameters: {total_params:,}")
    print(f"   • Trainable parameters: {trainable_params:,}")
    print(f"   • Input shape: {sample_input.shape}")
    print(f"   • Output shape: {output.shape}")
    print()

def main():
    parser = argparse.ArgumentParser(description='Fine-tune IMDN for thermal super-resolution using FLIR dataset')
    
    # Dataset arguments
    parser.add_argument('--flir_dataset_dir', type=str, required=True,
                        help='Directory containing prepared FLIR dataset')
    parser.add_argument('--scale', type=int, default=2, choices=[2, 3, 4],
                        help='Super-resolution scale factor')
    
    # Model arguments
    parser.add_argument('--pretrained', type=str, required=True,
                        help='Path to pretrained IMDN model')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    
    # Training strategy arguments
    parser.add_argument('--gradual_unfreeze', action='store_true', default=True,
                        help='Use gradual unfreezing strategy')
    parser.add_argument('--freeze_epochs', type=int, default=20,
                        help='Number of epochs to train with frozen backbone')
    
    # Loss function arguments
    parser.add_argument('--l1_weight', type=float, default=1.0,
                        help='Weight for L1 loss')
    parser.add_argument('--gradient_weight', type=float, default=0.1,
                        help='Weight for gradient loss')
    parser.add_argument('--thermal_weight', type=float, default=0.05,
                        help='Weight for thermal contrast loss')
    
    # System arguments
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loader workers')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (auto, cuda, cpu)')
    parser.add_argument('--mixed_precision', action='store_true', default=True,
                        help='Use automatic mixed precision training')
    
    # Output arguments
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/thermal',
                        help='Directory to save checkpoints')
    parser.add_argument('--log_interval', type=int, default=50,
                        help='Logging interval')
    parser.add_argument('--val_interval', type=int, default=5,
                        help='Validation interval (epochs)')
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed
    set_random_seed(args.seed)
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print("🔥 IMDN Thermal Fine-tuning - FLIR ADAS v2 Optimized")
    print("=" * 60)
    print(f"🖥️  Device: {device}")
    print(f"📊 Scale factor: {args.scale}x")
    print(f"📁 Dataset: {args.flir_dataset_dir}")
    print(f"🏋️ Pretrained model: {args.pretrained}")
    print()
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Setup datasets
    print("📂 Loading FLIR thermal dataset...")
    
    # Training dataset
    train_hr_dir = os.path.join(args.flir_dataset_dir, 'train', 'HR')
    train_lr_dir = os.path.join(args.flir_dataset_dir, 'train', f'LR_bicubic', f'X{args.scale}')
    
    # Create options object for ThermalDataset
    class TrainOpt:
        def __init__(self):
            self.scale = args.scale
            self.phase = 'train'
            self.hr_dir = train_hr_dir
            self.lr_dir = train_lr_dir
            self.ext = '.png'
            self.augment = True
            self.thermal_augment = True
    
    train_dataset = ThermalDataset(TrainOpt())
    
    # Validation dataset  
    val_hr_dir = os.path.join(args.flir_dataset_dir, 'val', 'HR')
    val_lr_dir = os.path.join(args.flir_dataset_dir, 'val', f'LR_bicubic', f'X{args.scale}')
    
    # Create options object for validation dataset
    class ValOpt:
        def __init__(self):
            self.scale = args.scale
            self.phase = 'val'
            self.hr_dir = val_hr_dir
            self.lr_dir = val_lr_dir
            self.ext = '.png'
            self.augment = False
            self.thermal_augment = False
    
    val_dataset = ThermalDataset(ValOpt())
    
    print(f"   • Training samples: {len(train_dataset)}")
    print(f"   • Validation samples: {len(val_dataset)}")
    
    # Setup data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Setup model
    print("🏗️ Setting up IMDN model...")
    model = IMDN(upscale=args.scale, in_nc=1, out_nc=1)  # Single channel for thermal
    
    # Load pretrained weights
    if os.path.exists(args.pretrained):
        print(f"📥 Loading pretrained weights from {args.pretrained}")
        checkpoint = torch.load(args.pretrained, map_location='cpu')
        
        # Extract state dict if it's wrapped in a checkpoint
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Remove 'module.' prefix if present (from DataParallel training)
        if any(key.startswith('module.') for key in state_dict.keys()):
            state_dict = {key.replace('module.', ''): value for key, value in state_dict.items()}
        
        # Adapt RGB pretrained model to thermal (3->1 channel)
        adapted_state_dict = {}
        for name, param in state_dict.items():
            if name == 'fea_conv.weight' and param.shape[1] == 3:  # RGB input layer
                # Average RGB channels to get single thermal channel
                adapted_param = param.mean(dim=1, keepdim=True)
                adapted_state_dict[name] = adapted_param
                print(f"   • Adapted {name}: {param.shape} -> {adapted_param.shape}")
            elif name == 'upsampler.0.weight' and param.shape[0] == 48:  # RGB output layer (3*16=48)
                # Take only first 16 channels for single thermal channel
                adapted_param = param[:16]  # Keep first 16 channels
                adapted_state_dict[name] = adapted_param
                print(f"   • Adapted {name}: {param.shape} -> {adapted_param.shape}")
            elif name == 'upsampler.0.bias' and param.shape[0] == 48:  # RGB output bias
                # Take only first 16 bias terms
                adapted_param = param[:16]
                adapted_state_dict[name] = adapted_param
                print(f"   • Adapted {name}: {param.shape} -> {adapted_param.shape}")
            else:
                adapted_state_dict[name] = param
        
        # Load adapted weights
        missing_keys, unexpected_keys = model.load_state_dict(adapted_state_dict, strict=False)
        if missing_keys:
            print(f"   ⚠️ Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"   ⚠️ Unexpected keys: {unexpected_keys}")
    else:
        print(f"⚠️ Pretrained model not found at {args.pretrained}, training from scratch")
    
    model = model.to(device)
    
    # Print model info
    sample_input = torch.randn(1, 1, 64, 64).to(device)
    print_model_info(model, sample_input)
    
    # Setup loss function
    criterion = ThermalLoss(
        l1_weight=args.l1_weight,
        gradient_weight=args.gradient_weight,
        thermal_weight=args.thermal_weight
    ).to(device)
    
    # Setup optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999)
    )
    
    # Setup learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.01
    )
    
    # Setup mixed precision training
    scaler = GradScaler() if args.mixed_precision and device.type == 'cuda' else None
    
    # Gradual unfreezing setup
    if args.gradual_unfreeze:
        print("🔐 Starting with frozen backbone (gradual unfreezing enabled)")
        freeze_layers(model, freeze_backbone=True)
    
    # Training loop
    print("🚀 Starting training...")
    print()
    
    best_psnr = 0
    start_time = time.time()
    
    for epoch in range(args.epochs):
        # Gradual unfreezing
        if args.gradual_unfreeze and epoch == args.freeze_epochs:
            print("🔓 Unfreezing backbone layers")
            freeze_layers(model, freeze_backbone=False)
            # Reduce learning rate when unfreezing
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5
        
        model.train()
        epoch_loss = 0
        epoch_l1 = 0
        epoch_gradient = 0
        epoch_thermal = 0
        
        for batch_idx, (lr, hr) in enumerate(train_loader):
            lr, hr = lr.to(device), hr.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if scaler is not None:
                with autocast():
                    sr = model(lr)
                    loss, loss_components = criterion(sr, hr)
                
                # Backward pass
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                sr = model(lr)
                loss, loss_components = criterion(sr, hr)
                loss.backward()
                optimizer.step()
            
            # Accumulate losses
            epoch_loss += loss.item()
            epoch_l1 += loss_components['l1']
            epoch_gradient += loss_components['gradient']
            epoch_thermal += loss_components['thermal']
            
            # Logging
            if batch_idx % args.log_interval == 0:
                progress = 100.0 * batch_idx / len(train_loader)
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch:3d} [{batch_idx:4d}/{len(train_loader)} ({progress:5.1f}%)] "
                      f"Loss: {loss.item():.6f} L1: {loss_components['l1']:.6f} "
                      f"Grad: {loss_components['gradient']:.6f} Thermal: {loss_components['thermal']:.6f} "
                      f"LR: {current_lr:.2e}")
        
        # Update learning rate
        scheduler.step()
        
        # Calculate epoch averages
        avg_loss = epoch_loss / len(train_loader)
        avg_l1 = epoch_l1 / len(train_loader)
        avg_gradient = epoch_gradient / len(train_loader)
        avg_thermal = epoch_thermal / len(train_loader)
        
        # Validation
        val_loss, val_psnr = 0, 0
        if epoch % args.val_interval == 0:
            val_loss, val_psnr = validate_model(model, val_loader, criterion, device)
            
            # Save checkpoint if best
            is_best = val_psnr > best_psnr
            if is_best:
                best_psnr = val_psnr
            
            save_checkpoint(model, optimizer, epoch, val_loss, val_psnr, args.checkpoint_dir, is_best)
        
        # Epoch summary
        elapsed = time.time() - start_time
        print(f"Epoch {epoch:3d} Summary: Loss={avg_loss:.6f} (L1:{avg_l1:.4f}, Grad:{avg_gradient:.4f}, Thermal:{avg_thermal:.4f}) "
              f"Val_PSNR={val_psnr:.2f}dB Best={best_psnr:.2f}dB Time={elapsed/60:.1f}min")
        print("-" * 100)
    
    # Final summary
    total_time = time.time() - start_time
    print()
    print("✅ Training completed!")
    print("=" * 60)
    print(f"🎯 Best validation PSNR: {best_psnr:.2f} dB")
    print(f"⏱️ Total training time: {total_time/3600:.2f} hours")
    print(f"💾 Best model saved at: {os.path.join(args.checkpoint_dir, 'thermal_best.pth')}")
    print()
    print("🔥 Your thermal super-resolution model is ready!")

if __name__ == '__main__':
    main()