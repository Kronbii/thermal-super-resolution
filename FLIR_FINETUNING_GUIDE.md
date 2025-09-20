## 🚀 Quick Start

### Step 1: Prepare the FLIR Dataset

```bash
# Navigate to your IMDN directory
cd /home/kronbii/repos/IMDN

# Prepare the FLIR dataset for training (this will take 5-10 minutes)
python prepare_flir_data.py \
    --flir_root /home/kronbii/repos/IMDN/FLIR-dataset \
    --output_dir ./datasets/flir_thermal_x2 \
    --scale 2


### Step 2: Start Fine-tuning

```bash
# Fine-tune for 2x thermal super-resolution
python finetune_flir.py \
    --flir_dataset_dir ./datasets/flir_thermal_x2 \
    --pretrained checkpoints/IMDN_x2.pth \
    --scale 2 \
    --epochs 80 \
    --batch_size 16

# Fine-tune for 4x thermal super-resolution  
python finetune_flir.py \
    --flir_dataset_dir ./datasets/flir_thermal_x4 \
    --pretrained checkpoints/IMDN_x4.pth \
    --scale 4 \
    --epochs 100 \
    --batch_size 8
```

### Step 3: Test Your Model

```bash
# Test the fine-tuned thermal model
python test_thermal_model.py \
    --model checkpoints/thermal/thermal_best.pth \
    --test_dir ./datasets/flir_thermal_x2/val/LR_bicubic/X2 \
    --scale 2 \
    --save_results
```

## 📊 FLIR Dataset Specifications

### Dataset Statistics
- **Training Images**: 10,742 thermal images
- **Validation Images**: 1,144 thermal images  
- **Image Format**: 8-bit thermal images (JPEG)
- **Resolution**: Varies (typically 512x640, 480x640)
- **Domain**: Automotive thermal imaging (ADAS)
- **Content**: Street scenes, vehicles, pedestrians, buildings

### Thermal Characteristics
- **Temperature Range**: Automotive environment temperatures
- **Sensor Type**: Uncooled microbolometer LWIR camera
- **Wavelength**: Long-wave infrared (8-14 μm)
- **Applications**: Autonomous driving, night vision, pedestrian detection

## 🛠️ Advanced Configuration

### Data Preparation Options

```bash
# Prepare with specific settings
python prepare_flir_data.py \
    --flir_root /home/kronbii/repos/IMDN/FLIR-dataset \
    --output_dir ./datasets/flir_custom \
    --scale 2 \
    --train_ratio 0.85 \           # Use 85% for training, 15% for validation
    --max_samples 5000 \           # Limit to 5000 samples for faster training
    --min_size 128 \               # Minimum image size (pixels)
    --seed 42                      # Random seed for reproducibility
```

### Fine-tuning Options

```bash
# Advanced fine-tuning configuration
python finetune_flir.py \
    --flir_dataset_dir ./datasets/flir_thermal_x2 \
    --pretrained checkpoints/IMDN_x2.pth \
    --scale 2 \
    \
    # Training parameters
    --epochs 100 \
    --batch_size 16 \
    --lr 1e-5 \
    --weight_decay 1e-4 \
    \
    # Gradual unfreezing strategy
    --gradual_unfreeze \
    --freeze_epochs 20 \
    \
    # Thermal-specific loss weights
    --l1_weight 1.0 \
    --gradient_weight 0.1 \
    --thermal_weight 0.05 \
    \
    # System optimization
    --mixed_precision \
    --num_workers 6 \
    --checkpoint_dir ./checkpoints/flir_thermal
```

## 🎯 Expected Results

### Performance Expectations
- **Training Time**: 3-6 hours (depending on hardware and settings)
- **PSNR Improvement**: 2-5 dB over bicubic interpolation
- **Visual Quality**: Significantly enhanced thermal details and edge preservation
- **Convergence**: Typically converges within 60-80 epochs

### Hardware Recommendations
- **GPU Memory**: 8GB+ VRAM recommended
- **RAM**: 16GB+ system RAM
- **Storage**: 5GB free space for prepared dataset
- **CPU**: Multi-core processor for data loading

## 🔧 Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory
```bash
# Reduce batch size
python finetune_flir.py ... --batch_size 8

# Or reduce image processing workers
python finetune_flir.py ... --num_workers 2
```

#### 2. Dataset Path Issues
```bash
# Verify FLIR dataset structure
ls -la /home/kronbii/repos/IMDN/FLIR-dataset/FLIR_ADAS_v2/THERMAL/

# Should show:
# images_thermal_train/
# images_thermal_val/
```

#### 3. Slow Training
```bash
# Enable mixed precision (automatic on CUDA)
python finetune_flir.py ... --mixed_precision

# Increase batch size if GPU memory allows
python finetune_flir.py ... --batch_size 32

# Use more workers for data loading
python finetune_flir.py ... --num_workers 8
```

#### 4. Poor Convergence
```bash
# Try different learning rate
python finetune_flir.py ... --lr 2e-5

# Adjust loss weights for thermal images
python finetune_flir.py ... --gradient_weight 0.2 --thermal_weight 0.1

# Increase training epochs
python finetune_flir.py ... --epochs 120
```

## 📈 Monitoring Training

### Key Metrics to Watch
- **L1 Loss**: Should decrease steadily (target: < 0.01)
- **Gradient Loss**: Preserves thermal edges (target: < 0.005)
- **Thermal Loss**: Maintains thermal contrast (target: < 0.002)
- **Validation PSNR**: Should improve over epochs (target: > 28 dB)

### Training Phases
1. **Phase 1 (Epochs 0-20)**: Frozen backbone, fast adaptation
2. **Phase 2 (Epochs 20+)**: Full model training, fine-tuning
3. **Convergence**: Usually around epoch 60-80

### Example Training Log
```
Epoch  50 [1200/1500 (80.0%)] Loss: 0.012456 L1: 0.011234 Grad: 0.001012 Thermal: 0.000210 LR: 5.23e-06
Epoch  50 Summary: Loss=0.013124 (L1:0.0118, Grad:0.0011, Thermal:0.0002) Val_PSNR=29.45dB Best=29.45dB Time=2.3h
💫 New best model saved! PSNR: 29.45
```

## 🔍 Validation and Testing

### Quick Validation
```bash
# Validate current best model
python validate_thermal.py \
    --model checkpoints/flir_thermal/thermal_best.pth \
    --val_dir ./datasets/flir_thermal_x2/val \
    --scale 2
```

### Visual Inspection
```bash
# Generate sample results for visual inspection
python test_thermal_samples.py \
    --model checkpoints/flir_thermal/thermal_best.pth \
    --input_dir ./datasets/flir_thermal_x2/val/LR_bicubic/X2 \
    --output_dir ./results/flir_samples \
    --scale 2 \
    --num_samples 20
```

## 📁 File Structure After Setup

```
IMDN/
├── FLIR-dataset/                    # Your original FLIR dataset
├── datasets/
│   └── flir_thermal_x2/            # Prepared dataset
│       ├── train/
│       │   ├── HR/                  # High-res training images
│       │   └── LR_bicubic/X2/      # Low-res training images
│       └── val/
│           ├── HR/                  # High-res validation images
│           └── LR_bicubic/X2/      # Low-res validation images
├── checkpoints/
│   └── flir_thermal/               # Training checkpoints
│       ├── thermal_best.pth        # Best model
│       └── thermal_epoch_*.pth     # Regular checkpoints
├── prepare_flir_data.py            # FLIR dataset preparation
├── finetune_flir.py               # FLIR-optimized fine-tuning
└── results/                       # Generated results
```

## 🎛️ Advanced Tips

### 1. Custom Loss Weights
Adjust loss weights based on your thermal imagery characteristics:
```bash
# For high-contrast thermal scenes
python finetune_flir.py ... --gradient_weight 0.2 --thermal_weight 0.1

# For subtle thermal variations
python finetune_flir.py ... --gradient_weight 0.05 --thermal_weight 0.15
```

### 2. Multi-Scale Training
Train models for different scales:
```bash
# Train 2x, 3x, and 4x models
for scale in 2 3 4; do
    python prepare_flir_data.py --scale $scale --output_dir ./datasets/flir_x${scale}
    python finetune_flir.py --scale $scale --flir_dataset_dir ./datasets/flir_x${scale}
done
```

### 3. Resume Training
```bash
# Resume from a checkpoint
python finetune_flir.py \
    --resume checkpoints/flir_thermal/thermal_epoch_50.pth \
    --epochs 100
```

### 4. Ensemble Models
Train multiple models with different seeds for ensemble:
```bash
# Train 3 models with different seeds
for seed in 42 123 456; do
    python finetune_flir.py --seed $seed --checkpoint_dir ./checkpoints/flir_${seed}
done
```

## 🏆 Best Practices

### 1. **Start with 2x Scale**: Easier to train and debug
### 2. **Use Gradual Unfreezing**: Better transfer learning from RGB pretrained models
### 3. **Monitor All Loss Components**: Not just total loss
### 4. **Save Regular Checkpoints**: Training can be interrupted
### 5. **Validate Frequently**: Catch overfitting early
### 6. **Use Mixed Precision**: Faster training on modern GPUs

## 🚀 Next Steps

After successful fine-tuning:

1. **Evaluate on Test Set**: Use separate thermal images for final evaluation
2. **Deploy Model**: Integrate into your thermal imaging pipeline
3. **Further Fine-tuning**: Use domain-specific thermal data if available
4. **Model Optimization**: Quantization, pruning for deployment

---

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Verify your FLIR dataset structure
3. Ensure sufficient GPU memory and disk space
4. Try reducing batch size or image resolution

Your FLIR ADAS v2 dataset is perfect for thermal super-resolution fine-tuning! 🔥