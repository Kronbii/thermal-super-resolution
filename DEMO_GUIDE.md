# 🔥 Thermal Super-Resolution Demo Guide

Complete end-to-end guide for running thermal super-resolution demos and preparing data.

## 📁 File Structure Overview

### Demo Applications
- **`demo_app.py`** - 🎯 **Main GUI demo** (PyQt5 interface)
- **`launch-model-demo.py`** - 🚀 **Demo launcher** with error handling
- **`cli-model-demo.py`** - ⚡ **Fast terminal performance test**

---

## 🚀 Quick Start Demo

### Prerequisites
```bash
# Activate your conda environment
conda activate FLIR
```

### Option 1: GUI Demo
```bash
python3 launch-model-demo.py
```

**What this does:**
- Launches professional PyQt5 interface
- Side-by-side video comparison (before/after)
- Real-time FPS benchmarking
- Automatic fallback to terminal if GUI fails

### Option 2: Quick Terminal Test
```bash
python3 cli-model-demo.py
```

**What this does:**
- Tests all available models (2x, 3x, 4x)
- Measures inference speed (FPS)
- Shows GPU memory usage
- No GUI dependencies

### Option 3: Direct GUI Launch (Advanced)
```bash
python3 demo_app.py
```

**What this does:**
- Direct launch of main demo application
- Requires proper environment setup
- Best performance but less error handling

---

## 📝 Preparing Your Own Thermal Data

### Data Format Requirements
- **File Types**: `.png`, `.jpg`, `.tiff`, `.npy`
- **Bit Depth**: 8-bit or 16-bit grayscale
- **Image Size**: Any size (will be processed accordingly)
- **Temperature Range**: Normalized to 0-255 or raw thermal values

### Dataset Structure Setup
1. **Create your dataset folder:**
```bash
mkdir -p datasets/your_thermal_data/train/{HR,LR_bicubic}
mkdir -p datasets/your_thermal_data/val/{HR,LR_bicubic}
```

2. **Prepare High-Resolution (HR) images:**
   - Place your original thermal images in `HR/` folders
   - Should be the target resolution you want to achieve

---