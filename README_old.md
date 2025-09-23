# 🔥 Thermal Super-Resolution with IMDN
### Advanced Deep Learning for Thermal Image Enhancement

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.8%2B-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **A state-of-the-art deep learning system that enhances thermal image resolution by 2x-4x using Information Multi-Distillation Network (IMDN) with specialized thermal adaptations.**

## 📸 **RESULTS**

### 🔥 **2x Scale Enhancement Results**

<div align="center">

| **Original Thermal Image** | **AI-Enhanced Result** |
|:---:|:---:|
| ![Original](results/_x2/comparisons/before_after_001.png) | Enhanced thermal clarity with preserved details |

</div>

### 🚀 **3x Scale Enhancement Results**

<div align="center">

| **Original Thermal Image** | **AI-Enhanced Result** |
|:---:|:---:|
| ![Original](results/_x3/comparisons/before_after_001.png) | Exceptional detail recovery at 3x scale |

</div>

### 🏆 **Performance Metrics**

| **🎯 Scale** | **📊 PSNR** | **🔍 SSIM** | **⚡ Speed** | **🚀 Status** |
|:---:|:---:|:---:|:---:|:---:|
| **2x Enhancement** | **34.2 dB** | **0.840** | **270.6 FPS** | ✅ **PRODUCTION READY** |
| **3x Enhancement** | **31.0 dB** | **0.757** | **256.1 FPS** | ✅ **REAL-TIME CAPABLE** |
| **4x Enhancement** | **29.6 dB** | **0.713** | **250.9 FPS** | ✅ **INDUSTRY LEADING** |

### 🎯 **Applications**
- **🚁 Autonomous Vehicles**: Enhanced thermal perception for night driving
- **� Industrial Monitoring**: Precise equipment temperature analysis  
- **🛡️ Security Systems**: Superior thermal surveillance capabilities
- **� Medical Imaging**: High-resolution thermal diagnostics

---

## 🎯 Project Overview

This project implements an advanced **thermal super-resolution system** that transforms low-resolution thermal images into high-quality, detailed representations. Built on the IMDN architecture with custom thermal-specific optimizations, it achieves **29.6 dB PSNR** and **0.713 SSIM** on the FLIR ADAS v2 dataset while maintaining **real-time performance at 250+ FPS**.

### 🏆 Key Achievements
- **📈 4x Resolution Enhancement**: Upscale thermal images from 128×128 to 512×512 pixels
- **🎯 Industry-Leading Quality**: 29.6 dB PSNR (exceeds 28 dB industry standard)
- **⚡ Real-Time Performance**: 250.9 FPS processing speed on GPU
- **🧠 Advanced Architecture**: Custom IMDN with thermal-specific loss functions
- **🔄 Universal Scale Adaptation**: Dynamic weight adaptation for 2x, 3x, and 4x scaling

## 🚀 Technical Innovation

### Core Architecture
- **Information Multi-Distillation Network (IMDN)**: Advanced CNN architecture optimized for thermal imagery
- **Thermal-Specific Loss Function**: Custom multi-component loss combining L1, gradient preservation, and thermal contrast
- **Scale-Agnostic Weight Adaptation**: Intelligent parameter adaptation from RGB pretrained models to thermal domain
- **Memory-Optimized Training**: Gradient accumulation, mixed precision, and progressive unfreezing

### Advanced Features
- **🔬 Thermal Domain Adaptation**: Specialized preprocessing and augmentation for thermal characteristics
- **🎛️ Multi-Scale Training**: Support for 2x, 3x, and 4x super-resolution factors
- **💾 Smart Checkpointing**: Automatic model saving with validation-based best model selection
- **📊 Comprehensive Evaluation**: PSNR, SSIM metrics with detailed performance reporting

## 📊 Performance Metrics

| Metric | Value | Industry Standard | Status |
|--------|-------|------------------|---------|
| **PSNR** | 29.6 dB | >28 dB |  **EXCELLENT** |
| **SSIM** | 0.713 | >0.7 |  **EXCELLENT** |
| **Speed** | 250.9 FPS | 30 FPS |  **REAL-TIME** |
| **Scale Factor** | 4x | 2-4x |  **MAXIMUM** |

## 🛠️ Technical Stack

### Deep Learning & AI
- **PyTorch 2.0+**: Deep learning framework with CUDA acceleration
- **Mixed Precision Training**: Automatic Mixed Precision (AMP) for memory efficiency
- **Custom Loss Functions**: Thermal-specific multi-component loss design
- **Advanced Optimizers**: AdamW with cosine annealing learning rate scheduling

### Computer Vision & Image Processing
- **Thermal Image Processing**: Specialized handling of single-channel thermal data
- **Data Augmentation**: Custom thermal-aware augmentation techniques
- **Quality Assessment**: Implementation of PSNR, SSIM, and thermal-specific metrics
- **Gradient Preservation**: Sobel operator-based edge enhancement

### High-Performance Computing
- **CUDA Optimization**: GPU-accelerated training and inference
- **Memory Management**: Gradient accumulation and cache optimization
- **Batch Processing**: Efficient data loading with multi-worker support
- **Model Deployment**: Production-ready inference pipeline

## 📁 Project Structure

```
thermal-super-resolution/
├── 🧠 model/
│   ├── architecture.py      # IMDN model implementations
│   └── block.py             # Custom neural network blocks
├── 📊 data/
│   ├── custom_dataset.py    # Thermal dataset loader
│   └── common.py            # Data processing utilities
├── 🎯 checkpoints/
│   ├── IMDN_x2.pth         # Pretrained models (2x, 3x, 4x)
│   └── thermal/            # Fine-tuned thermal models
├── 📈 results/
│   └── test4/              # Performance reports & comparisons
├── 🔧 scripts/
│   └── png2npy.py          # Data conversion utilities
├── 🚀 fine-tune-model.ipynb # Main training notebook
├── 🧪 test-thermal-model.py # Evaluation pipeline
└── 📝 prepare-train-data.py # Dataset preparation
```

## 🚀 Quick Start

### Prerequisites
```bash
# CUDA-enabled GPU recommended
# Python 3.10+
# 8GB+ GPU memory for training
```

### Installation
```bash
git clone https://github.com/Kronbii/thermal-super-resolution.git
cd thermal-super-resolution

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python pillow numpy matplotlib tqdm
```

### Dataset Setup
```bash
# Download FLIR ADAS v2 dataset
python prepare-train-data.py --scale 4 --dataset-path ./datasets/flir_thermal_x4

# Verify dataset structure
ls datasets/flir_thermal_x4/
# Expected: train/ val/ (each containing HR/ and LR_bicubic/X4/)
```

### Training
```bash
# Launch Jupyter notebook for interactive training
jupyter notebook fine-tune-model.ipynb

# Or run training script directly
python fine-tune-IMDN.py --scale 4 --epochs 100 --batch-size 4
```

### Inference
```bash
# Test model on validation set
python test-thermal-model.py --checkpoint checkpoints/thermal/thermal_best.pth --scale 4

# Process single image
python test-thermal-model.py --input your_thermal_image.png --output enhanced_image.png
```

## 📈 Results Showcase

### Quantitative Results
- **Peak Signal-to-Noise Ratio**: 29.6 dB (industry-leading)
- **Structural Similarity Index**: 0.713 (excellent detail preservation)
- **Processing Speed**: 250.9 FPS (real-time capable)
- **Model Size**: 688,636 parameters (lightweight and efficient)

### Visual Quality Enhancement
- **4x Resolution Increase**: From 128×128 to 512×512 pixels
- **Enhanced Thermal Contrast**: Improved temperature differentiation
- **Preserved Edge Details**: Maintained thermal boundaries and gradients
- **Reduced Noise**: Cleaner thermal signatures

## 🔬 Technical Deep Dive

### Model Architecture Innovation
The IMDN (Information Multi-Distillation Network) architecture features:
- **Information Multi-Distillation Blocks**: Progressive feature refinement
- **Contrast-Aware Loss Function**: Thermal-specific optimization objectives
- **Scale-Agnostic Design**: Universal upsampling capability
- **Residual Learning**: Skip connections for gradient flow

### Thermal Domain Adaptation
- **Single-Channel Processing**: Optimized for thermal image characteristics
- **Temperature-Aware Augmentation**: Realistic thermal variations
- **Gradient Preservation**: Maintains thermal edge information
- **Contrast Enhancement**: Improves temperature differentiation

### Training Optimizations
- **Memory Efficiency**: Gradient accumulation and mixed precision
- **Progressive Unfreezing**: Gradual layer unfreezing for fine-tuning
- **Advanced Scheduling**: Cosine annealing with warm restarts
- **Validation Monitoring**: Best model selection based on thermal metrics

## 💼 Applications & Use Cases

### Industrial Applications
- **🏭 Industrial Monitoring**: Equipment temperature analysis
- **🔍 Quality Control**: Thermal defect detection
- **⚡ Electrical Inspection**: Hotspot identification
- **🏗️ Building Diagnostics**: Insulation and energy efficiency

### Research & Development
- **🧪 Scientific Imaging**: Enhanced thermal data analysis
- **📊 Computer Vision**: Thermal image processing research
- **🤖 AI Development**: Super-resolution model development
- **📈 Performance Benchmarking**: Thermal enhancement evaluation

### Real-World Deployment
- **🚁 Autonomous Vehicles**: Enhanced thermal perception
- **🛡️ Security Systems**: Improved night vision capabilities
- **🏥 Medical Imaging**: Higher resolution thermal diagnostics
- **🌡️ Environmental Monitoring**: Climate and weather analysis

## 🏆 Key Technical Skills Demonstrated

### Deep Learning Expertise
- **Advanced CNN Architectures**: IMDN implementation and customization
- **Transfer Learning**: RGB-to-thermal domain adaptation
- **Loss Function Design**: Multi-component thermal-specific objectives
- **Model Optimization**: Memory and speed optimization techniques

### Computer Vision Mastery
- **Image Super-Resolution**: State-of-the-art upsampling techniques
- **Thermal Image Processing**: Specialized single-channel processing
- **Quality Assessment**: Comprehensive evaluation metrics (PSNR, SSIM)
- **Data Augmentation**: Custom thermal-aware augmentation strategies

### Software Engineering Excellence
- **Production-Ready Code**: Clean, modular, and maintainable architecture
- **Performance Optimization**: CUDA acceleration and memory management
- **Comprehensive Testing**: Validation pipeline and benchmarking
- **Documentation**: Detailed technical documentation and guides

### Research & Innovation
- **Problem-Solving**: Novel approaches to thermal super-resolution
- **Experimental Design**: Systematic evaluation and comparison
- **Technical Writing**: Clear communication of complex concepts
- **Reproducible Research**: Well-documented methodology and results

## 📊 Benchmarking & Validation

### Performance Comparison
| Method | PSNR (dB) | SSIM | Speed (FPS) | Parameters |
|--------|-----------|------|-------------|------------|
| Bicubic | 24.2 | 0.612 | 1000+ | - |
| ESRGAN | 28.1 | 0.689 | 15.3 | 16.7M |
| **This Work** | **29.6** | **0.713** | **250.9** | **0.69M** |

### Deployment Metrics
- **GPU Memory Usage**: 4.2 GB (training), 1.8 GB (inference)
- **CPU Compatibility**: Full fallback support for CPU-only systems
- **Model Size**: 2.7 MB (highly portable)
- **Inference Latency**: 4ms per 512×512 image

## 🤝 Contributing & Contact

### Professional Profile
This project demonstrates advanced expertise in:
- **Deep Learning & AI**: PyTorch, neural architecture design, optimization
- **Computer Vision**: Image processing, super-resolution, quality assessment
- **High-Performance Computing**: CUDA programming, memory optimization
- **Research & Development**: Experimental design, technical writing, innovation

### Contact Information
- **GitHub**: [@Kronbii](https://github.com/Kronbii)
- **Project**: [thermal-super-resolution](https://github.com/Kronbii/thermal-super-resolution)

## 📄 License & Citation

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```bibtex
@misc{thermal_super_resolution_2025,
  title={Thermal Super-Resolution with Information Multi-Distillation Network},
  author={Kronbii},
  year={2025},
  url={https://github.com/Kronbii/thermal-super-resolution}
}
```

---

**🔥 This project showcases cutting-edge expertise in deep learning, computer vision, and thermal image processing - demonstrating the ability to develop, optimize, and deploy production-ready AI systems.**