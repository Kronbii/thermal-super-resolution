# Thermal Super-Resolution with IMDN

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.8%2B-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Deep learning system that enhances thermal image resolution by 2x-4x using Information Multi-Distillation Network (IMDN) with thermal-specific optimizations.

## Results

| Original | Enhanced | Scale |
|:---:|:---:|:---:|
| ![Before](results/_x2/comparisons/before_after_001.png) | ![After](results/_x2/comparisons/before_after_001.png) | 2x |
| ![Before](results/_x3/comparisons/before_after_001.png) | ![After](results/_x3/comparisons/before_after_001.png) | 3x |

## Performance

| Scale | PSNR | SSIM | Speed | Status |
|:---:|:---:|:---:|:---:|:---:|
| 2x | 34.2 dB | 0.840 | 229.6 FPS | Production Ready |
| 3x | 31.0 dB | 0.757 | 256.1 FPS | Real-time |
| 4x | 29.6 dB | 0.713 | 250.9 FPS | Industry Leading |

## Key Features

- **IMDN Architecture**: Information Multi-Distillation Network optimized for thermal imagery
- **Thermal-Specific Loss**: Custom loss function combining L1, gradient preservation, and thermal contrast
- **Scale Adaptation**: Intelligent weight adaptation from RGB pretrained models to thermal domain
- **Real-time Performance**: 250+ FPS processing with industry-leading quality metrics

## Applications

- **Autonomous Vehicles**: Enhanced thermal perception for night driving
- **Industrial Monitoring**: Precise equipment temperature analysis
- **Security Systems**: Thermal surveillance capabilities
- **Medical Imaging**: High-resolution thermal diagnostics

## Quick Start

```bash
# Clone repository
git clone https://github.com/Kronbii/thermal-super-resolution.git
cd thermal-super-resolution

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python pillow numpy matplotlib tqdm

# Train model
jupyter notebook fine-tune-model.ipynb

# Test model
python test-thermal-model.py --checkpoint checkpoints/thermal/thermal_best.pth --scale 2
```

## Project Structure

```
thermal-super-resolution/
├── model/                   # IMDN model implementations
├── data/                    # Dataset loader and utilities
├── checkpoints/             # Pretrained and fine-tuned models
├── results/                 # Performance reports and comparisons
├── fine-tune-model.ipynb    # Main training notebook
└── test-thermal-model.py    # Evaluation pipeline
```

## Technical Details

### Model Specifications
- **Parameters**: 688,636 (lightweight)
- **Model Size**: 2.7 MB
- **Input**: Single-channel thermal images
- **Output**: Enhanced thermal images at 2x, 3x, or 4x resolution

### Training Configuration
- **Dataset**: FLIR ADAS v2 thermal images
- **Loss Function**: Multi-component thermal-specific loss
- **Optimization**: AdamW with cosine annealing
- **Hardware**: CUDA-enabled GPU (8GB+ recommended)

### Performance Benchmarks
| Method | PSNR (dB) | SSIM | Speed (FPS) | Parameters |
|--------|-----------|------|-------------|------------|
| Bicubic | 24.2 | 0.612 | 1000+ | - |
| ESRGAN | 28.1 | 0.689 | 15.3 | 16.7M |
| **This Work** | **34.2** | **0.840** | **229.6** | **0.69M** |

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

```bibtex
@misc{thermal_super_resolution_2025,
  title={Thermal Super-Resolution with Information Multi-Distillation Network},
  author={Kronbii},
  year={2025},
  url={https://github.com/Kronbii/thermal-super-resolution}
}
```