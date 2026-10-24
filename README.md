# Thermal Super-Resolution with IMDN

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.8%2B-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An RGB-pretrained IMDN (Information Multi-Distillation Network) adapted to single-channel thermal imagery using a thermal-specific loss, FLIR ADAS v2, and custom-curated data. It reaches **34.2 dB PSNR / 0.840 SSIM** at ×2, **31.0 / 0.757** at ×3, and **29.6 / 0.713** at ×4, running at **~130 FPS at ×2 and ~60 FPS at ×4 on an RTX 3070 laptop GPU**, and at **20–30 FPS on an NVIDIA Jetson Orin** onboard a drone with FP16/INT8 inference.

## TL;DR: The Results

| Result | Scale |
|:---:|:---:|
| ![Before](results/showcase/_x2_showcase.png) | 2x |
| ![Before](results/showcase/_x3_showcase.png) | 3x |

## Results

| Scale | PSNR | SSIM |
|:---:|:---:|:---:|
| **2x** | **34.2 dB** | **0.840** |
| **3x** | **31.0 dB** | **0.757** |
| **4x** | **29.6 dB** | **0.713** |

On an RTX 3070 laptop GPU the model runs at ~130 FPS at ×2 and ~60 FPS at ×4. Optimized with FP16/INT8, it runs at 20–30 FPS on an NVIDIA Jetson Orin onboard a drone.

## Approach

- **IMDN Backbone**: Information Multi-Distillation Network, pretrained on RGB
- **Thermal-Aware Loss Function**: Multi-component loss preserving thermal gradients and contrast characteristics
- **Cross-Domain Transfer**: RGB-pretrained weights adapted to single-channel thermal imagery
- **Edge Inference**: FP16/INT8-optimized inference on NVIDIA Jetson Orin, for drones and ground robots

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
- **Input**: Single-channel thermal images
- **Output**: Enhanced thermal images at 2x, 3x, or 4x resolution

### Training Configuration
- **Dataset**: FLIR ADAS v2 thermal images and custom-curated thermal data
- **Loss Function**: Multi-component thermal-specific loss
- **Optimization**: AdamW with cosine annealing
- **Hardware**: CUDA-enabled GPU (8GB+ recommended)

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