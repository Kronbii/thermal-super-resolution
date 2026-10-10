# Benchmark results

Protocol v1.0 (`benchmark/protocol.py`), `python -m benchmark.run --scale 2 3 4 --sets repo17 tno-lwir`, commit `a4ea485010`, 2026-10-10T12:36:31Z.

## Quality

**test-data/ (17 FLIR ADAS v2 frames, 320x256)**, n = 17. The 'HR' reference here is itself a 2x bicubic downscale of the native 640x512 frame, so this set measures the operator on 320x256 content, not on native-resolution frames.

| Scale | Method | PSNR (dB) | SSIM | PSNR gain vs bicubic (dB) |
|:---:|:---|---:|---:|---:|
| x2 | Bicubic (OpenCV) | 31.81 | 0.8291 | - |
| x2 | IMDN, upstream RGB weights | 29.17 | 0.7868 | -2.63 |
| x2 | **IMDN, thermal fine-tune (this repo)** | 32.56 | 0.8469 | +0.76 |
| x3 | Bicubic (OpenCV) | 27.99 | 0.6937 | - |
| x3 | IMDN, upstream RGB weights | 25.30 | 0.6280 | -2.68 |
| x3 | **IMDN, thermal fine-tune (this repo)** | 28.94 | 0.7342 | +0.95 |
| x4 | Bicubic (OpenCV) | 27.09 | 0.6522 | - |
| x4 | IMDN, upstream RGB weights | 24.26 | 0.5979 | -2.84 |
| x4 | **IMDN, thermal fine-tune (this repo)** | 27.87 | 0.6816 | +0.78 |

Thermal model with fp16 autocast: PSNR less than 0.001 dB from fp32 at every scale.

**TNO Image Fusion Dataset, thermal band (51 images, other sensors)**, n = 51. Native-resolution thermal images from cameras other than the one the model was trained on: a generalisation check, not an in-distribution test.

| Scale | Method | PSNR (dB) | SSIM | PSNR gain vs bicubic (dB) |
|:---:|:---|---:|---:|---:|
| x2 | Bicubic (OpenCV) | 41.10 | 0.9676 | - |
| x2 | IMDN, upstream RGB weights | 38.20 | 0.9432 | -2.89 |
| x2 | **IMDN, thermal fine-tune (this repo)** | 42.00 | 0.9649 | +0.91 |
| x3 | Bicubic (OpenCV) | 36.12 | 0.9151 | - |
| x3 | IMDN, upstream RGB weights | 33.26 | 0.8799 | -2.86 |
| x3 | **IMDN, thermal fine-tune (this repo)** | 36.52 | 0.9134 | +0.40 |
| x4 | Bicubic (OpenCV) | 33.79 | 0.8783 | - |
| x4 | IMDN, upstream RGB weights | 30.17 | 0.8272 | -3.63 |
| x4 | **IMDN, thermal fine-tune (this repo)** | 34.49 | 0.8863 | +0.70 |

Thermal model with fp16 autocast: PSNR within 0.002 dB from fp32 at every scale.

## Degradation sensitivity (test-data/)

Same frames, same methods, but the LR input is made with an anti-aliased bicubic downscale (PIL, MATLAB-like) instead of the OpenCV bicubic used for training. This is a sensitivity check, not the headline protocol.

| Scale | Method | PSNR (dB), OpenCV LR | PSNR (dB), anti-aliased LR |
|:---:|:---|---:|---:|
| x2 | Bicubic (OpenCV) | 31.81 | 31.80 |
| x2 | IMDN, upstream RGB weights | 29.17 | 33.12 |
| x2 | **IMDN, thermal fine-tune (this repo)** | 32.56 | 32.26 |
| x3 | Bicubic (OpenCV) | 27.99 | 29.66 |
| x3 | IMDN, upstream RGB weights | 25.30 | 30.60 |
| x3 | **IMDN, thermal fine-tune (this repo)** | 28.94 | 29.86 |
| x4 | Bicubic (OpenCV) | 27.09 | 28.52 |
| x4 | IMDN, upstream RGB weights | 24.26 | 29.29 |
| x4 | **IMDN, thermal fine-tune (this repo)** | 27.87 | 28.57 |

## Speed (thermal model)

| Scale | Input -> output (W x H) | Precision | Model latency median / p95 (ms) | Trial medians min-max (ms) | Model FPS | End-to-end median / p95 (ms) | End-to-end FPS |
|:---:|:---:|:---:|---:|---:|---:|---:|---:|
| x2 | 320x256 -> 640x512 | fp32 | 21.90 / 22.26 | 21.50-22.08 | 46 | 22.29 / 22.50 | 45 |
| x2 | 320x256 -> 640x512 | fp16 | 14.22 / 14.33 | 13.87-14.26 | 70 | 14.60 / 14.75 | 69 |
| x3 | 213x170 -> 639x510 | fp32 | 10.59 / 10.74 | 10.32-10.70 | 94 | 10.77 / 10.88 | 93 |
| x3 | 213x170 -> 639x510 | fp16 | 6.96 / 7.03 | 6.74-7.01 | 144 | 7.31 / 7.47 | 137 |
| x4 | 160x128 -> 640x512 | fp32 | 5.84 / 6.16 | 5.67-6.15 | 171 | 6.00 / 6.31 | 167 |
| x4 | 160x128 -> 640x512 | fp16 | 3.78 / 4.07 | 3.66-4.05 | 265 | 4.46 / 4.79 | 224 |

Batch 1. 5 interleaved trials x (50 warm-up + 300 timed iterations), samples pooled; `torch.cuda.synchronize()` around every timed call, FPS = 1000 / median. *Model latency*: forward pass only, input already on the GPU. *End-to-end*: uint8 frame in host memory to uint8 frame in host memory (conversion, copies, forward, rounding). fp32 has TF32 disabled; fp16 is `torch.autocast(float16)` with float32 weights. Plain PyTorch eager mode (no TensorRT, CUDA graphs or torch.compile).

Hardware: NVIDIA GeForce RTX 3070 Laptop GPU (8.19 GB), driver 580.173.02, CUDA 12.8, cuDNN 9.10.2, PyTorch 2.9.1+cu128, Python 3.12.3.

Conditions: GPU at 70-87 °C and SM clock 1440-1710 MHz (max 2100 MHz) at the end of the trials, 1-min CPU load average 0.8-3.2. Trials started at 38-60 °C with CPU load average 0.3-3.9. Treat the trial spread as the uncertainty of these timings.

## Weights evaluated

| Method | Scale | Params | state-dict SHA-256 |
|:---|:---:|---:|:---|
| imdn_rgb_pretrained | x2 | 694,404 | `b112699ca81f4cbaeb1ca17aea66919c6fc70d272e6a26ce80d9da2baf83a5ae` |
| imdn_rgb_pretrained | x3 | 703,059 | `1a0fc9ab49730c10bcc6e63b5e23df4525ebbe6b97db755c329f9514cd0f1c54` |
| imdn_rgb_pretrained | x4 | 715,176 | `6ee0b6f7ca20096269ae0b3b4ec385926d0f002d5bdb1fa94ad4f7a882879d5c` |
| imdn_thermal | x2 | 688,636 | `3e6e1adf722b1565c994df558d0b4ad61dd358a9995f916630a94aab8e73d9a6` |
| imdn_thermal | x3 | 691,521 | `d9ce9a52316bc3dafaa67247673881b140637df413ee64f3ba405bf29709fec3` |
| imdn_thermal | x4 | 695,560 | `69d6e133620bf7ab25b6186d4eca2dd7ba42ead1fbc2cfbefebd8f22f7850f7a` |
