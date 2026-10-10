# Thermal super-resolution: portfolio summary

All numbers on this page are generated from `results/benchmark.json` by
`python -m benchmark.report` and match the README. To change a number, rerun
the benchmark; editing it here by hand would be overwritten by the next
report.

## Summary

This project fine-tunes IMDN, a lightweight super-resolution network built for
RGB photos, on single-channel thermal frames from the FLIR ADAS v2 driving
dataset, at x2, x3 and x4. Its evaluation was then rebuilt so the numbers can
be checked. A single protocol file defines the test images, how the
low-resolution inputs are made, the metrics (verified against scikit-image)
and a GPU-synchronised timing method. CI keeps the README equal to the
benchmark output. Measured this way, the model gives a modest PSNR
improvement over bicubic interpolation at every scale, at real-time speed on a
laptop GPU. The
earlier, larger claims did not survive re-measurement and were withdrawn.

<!-- BENCHMARK:HEADLINE:START -->
Measured with `python -m benchmark.run` (protocol v1.0): at x2 on the 17 frames in `test-data/`, the fine-tuned model reaches **32.56 dB PSNR / 0.8469 SSIM**, against 31.81 dB / 0.8291 for bicubic interpolation. One 320x256 -> 640x512 frame takes **14.22 ms median** for the forward pass with fp16 autocast on an NVIDIA GeForce RTX 3070 Laptop GPU (70 FPS; 14.60 ms, 69 FPS including host-device copies and conversion); in fp32 it takes 21.90 ms (46 FPS).
<!-- BENCHMARK:HEADLINE:END -->

## Headline measurements

<!-- BENCHMARK:MEASUREMENTS:START -->
Suggested portfolio tiles (label: value, context):

- **x2 PSNR**: 32.56 dB (bicubic 31.81 dB) on 17 FLIR ADAS v2 frames, protocol v1.0
- **x2 latency**: 14.2 ms median (70 FPS, batch 1) for 320x256 -> 640x512, fp16 autocast, RTX 3070 Laptop GPU
- **x4 latency**: 3.8 ms median (265 FPS, batch 1) for 160x128 -> 640x512, fp16 autocast, RTX 3070 Laptop GPU
- **Other cameras, x2**: +0.91 dB PSNR over bicubic on 51 TNO thermal images

All:

| Label | Value | Context |
|:---|:---|:---|
| PSNR / SSIM, x2 | 32.56 dB / 0.8469 | 17 frames in test-data/, bicubic 31.81 dB / 0.8291, border 2 px, uint8 |
| PSNR / SSIM, x3 | 28.94 dB / 0.7342 | 17 frames in test-data/, bicubic 27.99 dB / 0.6937, border 3 px, uint8 |
| PSNR / SSIM, x4 | 27.87 dB / 0.6816 | 17 frames in test-data/, bicubic 27.09 dB / 0.6522, border 4 px, uint8 |
| Latency, x2 fp32 | 21.90 ms median (p95 22.26), 46 FPS | 320x256 -> 640x512, batch 1, forward only; end-to-end 22.29 ms |
| Latency, x2 fp16 | 14.22 ms median (p95 14.33), 70 FPS | 320x256 -> 640x512, batch 1, forward only; end-to-end 14.60 ms |
| Latency, x3 fp32 | 10.59 ms median (p95 10.74), 94 FPS | 213x170 -> 639x510, batch 1, forward only; end-to-end 10.77 ms |
| Latency, x3 fp16 | 6.96 ms median (p95 7.03), 144 FPS | 213x170 -> 639x510, batch 1, forward only; end-to-end 7.31 ms |
| Latency, x4 fp32 | 5.84 ms median (p95 6.16), 171 FPS | 160x128 -> 640x512, batch 1, forward only; end-to-end 6.00 ms |
| Latency, x4 fp16 | 3.78 ms median (p95 4.07), 265 FPS | 160x128 -> 640x512, batch 1, forward only; end-to-end 4.46 ms |
| Cross-sensor PSNR, x2 | 42.00 dB (bicubic 41.10 dB) | 51 TNO thermal images, other cameras |
| Cross-sensor PSNR, x3 | 36.52 dB (bicubic 36.12 dB) | 51 TNO thermal images, other cameras |
| Cross-sensor PSNR, x4 | 34.49 dB (bicubic 33.79 dB) | 51 TNO thermal images, other cameras |

Hardware: NVIDIA GeForce RTX 3070 Laptop GPU (8.19 GB), driver 580.173.02, CUDA 12.8, cuDNN 9.10.2, PyTorch 2.9.1+cu128, Python 3.12.3.
<!-- BENCHMARK:MEASUREMENTS:END -->

## CV lines

<!-- BENCHMARK:CV:START -->
- Adapted an RGB-pretrained IMDN super-resolution network (0.69M parameters) to single-channel thermal images (x2/x3/x4) by fine-tuning on 10,697 FLIR ADAS v2 frames; PSNR gain over bicubic of +0.76/+0.95/+0.78 dB on 17 validation-split frames and +0.91/+0.40/+0.70 dB on 51 images from other thermal cameras (TNO).
- Measured 14.2 ms median latency (70 FPS) for 320x256 -> 640x512 frames with fp16 autocast on an RTX 3070 Laptop GPU (PyTorch eager, batch 1, GPU-synchronised timing; 14.6 ms including host-device transfers).
- Built a reproducible benchmark (single protocol file, metrics verified against scikit-image, CPU CI) and replaced unverifiable README claims with its output, including speed figures that had been timed without GPU synchronisation.
<!-- BENCHMARK:CV:END -->

## Portfolio fields

- **What.** Super-resolution for single-channel thermal (LWIR) images at x2,
  x3 and x4, with a reproducible benchmark.
- **Problem.** Thermal cameras have low native resolution. Learned
  super-resolution models are trained on RGB photos and expect a different
  kind of image, and the evaluations that come with them are rarely
  reproducible.
- **How.** IMDN was fine-tuned from its RGB weights: the first and last layers
  were converted to one channel by averaging, the backbone was unfrozen in
  stages, and the loss was L1 plus gradient and local-contrast terms. The
  training data were OpenCV-bicubic LR/HR pairs from FLIR ADAS v2. The
  benchmark adds one protocol, bicubic and upstream-weight baselines,
  GPU-synchronised latency, and CI.
- **Role.** Sole author: data preparation, training, evaluation, benchmark and
  real-time demo.

## Stages

1. **Data.** FLIR ADAS v2 thermal images became paired LR/HR images with OpenCV
   bicubic downscaling, split 10,697 train / 1,189 validation.
2. **Transfer.** The RGB IMDN was converted to one input and one output channel
   by averaging the pretrained weights.
3. **Fine-tuning.** One model per scale (x2/x3/x4) was trained on an 8 GB
   laptop GPU with mixed precision and a staged backbone unfreeze.
4. **Benchmark.** A single protocol, metrics verified against scikit-image,
   bicubic and upstream-weight baselines, synchronised fp32/fp16 timing, and a
   CI check that keeps the docs equal to the results.
5. **Real-time demo.** A threaded acquire, pre-process, infer and display
   pipeline with fp16 autocast (`threaded_display_60fps.py`).

## Limits

- The built-in test set is small (17 frames from 3 videos) and its reference
  frames are 2x downscales, not native sensor frames. The native held-out
  FLIR split (`flir-video-test`) is supported but has not been run yet.
- The 17 frames come from the validation split that also selected the
  checkpoint.
- The model learned OpenCV's non-anti-aliased bicubic degradation. With an
  anti-aliased degradation, the original RGB weights do better (see the
  degradation sensitivity table in the README).
- On images from other thermal cameras (TNO), PSNR improves at every scale,
  but SSIM is slightly below bicubic at x2 and x3.
- Timings come from a laptop GPU in PyTorch eager mode and vary with
  temperature. No TensorRT or CUDA-graph optimisation was used.
- The fine-tuned weights are not yet published, so third parties can rerun the
  pipeline but not the model numbers.

## Hero image

Use **`docs/figures/benchmark.png`**: PSNR gain over bicubic per scale on both
test sets, and forward latency per scale in fp32 and fp16. It is regenerated
from the same JSON with `python -m benchmark.figure`. A metrics chart was
chosen over image crops because the test frames are street scenes that may
contain people. No thermal imagery was inspected or cropped for the figure.
