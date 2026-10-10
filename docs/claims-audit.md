# Claims audit

This audit lists every quantitative or comparative claim in the README as it
stood on `main` at commit `5a23feb`, plus the claims in the files that README
drew on. For each claim it records where the claim came from, whether the
claim can be reproduced from this repository, and what was done with it.
Measured numbers in this file come from `results/benchmark.json` and are
inserted by `python -m benchmark.report`. They are never typed by hand.

## What was examined

| Material | In the public repo? | Notes |
|:---|:---|:---|
| `README.md` | yes | The claims audited below. |
| `scripts/prepare-train-data.ipynb`, `scripts/fine-tune-model.ipynb`, `scripts/test-thermal-model.ipynb` | **no**: `scripts/` has been git-ignored since `c93c683` | These notebooks were read from the local working copy. They show how the data, training and evaluation were done. |
| `results/_x2/performance_report.txt`, `results/_x3/performance_report.txt` | **no**: `*performance_report.txt` is git-ignored | These are the source of the x2/x3 PSNR, SSIM and FPS figures. No x4 report exists. |
| `checkpoints/` (fine-tuned `thermal_best.pth` for x2/x3/x4, upstream `IMDN_x*.pth`) | **no**: git-ignored | The checkpoints store their epoch and the validation PSNR measured during training. |
| `model/`, `data/common.py`, `data/image_folder.py`, `RGB/`, `FLOPs/`, `checkpoints/pretrained/*.pth` | yes (weights no) | Byte-identical to the upstream [Zheng222/IMDN](https://github.com/Zheng222/IMDN) (git blob hashes compared). The upstream repository declares no licence. |
| `test-data/` (17 PNGs) | yes | See [test-data/README.md](../test-data/README.md) for provenance and licence. |
| `threaded_display_*.py`, `convert-LR-HR-video.py` | yes | Demo scripts. They print FPS at run time, but the README quotes none of it. |

## How the original numbers were produced

Reconstructed from the notebooks and checkpoint files:

1. **Data.** FLIR ADAS v2 `images_thermal_train` (10,742 images) and
   `images_thermal_val` (1,144 images) were pooled, shuffled (seed 42) and split
   90/10 into 10,697 training and 1,189 validation images. Each HR image was
   cropped to a multiple of the scale, and the LR image was made with
   `cv2.resize(..., INTER_CUBIC)`.
2. **Fine-tuning.** Training started from the upstream DIV2K `IMDN_x{s}.pth`.
   The input convolution and the upsampler were averaged over the RGB channels.
   The optimiser was AdamW (lr 2.2e-5) with a cosine schedule for 60 epochs.
   The loss was L1 + 0.08 x Sobel-gradient L1 + 0.03 x local-variance L1.
   Training used AMP, 144 px patches and batch 10 with 5-step gradient
   accumulation, and the backbone was frozen for the first 10 epochs. These
   are the notebook's saved settings, last used for x3. The x4 checkpoint files
   show 100 epochs with validation every 5 epochs; the other x4 settings were
   not recorded.
3. **Checkpoint selection.** `thermal_best.pth` is the checkpoint with the best
   validation PSNR, checked every 3rd epoch (x4: every 5th) on the first 50
   validation batches. The selected epochs are 57 (x2), 54 (x3) and 90 (x4).
4. **Evaluation** (`test-thermal-model.ipynb`). The evaluation used the first
   1,100 of the same 1,189 validation images, which had also selected the
   checkpoint, so they do not form an independent test set. PSNR was computed
   on the unquantised float output in [0, 1] over the whole image, with no
   border crop. SSIM used zero padding at the image borders. The reported
   value is the mean over images. Neither the split nor the evaluation images
   are published.
5. **Speed.** The notebook wrapped `model(lr)` in `time.time()` calls with no
   `torch.cuda.synchronize()`. CUDA kernels run asynchronously, so this measures
   Python and kernel-launch time, not GPU work. FPS was computed as images
   divided by summed time.

## Claim by claim

| # | Claim (old README) | Where it came from | Reproducible from this repo? | Action |
|:---:|:---|:---|:---|:---|
| 1 | "**First** thermal super-resolution system to achieve **34.2 dB PSNR at 229+ FPS**" | 34.2 dB and 229.6 FPS: x2 `performance_report.txt`. "First": no source. | No. The evaluation split is not published, the weights are not in git, and the timing did not synchronise the GPU. | **Removed.** Replaced by the benchmark headline. |
| 2 | "using **novel IMDN architecture** with specialized thermal adaptations" | The architecture is IMDN (Hui et al., ACM MM 2019), and `model/` is byte-identical to upstream. The "adaptations" are 1-channel input/output plus weight averaging. | n/a | **Removed** "novel". The README now says "IMDN, fine-tuned". |
| 3 | "**Outperforms existing methods** while maintaining real-time inference" | No comparison with any published method exists in the repo. | No | **Removed.** Replaced by measured baselines (bicubic, upstream IMDN weights). |
| 4 | Table, x2: 34.2 dB / 0.840 / **270.6 FPS** / "**New SOTA for thermal SR**" | PSNR and SSIM match the x2 report. **270.6 FPS appears nowhere**: the report says 229.6. "SOTA": no comparison with published thermal SR results (e.g. the PBVS thermal SR challenges). | No (see 1) | **Removed.** Replaced by the generated table. |
| 5 | Table, x3: 31.0 / 0.757 / 256.1 FPS / "**15x faster than competitors**" | The figures match the x3 report. "15x" = 229.6 / 15.3, and the 15.3 FPS ESRGAN figure has no source (see 14). | No | **Removed.** |
| 6 | Table, x4: 29.6 / 0.713 / 250.9 FPS / "**First real-time 4x thermal SR**" | No x4 report or log exists in the repo. "First": no source. | No | **Removed.** |
| 7 | "Novel IMDN Adaptation: **First application** of IMDN to thermal domain" | No source, and no literature search supports it. | n/a | **Removed.** |
| 8 | "Thermal-Aware Loss Function: multi-component loss **preserving thermal gradients and contrast**" | The loss exists (notebook cell 3), but its effect was never measured (no ablation). | Loss: yes (notebook, not published). Effect: no. | **Kept as a description** of the loss, without the effect claim. |
| 9 | "Cross-Domain Transfer: **Breakthrough method** for adapting RGB pretrained models" | The method averages the first-layer and upsampler weights over RGB, a standard technique. | Yes (notebook) | **Kept as a description**, with "breakthrough" removed. |
| 10 | "Efficiency Optimization: **40x parameter reduction** vs. competing methods with **superior quality**" | No source. The 0.69M parameters are correct. 16.7M (ESRGAN's RRDB generator) / 0.69M would be 24x, not 40x. Nothing was compared. | No | **Removed.** Only the parameter count is kept. |
| 11 | "Parameters: 688,636" | Correct for x2 (x3: 691,521; x4: 695,560). | Yes (`tests/test_data_and_models.py`) | **Kept**, with per-scale values. |
| 12 | "Model Size: 2.7 MB" | The fp32 weights are 2.75 MB. The `thermal_best.pth` files are 8.4 MB because they also hold optimiser state. | Yes | **Qualified** ("2.75 MB of fp32 weights"). |
| 13 | "Dataset FLIR ADAS v2, multi-component loss, AdamW + cosine annealing, 8 GB+ GPU recommended" | Consistent with the notebook. Training ran on an 8 GB RTX 3070 Laptop GPU. | Notebook only (not published) | **Kept.** "Recommended" becomes "trained on". |
| 14 | Comparative table: Bicubic **24.2 dB / 0.612 / 1000+ FPS**; ESRGAN **28.1 / 0.689 / 15.3 FPS / 16.7M**; this work "**+6.1 dB, 15x faster**" | No code, data, log or citation for the bicubic or ESRGAN rows. ESRGAN was never run here, though 16.7M is the published ESRGAN generator size. Under the benchmark protocol, bicubic scores far above 24.2 dB on the committed frames (see below). | No | **Removed.** |
| 15 | "Significance: **the largest PSNR improvement** in thermal super-resolution ... **24x fewer parameters** than existing methods" | No source | No | **Removed.** |
| 16 | Applications: "Autonomous Vehicles", "Industrial Monitoring", "Security", "Medical Imaging: high-resolution thermal diagnostics" | Aspirational. None of these was evaluated, and the model was trained on driving scenes only. | n/a | **Reduced** to one sentence marked "not evaluated". |
| 17 | Badges: Python 3.10+, PyTorch 2.0+, CUDA 11.8+ | Never tested as ranges. | n/a | **Replaced** by the tested versions (Python 3.12, PyTorch 2.9.1, CUDA 12.8). |
| 18 | Quick Start: `jupyter notebook fine-tune-model.ipynb`, `python test-thermal-model.py --checkpoint checkpoints/thermal/thermal_best.pth` | The notebook lives in the git-ignored `scripts/`. `test-thermal-model.py` and `checkpoints/thermal/` do not exist. | No (broken commands) | **Replaced** by the benchmark commands. |
| 19 | "TL;DR: The Results" showcase images | Qualitative LR / SR / GT figures from the test notebook on the validation split. | Images only | **Kept**, labelled qualitative. |
| 20 | `performance_report.txt`: "EXCELLENT - Exceeds industry standards (>28 dB)", "APPROVED FOR PRODUCTION" | Thresholds hard-coded in the notebook, not an external standard. | n/a (files not public) | Not quoted anywhere now. |

## Old figures next to the measured ones

Different images and a different protocol, so the two columns are context,
not a like-for-like comparison. The measured column is the one this
repository now stands behind.

<!-- BENCHMARK:CLAIMS:START -->
| Figure | Old README | Measured now | Why they differ |
|:---|:---|:---|:---|
| x2 PSNR / SSIM | 34.2 dB / 0.840 | 32.56 dB / 0.8469 (bicubic 31.81 dB / 0.8291) | Old: 1,100 unpublished validation frames at 640x512 that also picked the checkpoint, no border crop, unquantised output. New: 17 committed frames, protocol v1.0. Not the same images, so not directly comparable. |
| x3 PSNR / SSIM | 31.0 dB / 0.757 | 28.94 dB / 0.7342 (bicubic 27.99 dB / 0.6937) | As above. |
| x4 PSNR / SSIM | 29.6 dB / 0.713 | 27.87 dB / 0.6816 (bicubic 27.09 dB / 0.6522) | No x4 evaluation output exists in the repository, so how this figure was produced is unknown. |
| x2 bicubic baseline | 24.2 dB / 0.612 | 31.81 dB / 0.8291 | No code, data or citation for the old row. |
| x2 gain over bicubic | +10.0 dB (34.2 vs 24.2) | +0.76 dB | Same images and protocol for both methods now. |
| x2 speed | 270.6 (table) / 229.6 (report) FPS | fp32 46 FPS, fp16 70 FPS (320x256 -> 640x512, forward only) | Old timing used time.time() without torch.cuda.synchronize(), which measures kernel launches, not GPU work. Repeating the notebook's loop on this laptop (`python -m benchmark.legacy_timing`) reports x2 186 FPS, x3 208 FPS, x4 211 FPS: nearly flat across scales, although the synchronised x2 forward pass costs 3.7x the x4 one. |
| x3 speed | 256.1 FPS | fp32 94 FPS, fp16 144 FPS (213x170 -> 639x510, forward only) | As above. |
| x4 speed | 250.9 FPS | fp32 171 FPS, fp16 265 FPS (160x128 -> 640x512, forward only) | As above. |
<!-- BENCHMARK:CLAIMS:END -->

## Reconciling with the 2025-09-23 evaluation report

The repository's own evaluation report, `results/_x2/performance_report.txt`
(timestamp 2025-09-23 21:31:13), is the source that later summaries quote:
**229.6 FPS, 34.2 dB PSNR, 0.840 SSIM on 1,100 FLIR images**. Its x3 sibling,
`results/_x3/performance_report.txt` (21:33:03), reports 256.1 FPS,
31.0 dB and 0.757. Both files exist in the local working copy, but they are
git-ignored (`*performance_report.txt`), so they were never public. They were
written by `scripts/test-thermal-model.ipynb`, whose saved output matches the
x3 report.

- **The frame list is not in the repository.** The report names the folder
  `./datasets/flir_thermal_x2/val/LR_bicubic/X2`, which is git-ignored and not
  present in the working copy. The notebook took the first 1,100 of the 1,189
  validation files in filesystem order (unsorted `Path.glob`), so the exact
  list cannot be rebuilt from the repository. The 1,100 local outputs in
  `results/_x2/enhanced/` are numbered (`thermal_enhanced_001.png`, ...), not
  named after their source frames, so they cannot be mapped back either.
  These are the same validation images that selected the checkpoint. The
  17 frames in `test-data/` come from the same split, and whether they are
  among the 1,100 is unknown.
- **The GPU is not named.** The report and the notebook output say only
  `cuda`. Nothing in the repository supports "desktop GPU". The report's
  checkpoint path, `/home/kronbii/repos/thermal-super-resolution/...`, is this
  machine's path, and the training notebook's log records an NVIDIA GeForce
  RTX 3070 Laptop GPU. The most likely reading is that the report was produced
  on the same laptop GPU used for the benchmark here, but this cannot be
  proven.
- **The quality figures cannot be recomputed** without the unpublished frames.
  They were computed on unquantised float output, over the whole image (no
  border crop), with zero-padded SSIM. That is a different protocol from
  v1.0 on different images, so the 34.2 dB and the benchmark's numbers are not
  comparable.
- **The speed figure is an artefact of the timing method.** Repeating the
  notebook's exact loop on this laptop gives the same kind of number: high
  and almost independent of scale (see the speed rows in the table above). The
  forward pass, timed with synchronisation, gives the figures in the README.
  The report's FPS is therefore withdrawn, not corrected.
- The report's labels "EXCELLENT - Exceeds industry standards (>28 dB)" and
  "APPROVED FOR PRODUCTION" are thresholds hard-coded in the notebook, not an
  assessment against any external standard.

## Other findings

- **fp16 overflow.** With pure half precision (`model.half()`), the IMDN
  contrast-aware channel attention sums activations over the whole feature
  map, and the sum overflows float16 at 320x256 input. Every output pixel then
  becomes NaN (checked: 100% NaN at x2 on a `test-data/` frame).
  `threaded_display_30fps.py` runs this way, so its output is not a valid image,
  and the FPS it prints is not meaningful. `torch.autocast(float16)`, which
  `threaded_display_60fps*.py` and the benchmark use, keeps reductions in
  float32 and matches fp32 quality.
- **Degradation dependence.** The fine-tuned model learned OpenCV's
  non-anti-aliased bicubic downscaling. When the LR input is made with an
  anti-aliased (PIL/MATLAB-style) bicubic instead, the upstream RGB weights
  beat it (see the "Degradation sensitivity" table in `results/benchmark.md`).
  Part of its advantage is specialisation to the training degradation.
- **SSIM vs bicubic on TNO.** On the cross-sensor TNO images the model gains
  PSNR at every scale, but its SSIM is slightly below bicubic at x2 and x3.
- `threaded_display_60fps.py` sets `ACQ_PERIOD = 1.0 / 5`, which is 5 FPS
  acquisition, while its comment says 60 FPS.
- **Licences.** The MIT licence covers this repository's own code. The IMDN
  code and weights come from an upstream repository with no declared licence,
  and the frames in `test-data/` fall under FLIR's non-commercial dataset
  licence. See [test-data/README.md](../test-data/README.md).

## How to reproduce what replaced these claims

See "Reproduce" in the [README](../README.md#reproduce). The protocol is
defined in `benchmark/protocol.py`, and the full output is in
`results/benchmark.json` and `results/benchmark.md`.
