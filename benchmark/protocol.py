"""The evaluation protocol, defined in one place.

Every number the benchmark reports is produced under the settings below. Change
them here (and bump ``PROTOCOL_VERSION``) or not at all; the rest of the package
only reads these values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PROTOCOL_VERSION = "1.0"

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_CACHE = REPO_ROOT / ".bench-data"  # downloaded data lives here; git-ignored

SCALES = (2, 3, 4)
SEED = 0

# --------------------------------------------------------------------------
# Degradation: how the low-resolution (LR) input is made from the reference
# (HR) image. This mirrors scripts/prepare-train-data.ipynb, which produced the
# training pairs: grayscale uint8, crop to a multiple of the scale from the
# top-left, then cv2.resize(..., INTER_CUBIC) to (W/s, H/s). No blur, no noise,
# no anti-aliasing (OpenCV's bicubic does not pre-filter when downscaling).
# --------------------------------------------------------------------------
DEGRADATION = {
    "read": "cv2.imread(path, cv2.IMREAD_GRAYSCALE) -> uint8, 1 channel",
    "modcrop": "HR[:H - H % s, :W - W % s] (top-left anchored)",
    "downscale": "cv2.resize(HR, (W // s, H // s), interpolation=cv2.INTER_CUBIC)",
    "matches_training": "scripts/prepare-train-data.ipynb, generate_lr_image()",
}

PRIMARY_DEGRADATION = "cv2_bicubic"
# Sensitivity check only (reported separately, never as the headline): PIL's
# bicubic resize widens the kernel by the scale factor when downscaling, i.e.
# it anti-aliases, like MATLAB imresize, which the upstream RGB IMDN was
# trained with. It shows how much the comparison depends on the degradation.
ABLATION_DEGRADATIONS = {
    "pil_bicubic_antialias": "PIL.Image.resize((W // s, H // s), Image.BICUBIC) on the "
    "same modcropped HR",
}
ABLATION_SETS = ("repo17",)

# Model input / output handling.
IO = {
    "input": "float32 tensor = uint8 / 255, shape (1, 1, h, w), range [0, 1]",
    "rgb_baseline_input": "gray replicated to 3 channels (upstream IMDN is RGB)",
    "rgb_baseline_output": "mean of the 3 output channels",
    "output": "clamp to [0, 1], x255, round half to even (torch.round), uint8",
}

# --------------------------------------------------------------------------
# Metrics. Computed on the uint8 output vs the uint8 HR, single channel,
# after removing BORDER = scale pixels on every side (the IMDN/EDSR
# convention). Reported value = mean over images of the per-image metric.
# --------------------------------------------------------------------------
DATA_RANGE = 255.0


def border_for(scale: int) -> int:
    return scale


PSNR_DEF = "10*log10(255^2 / MSE), MSE over the border-cropped image, float64"
SSIM_DEF = {
    "reference": "Wang et al. 2004, as skimage.metrics.structural_similarity("
    "gaussian_weights=True, sigma=1.5, use_sample_covariance=False, "
    "data_range=255)",
    "window": "11x11 Gaussian, sigma=1.5 (radius 5), normalised to sum 1",
    "K1": 0.01,
    "K2": 0.03,
    "covariance": "population (no N/(N-1) correction)",
    "aggregation": "mean of the SSIM map over the 'valid' region (window fully inside the image)",
}
SSIM_SIGMA = 1.5
SSIM_RADIUS = 5
SSIM_K1 = 0.01
SSIM_K2 = 0.03

# --------------------------------------------------------------------------
# Speed. Batch 1, synthetic input of the stated size (content does not change
# the cost of a convolutional network). Output is ~640x512, the native
# resolution of the FLIR Tau 2 camera the training data was recorded with.
# --------------------------------------------------------------------------
SPEED_OUTPUT_HW = (512, 640)


def speed_input_hw(scale: int) -> tuple[int, int]:
    h, w = SPEED_OUTPUT_HW
    return h // scale, w // scale


SPEED = {
    "batch": 1,
    "warmup_iters": 50,
    "timed_iters": 300,
    "trials": 5,
    "interleaving": "for trial: for scale: for precision (thermal drift hits all "
    "configurations alike); the 5 x 300 samples of a configuration are pooled",
    "cooldown": "before every trial, wait until the GPU is at or below "
    "cooldown_temp_c and the 1-minute CPU load average is at or below "
    "max_cpu_loadavg (at most cooldown_timeout_s), so trials start from a "
    "comparable state on a shared laptop; what was reached is recorded",
    "cooldown_temp_c": 60,
    "max_cpu_loadavg": 4.0,
    "cooldown_timeout_s": 300,
    "memory_format": "default contiguous NCHW; no TensorRT, CUDA graphs or torch.compile",
    "precisions": ("fp32", "fp16"),
    "fp32": "float32 weights and activations, TF32 disabled",
    "fp16": "automatic mixed precision: torch.autocast(float16), weights kept in "
    "float32 (as in threaded_display_60fps.py). Pure model.half() is not used: "
    "the channel-attention statistics overflow float16 on 160x128 and larger "
    "inputs and the output becomes NaN",
    "cudnn_benchmark": True,
    "model_latency": "input already on device; torch.cuda.synchronize() before "
    "and after each forward; wall clock via time.perf_counter",
    "end_to_end_latency": "uint8 LR frame in host memory -> float conversion -> "
    "host-to-device copy -> forward -> clamp/round/uint8 -> device-to-host copy "
    "-> uint8 numpy frame (the copy back synchronises)",
    "fps": "1000 / median latency in ms",
}

QUICK = {
    "max_images": 3,
    "warmup_iters": 1,
    "timed_iters": 3,
    "trials": 1,
    "speed_output_hw": (64, 96),
}

# --------------------------------------------------------------------------
# Weights. File names as produced by scripts/fine-tune-model.ipynb. The
# directories are git-ignored; digests identify the exact files evaluated.
# ``state_dict_sha256`` is a container-independent digest of the tensors (see
# models.state_dict_digest), so a slimmed re-export of the same weights
# verifies too.
# --------------------------------------------------------------------------
WEIGHTS_DIR = REPO_ROOT / "checkpoints"
THERMAL_WEIGHTS = {
    2: "_x2/thermal_best.pth",
    3: "_x3/thermal_best.pth",
    4: "x4/thermal_best.pth",
}
# Slim, state-dict-only copies made by benchmark/export_weights.py for
# publishing; looked up in the weights directory if the checkpoint is absent.
EXPORTED_WEIGHTS = "imdn_thermal_x{s}.pth"
UPSTREAM_WEIGHTS = {s: f"pretrained/IMDN_x{s}.pth" for s in SCALES}
UPSTREAM_WEIGHTS_URL = "https://github.com/Zheng222/IMDN/raw/master/checkpoints/IMDN_x{s}.pth"


@dataclass(frozen=True)
class TestSet:
    name: str
    description: str
    source: str
    licence: str
    hr_note: str
    files: tuple[str, ...] = field(default_factory=tuple)  # relative to root
    root: Path = REPO_ROOT
    download: dict | None = None
    exclude_note: str = ""


REPO17 = TestSet(
    name="repo17",
    description="The 17 thermal frames committed in test-data/ (320x256, 8-bit grayscale).",
    source="FLIR ADAS v2 thermal images (Teledyne FLIR). The file names follow "
    "the dataset's naming (video-<id>-frame-<n>-<id>). Their size (320x256 = "
    "640x512 / 2) and the realtime scripts that read them indicate they are the "
    "x2 bicubic LR versions of frames from this repo's own validation split "
    "(scripts/prepare-train-data.ipynb), copied in for the threaded demo.",
    licence="Teledyne FLIR Non-Commercial Image Dataset License Agreement (FLIR "
    "keeps ownership; research / non-commercial use). Not covered by this "
    "repository's MIT licence.",
    hr_note="The 'HR' reference here is itself a 2x bicubic downscale of the "
    "native 640x512 frame, so this set measures the operator on 320x256 "
    "content, not on native-resolution frames.",
)


TNO = TestSet(
    name="tno-lwir",
    description="Optional cross-sensor set: 51 single-band thermal images from "
    "the TNO Image Fusion Dataset (one per scene; image sequences excluded).",
    source="Toet, A. (2014). TNO Image Fusion Dataset. figshare. "
    "https://doi.org/10.6084/m9.figshare.1008029 (Athena, DHV, TRICLOBS "
    "camera systems; 8-bit, from 359x247 to 945x709 pixels).",
    licence="CC BY 4.0 (figshare record). Credit: TNO, Soesterberg, The "
    "Netherlands. Downloaded on demand to .bench-data/, never committed.",
    hr_note="Native-resolution thermal images from cameras other than the one "
    "the model was trained on: a generalisation check, not an in-distribution "
    "test.",
    root=DATA_CACHE / "tno" / "TNO_Image_Fusion_Dataset",
    download={
        "url": "https://ndownloader.figshare.com/files/1475454",
        "md5": "1ab9ce3b84dd3e5894630c13abd20953",
        "sha256": "7aa2ae534a36d71698a0c88f2cce17fc4b595521959fa4bf053469d817f32190",
        "size": 122712465,
        "archive_name": "TNO_Image_Fusion_Dataset.zip",
        "extract_dir": "tno",
    },
    exclude_note="Selection: files whose name marks the thermal band (IR_*, ir, "
    "*_IR, IR<n>, *LWIR*) in single-scene folders; the FEL and DHV image "
    "sequences are excluded (near-duplicate frames). tank/LWIR.tif is excluded "
    "because its three channels differ (not a plain single-band image). "
    "APC_2 is excluded because its file names do not identify the band. The "
    "exact list and per-file SHA-256 are in benchmark/manifests/tno_lwir.json.",
)

FLIR_VIDEO_TEST = TestSet(
    name="flir-video-test",
    description="Optional, user-supplied: the FLIR ADAS v2 video_thermal_test "
    "frames at native 640x512. scripts/prepare-train-data.ipynb drew its train "
    "and validation pairs only from images_thermal_train and images_thermal_val, "
    "so this split was not used for training or checkpoint selection.",
    source="Teledyne FLIR ADAS v2, video_thermal_test/data (obtain from Teledyne "
    "FLIR after accepting its terms; point the benchmark at the folder with "
    "--flir-video-test-dir or FLIR_VIDEO_TEST_DIR).",
    licence="Teledyne FLIR Non-Commercial Image Dataset License Agreement. Never "
    "committed; per-file SHA-256 is recorded in the results.",
    hr_note="Native 640x512 frames from the camera the model was trained on: the "
    "in-distribution held-out test. Whether its source video overlaps the "
    "train/validation videos has not been checked.",
    root=Path("FLIR_VIDEO_TEST_DIR"),  # resolved at run time from the environment
)

TEST_SETS = {REPO17.name: REPO17, TNO.name: TNO, FLIR_VIDEO_TEST.name: FLIR_VIDEO_TEST}
DEFAULT_SETS = (REPO17.name,)


def protocol_dict() -> dict:
    """Machine-readable copy of the protocol, embedded in results JSON."""
    return {
        "version": PROTOCOL_VERSION,
        "seed": SEED,
        "scales": list(SCALES),
        "degradation": DEGRADATION,
        "ablation_degradations": ABLATION_DEGRADATIONS,
        "io": IO,
        "metrics": {
            "channel": "single (grayscale)",
            "data_range": DATA_RANGE,
            "border_crop": "scale pixels on each side",
            "psnr": PSNR_DEF,
            "ssim": SSIM_DEF,
            "aggregation": "mean over images of per-image values",
        },
        "speed": {
            **{k: (list(v) if isinstance(v, tuple) else v) for k, v in SPEED.items()},
            "output_hw": list(SPEED_OUTPUT_HW),
            "input_hw_per_scale": {str(s): list(speed_input_hw(s)) for s in SCALES},
        },
    }
