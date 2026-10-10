"""Methods under test: the fine-tuned thermal IMDN and the baselines.

Every method maps a uint8 LR frame (H, W) to a uint8 SR frame (sH, sW).
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch

from model.architecture import IMDN

from . import protocol as P
from .data import sha256_file


def _strip_module(sd: dict) -> dict:
    if any(k.startswith("module.") for k in sd):
        return {k[len("module.") :] if k.startswith("module.") else k: v for k, v in sd.items()}
    return sd


def load_state_dict(path: Path) -> dict:
    """Load weights from a training checkpoint or a bare state dict."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        ckpt = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    return _strip_module(dict(ckpt))


def state_dict_digest(sd: dict) -> str:
    """SHA-256 over sorted (name, dtype, shape, bytes): independent of the container."""
    h = hashlib.sha256()
    for k in sorted(sd):
        t = sd[k].detach().cpu().contiguous()
        h.update(k.encode())
        h.update(str(t.dtype).encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def build_thermal_imdn(scale: int) -> torch.nn.Module:
    return IMDN(in_nc=1, out_nc=1, upscale=scale)


def build_rgb_imdn(scale: int) -> torch.nn.Module:
    return IMDN(in_nc=3, out_nc=3, upscale=scale)


@dataclass
class Method:
    key: str
    label: str
    scale: int
    kind: str  # "torch" or "opencv"
    model: torch.nn.Module | None = None
    in_channels: int = 1
    info: dict = field(default_factory=dict)
    fn: Callable[[np.ndarray], np.ndarray] | None = None


def bicubic(scale: int) -> Method:
    def fn(lr: np.ndarray) -> np.ndarray:
        h, w = lr.shape
        return cv2.resize(lr, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)

    return Method(
        key="bicubic",
        label="Bicubic (OpenCV)",
        scale=scale,
        kind="opencv",
        info={"definition": "cv2.resize(LR, (sW, sH), interpolation=cv2.INTER_CUBIC)"},
        fn=fn,
    )


def _weights_info(path: Path, sd: dict) -> dict:
    return {
        "file": str(path.relative_to(P.REPO_ROOT))
        if path.is_relative_to(P.REPO_ROOT)
        else str(path),
        "file_sha256": sha256_file(path),
        "state_dict_sha256": state_dict_digest(sd),
    }


def thermal_imdn(
    scale: int, weights_dir: Path = P.WEIGHTS_DIR, random_init: bool = False
) -> Method:
    if random_init:
        torch.manual_seed(P.SEED)
    model = build_thermal_imdn(scale)
    if random_init:
        info = {"weights": "RANDOM INIT (smoke test only; numbers are meaningless)"}
    else:
        path = weights_dir / P.THERMAL_WEIGHTS[scale]
        if not path.exists():
            path = weights_dir / P.EXPORTED_WEIGHTS.format(s=scale)
        if not path.exists():
            raise FileNotFoundError(
                f"fine-tuned x{scale} weights not found in {weights_dir} (looked for "
                f"{P.THERMAL_WEIGHTS[scale]} and {path.name}). They are not in git; see README "
                "'Reproduce'. Use --random-weights for a pipeline smoke test."
            )
        sd = load_state_dict(path)
        model.load_state_dict(sd, strict=True)
        info = _weights_info(path, sd)
    info["params"] = count_params(model)
    return Method(
        key="imdn_thermal",
        label="IMDN, thermal fine-tune (this repo)",
        scale=scale,
        kind="torch",
        model=model.eval(),
        in_channels=1,
        info=info,
    )


def rgb_imdn(scale: int, weights_dir: Path = P.WEIGHTS_DIR) -> Method:
    """Upstream IMDN (Hui et al. 2019, DIV2K/RGB), no thermal fine-tuning."""
    path = weights_dir / P.UPSTREAM_WEIGHTS[scale]
    if not path.exists():
        raise FileNotFoundError(
            f"upstream IMDN weights not found: {path} "
            f"(download from {P.UPSTREAM_WEIGHTS_URL.format(s=scale)})"
        )
    sd = load_state_dict(path)
    model = build_rgb_imdn(scale)
    model.load_state_dict(sd, strict=True)
    info = _weights_info(path, sd)
    info["params"] = count_params(model)
    info["source"] = P.UPSTREAM_WEIGHTS_URL.format(s=scale)
    return Method(
        key="imdn_rgb_pretrained",
        label="IMDN, upstream RGB weights (no thermal fine-tune)",
        scale=scale,
        kind="torch",
        model=model.eval(),
        in_channels=3,
        info=info,
    )


PRECISIONS = ("fp32", "fp16")


def autocast(device: torch.device, precision: str):
    """fp16 = autocast to float16 with float32 weights; fp32 = plain float32."""
    if precision not in PRECISIONS:
        raise ValueError(f"unknown precision {precision!r}")
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=precision == "fp16")


def prepare(method: Method, device: torch.device) -> torch.nn.Module:
    """Copy of the method's model on the device (weights stay float32)."""
    import copy

    return copy.deepcopy(method.model).to(device=device, dtype=torch.float32).eval()


@torch.inference_mode()
def run_torch(
    model: torch.nn.Module,
    lr: np.ndarray,
    device: torch.device,
    precision: str = "fp32",
    in_channels: int = 1,
) -> np.ndarray:
    """uint8 LR (H, W) on the host -> uint8 SR (sH, sW) on the host."""
    if not lr.flags.writeable:
        lr = lr.copy()
    x = torch.from_numpy(lr).to(device).to(torch.float32).div_(255.0)[None, None]
    if in_channels == 3:
        x = x.expand(1, 3, *x.shape[-2:])
    with autocast(device, precision):
        y = model(x)
    y = y.to(torch.float32).clamp_(0.0, 1.0)
    if y.shape[1] == 3:
        y = y.mean(dim=1, keepdim=True)
    y = y.mul_(255.0).round_().to(torch.uint8)
    return y[0, 0].cpu().numpy()


def apply(
    method: Method,
    lr: np.ndarray,
    device: torch.device,
    precision: str = "fp32",
    prepared: torch.nn.Module | None = None,
) -> np.ndarray:
    if method.kind == "opencv":
        return method.fn(lr)
    model = prepared if prepared is not None else prepare(method, device)
    return run_torch(model, lr, device, precision, method.in_channels)
