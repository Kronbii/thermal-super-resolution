"""PSNR and SSIM for single-channel images.

Implemented directly in NumPy (float64) so that the benchmark does not depend
on scikit-image; tests/test_metrics.py checks both against scikit-image.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from . import protocol as P


def crop_border(img: np.ndarray, border: int) -> np.ndarray:
    if border <= 0:
        return img
    if img.shape[0] <= 2 * border or img.shape[1] <= 2 * border:
        raise ValueError(f"image {img.shape} too small for border {border}")
    return img[border:-border, border:-border]


def _check_pair(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    if a.ndim != 2:
        raise ValueError(f"expected a single-channel 2-D image, got {a.shape}")
    return a.astype(np.float64), b.astype(np.float64)


def psnr(sr: np.ndarray, hr: np.ndarray, data_range: float = P.DATA_RANGE) -> float:
    """Peak signal-to-noise ratio in dB."""
    x, y = _check_pair(sr, hr)
    mse = float(np.mean((x - y) ** 2))
    if mse == 0.0:
        return math.inf
    return 10.0 * math.log10(data_range**2 / mse)


def gaussian_kernel_1d(sigma: float = P.SSIM_SIGMA, radius: int = P.SSIM_RADIUS) -> np.ndarray:
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def _filter_valid(img: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Separable 'valid' filtering (the window never leaves the image)."""
    n = k.size
    rows = sliding_window_view(img, n, axis=1) @ k
    return sliding_window_view(rows, n, axis=0) @ k


def ssim(
    sr: np.ndarray,
    hr: np.ndarray,
    data_range: float = P.DATA_RANGE,
    sigma: float = P.SSIM_SIGMA,
    radius: int = P.SSIM_RADIUS,
    k1: float = P.SSIM_K1,
    k2: float = P.SSIM_K2,
) -> float:
    """Mean structural similarity (Wang et al. 2004), Gaussian-weighted."""
    x, y = _check_pair(sr, hr)
    win = 2 * radius + 1
    if x.shape[0] < win or x.shape[1] < win:
        raise ValueError(f"image {x.shape} smaller than the {win}x{win} SSIM window")
    k = gaussian_kernel_1d(sigma, radius)
    mu_x = _filter_valid(x, k)
    mu_y = _filter_valid(y, k)
    var_x = _filter_valid(x * x, k) - mu_x * mu_x
    var_y = _filter_valid(y * y, k) - mu_y * mu_y
    cov = _filter_valid(x * y, k) - mu_x * mu_y
    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2
    num = (2 * mu_x * mu_y + c1) * (2 * cov + c2)
    den = (mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2)
    return float(np.mean(num / den))


def evaluate_pair(sr: np.ndarray, hr: np.ndarray, scale: int) -> dict:
    """Protocol metrics for one SR/HR pair (both uint8, same shape)."""
    if sr.dtype != np.uint8 or hr.dtype != np.uint8:
        raise TypeError("protocol metrics are defined on uint8 images")
    b = P.border_for(scale)
    s, h = crop_border(sr, b), crop_border(hr, b)
    return {"psnr": psnr(s, h), "ssim": ssim(s, h)}
