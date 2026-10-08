"""The benchmark's own PSNR/SSIM must agree with scikit-image on known inputs."""

import math

import cv2
import numpy as np
import pytest
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim

from benchmark import data, metrics


def sk_ssim_protocol(a, b):
    return sk_ssim(
        a,
        b,
        data_range=255,
        gaussian_weights=True,
        sigma=1.5,
        use_sample_covariance=False,
        K1=0.01,
        K2=0.03,
    )


def noisy_pair(seed, shape, sigma):
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 256, shape).astype(np.uint8)
    b = np.clip(a + rng.normal(0, sigma, shape), 0, 255).astype(np.uint8)
    return a, b


@pytest.mark.parametrize(
    "seed,shape,sigma", [(0, (64, 80), 5), (1, (31, 47), 20), (2, (128, 96), 60), (3, (11, 11), 10)]
)
def test_psnr_matches_skimage(seed, shape, sigma):
    a, b = noisy_pair(seed, shape, sigma)
    assert metrics.psnr(b, a) == pytest.approx(sk_psnr(a, b, data_range=255), abs=1e-10)


@pytest.mark.parametrize(
    "seed,shape,sigma", [(0, (64, 80), 5), (1, (31, 47), 20), (2, (128, 96), 60), (3, (11, 11), 10)]
)
def test_ssim_matches_skimage(seed, shape, sigma):
    a, b = noisy_pair(seed, shape, sigma)
    assert metrics.ssim(a, b) == pytest.approx(sk_ssim_protocol(a, b), abs=1e-10)


def test_metrics_on_real_frame_bicubic_round_trip():
    img = data.load_gray_u8(data.list_images("repo17")[0].path)
    for s in (2, 3, 4):
        hr = data.modcrop(img, s)
        lr = data.degrade(hr, s)
        up = cv2.resize(lr, hr.shape[::-1], interpolation=cv2.INTER_CUBIC)
        hr_c, up_c = metrics.crop_border(hr, s), metrics.crop_border(up, s)
        assert metrics.psnr(up_c, hr_c) == pytest.approx(
            sk_psnr(hr_c, up_c, data_range=255), abs=1e-10
        )
        assert metrics.ssim(up_c, hr_c) == pytest.approx(sk_ssim_protocol(up_c, hr_c), abs=1e-10)


def test_known_values():
    a = np.zeros((16, 16), np.uint8)
    b = np.full((16, 16), 255, np.uint8)
    assert metrics.psnr(a, b) == pytest.approx(0.0)  # MSE = 255^2
    assert metrics.psnr(a, a) == math.inf
    assert metrics.ssim(a, a) == pytest.approx(1.0)
    c = a.copy()
    c[0, 0] = 255  # MSE = 255^2 / 256  ->  10*log10(256)
    assert metrics.psnr(c, a) == pytest.approx(10 * math.log10(256))


def test_border_crop_and_types():
    x = np.arange(100, dtype=np.uint8).reshape(10, 10)
    assert metrics.crop_border(x, 2).shape == (6, 6)
    assert metrics.crop_border(x, 2)[0, 0] == x[2, 2]
    assert metrics.crop_border(x, 0) is x
    with pytest.raises(ValueError):
        metrics.crop_border(x, 5)
    with pytest.raises(TypeError):
        metrics.evaluate_pair(x.astype(np.float32), x.astype(np.float32), 2)
    with pytest.raises(ValueError):
        metrics.psnr(x, x[:5])


def test_gaussian_window():
    k = metrics.gaussian_kernel_1d()
    assert k.size == 11
    assert k.sum() == pytest.approx(1.0)
    assert np.allclose(k, k[::-1])
