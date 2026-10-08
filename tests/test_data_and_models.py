"""Test-set integrity, LR synthesis and model loading (CPU, tiny inputs)."""

import cv2
import numpy as np
import pytest
import torch

from benchmark import data, models
from benchmark import protocol as P

EXPECTED_PARAMS = {2: 688_636, 3: 691_521, 4: 695_560}


def test_repo17_integrity():
    images = data.list_images("repo17")  # verifies every SHA-256 against the manifest
    assert len(images) == 17
    for im in images:
        img = data.load_gray_u8(im.path)
        assert img.dtype == np.uint8 and img.shape == (256, 320)


@pytest.mark.parametrize("scale", P.SCALES)
def test_degradation_matches_training_preparation(scale):
    img = data.load_gray_u8(data.list_images("repo17")[0].path)
    hr = data.modcrop(img, scale)
    assert hr.shape[0] % scale == 0 and hr.shape[1] % scale == 0
    h, w = img.shape
    assert hr.shape == (h // scale * scale, w // scale * scale)
    lr = data.degrade(hr, scale)
    ref = cv2.resize(
        hr, (hr.shape[1] // scale, hr.shape[0] // scale), interpolation=cv2.INTER_CUBIC
    )
    assert np.array_equal(lr, ref)
    with pytest.raises(ValueError):
        data.degrade(img[:-1, :], 2)


def test_ablation_degradation_differs_from_primary():
    hr = data.modcrop(data.load_gray_u8(data.list_images("repo17")[0].path), 4)
    a = data.degrade(hr, 4, "cv2_bicubic")
    b = data.degrade(hr, 4, "pil_bicubic_antialias")
    assert a.shape == b.shape and not np.array_equal(a, b)


@pytest.mark.parametrize("scale", P.SCALES)
def test_thermal_model_runs_on_cpu(scale):
    torch.manual_seed(0)
    net = models.build_thermal_imdn(scale).eval()
    assert models.count_params(net) == EXPECTED_PARAMS[scale]
    lr = np.random.default_rng(0).integers(0, 256, (12, 16), dtype=np.uint8)
    sr = models.run_torch(net, lr, torch.device("cpu"))
    assert sr.dtype == np.uint8 and sr.shape == (12 * scale, 16 * scale)


def test_rgb_path_shapes():
    net = models.build_rgb_imdn(2).eval()
    lr = np.zeros((8, 8), np.uint8)
    sr = models.run_torch(net, lr, torch.device("cpu"), in_channels=3)
    assert sr.shape == (16, 16) and sr.dtype == np.uint8


def test_state_dict_digest_is_container_independent(tmp_path):
    net = models.build_thermal_imdn(2)
    sd = net.state_dict()
    torch.save({"epoch": 1, "model_state_dict": sd}, tmp_path / "ckpt.pth")
    torch.save({f"module.{k}": v for k, v in sd.items()}, tmp_path / "bare.pth")
    d1 = models.state_dict_digest(models.load_state_dict(tmp_path / "ckpt.pth"))
    d2 = models.state_dict_digest(models.load_state_dict(tmp_path / "bare.pth"))
    assert d1 == d2 == models.state_dict_digest(sd)


@pytest.mark.skipif(
    not (P.WEIGHTS_DIR / P.THERMAL_WEIGHTS[2]).exists(), reason="fine-tuned weights are not in git"
)
@pytest.mark.parametrize("scale", P.SCALES)
def test_trained_weights_load_strictly(scale):
    m = models.thermal_imdn(scale)
    assert m.info["params"] == EXPECTED_PARAMS[scale]
    sr = models.apply(m, np.full((10, 10), 128, np.uint8), torch.device("cpu"))
    assert sr.shape == (10 * scale, 10 * scale)
