"""Latency measurement under the protocol in protocol.SPEED."""

from __future__ import annotations

import os
import subprocess
import time

import numpy as np
import torch

from .models import autocast, run_torch


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def summarise(times_s: list[float], trial_medians_ms: list[float] | None = None) -> dict:
    ms = np.asarray(times_s, dtype=np.float64) * 1000.0
    median = float(np.median(ms))
    out = {
        "n": int(ms.size),
        "median_ms": median,
        "p95_ms": float(np.percentile(ms, 95)),
        "mean_ms": float(ms.mean()),
        "min_ms": float(ms.min()),
        "max_ms": float(ms.max()),
        "fps": 1000.0 / median,
    }
    if trial_medians_ms:
        out["trial_medians_ms"] = [round(t, 4) for t in trial_medians_ms]
    return out


@torch.inference_mode()
def model_times(
    model: torch.nn.Module,
    in_hw: tuple[int, int],
    in_channels: int,
    device: torch.device,
    precision: str,
    warmup: int,
    iters: int,
    seed: int,
) -> list[float]:
    """Forward pass only; input already on the device. Seconds per call."""
    g = torch.Generator().manual_seed(seed)
    x = torch.rand((1, in_channels, *in_hw), generator=g).to(device=device)
    times = []
    with autocast(device, precision):
        for _ in range(warmup):
            model(x)
        _sync(device)
        for _ in range(iters):
            _sync(device)
            t0 = time.perf_counter()
            model(x)
            _sync(device)
            times.append(time.perf_counter() - t0)
    return times


def end_to_end_times(
    model: torch.nn.Module,
    in_hw: tuple[int, int],
    in_channels: int,
    device: torch.device,
    precision: str,
    warmup: int,
    iters: int,
    seed: int,
) -> list[float]:
    """uint8 host frame in -> uint8 host frame out (see protocol.SPEED)."""
    rng = np.random.default_rng(seed)
    frame = rng.integers(0, 256, size=in_hw, dtype=np.uint8)
    for _ in range(warmup):
        run_torch(model, frame, device, precision, in_channels)
    _sync(device)
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        out = run_torch(model, frame, device, precision, in_channels)  # .cpu() synchronises
        times.append(time.perf_counter() - t0)
    assert out.dtype == np.uint8
    return times


def gpu_telemetry() -> dict | None:
    """Temperature, SM clock, power and active clock-limit reasons (nvidia-smi)."""
    fields = "temperature.gpu,clocks.sm,clocks.max.sm,power.draw,clocks_event_reasons.active"
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=20,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    vals = [v.strip() for v in out.strip().splitlines()[0].split(",")]
    keys = ["temp_c", "sm_clock", "sm_clock_max", "power_draw", "clock_limit_reasons"]
    tel = dict(zip(keys, vals, strict=False))
    tel["cpu_loadavg_1m"] = round(os.getloadavg()[0], 2)  # other work on the machine
    return tel


def wait_for_quiet(max_temp_c: int, max_load: float, timeout_s: float) -> dict:
    """Wait (bounded) until the GPU is cool and the machine is not busy with other
    work, so every trial starts from a comparable state. Records what it saw."""
    t0 = time.time()
    tel = gpu_telemetry()

    def busy(t):
        return t and (float(t["temp_c"]) > max_temp_c or t["cpu_loadavg_1m"] > max_load)

    while busy(tel) and time.time() - t0 < timeout_s:
        time.sleep(5)
        tel = gpu_telemetry()
    return {
        "waited_s": round(time.time() - t0, 1),
        "target_temp_c": max_temp_c,
        "target_loadavg": max_load,
        "reached": not busy(tel),
        "telemetry": tel,
    }


def peak_memory_mb(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    return torch.cuda.max_memory_allocated(device) / 2**20
