"""Replicate the timing method of scripts/test-thermal-model.ipynb (for the audit).

    python -m benchmark.legacy_timing     # -> results/legacy_timing.json

The 2025-09-23 evaluation report (229.6 FPS at x2) timed ``model(lr)`` with
``time.time()`` and no ``torch.cuda.synchronize()``. CUDA kernels are queued
asynchronously, so that window closes as soon as the kernels are launched; the
GPU work finishes later, outside it (here at the ``.item()`` that follows, in the
notebook at the PSNR computation). This script repeats that loop so the audit
can show what the method reports on the same laptop. These numbers are NOT a
speed measurement; benchmark.run is.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from . import models
from . import protocol as P
from .env import environment

OUT = P.REPO_ROOT / "results" / "legacy_timing.json"
N_IMAGES = 1100  # the report's image count


@torch.no_grad()
def notebook_loop(model: torch.nn.Module, in_hw: tuple[int, int], device: torch.device) -> dict:
    rng = np.random.default_rng(P.SEED)
    frame = rng.integers(0, 256, size=in_hw, dtype=np.uint8)
    total = 0.0
    for _ in range(N_IMAGES):
        lr = torch.from_numpy(frame.astype(np.float32) / 255.0)[None, None].to(device)
        start = time.time()  # as in the notebook
        sr = model(lr)
        sr = torch.clamp(sr, 0, 1)
        total += time.time() - start
        float(sr.mean().item())  # the notebook's PSNR/.item() forces completion here
    return {"n": N_IMAGES, "total_s": total, "fps_as_reported": 1.0 / (total / N_IMAGES)}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m benchmark.legacy_timing", description=__doc__)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)
    if not torch.cuda.is_available():
        raise SystemExit("needs a CUDA GPU (the effect is specific to asynchronous execution)")
    device = torch.device("cuda")
    torch.manual_seed(P.SEED)
    res = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "what": "FPS computed exactly like scripts/test-thermal-model.ipynb: time.time() around "
        "model(lr) + clamp, no torch.cuda.synchronize(), FPS = N / summed time. Invalid as "
        "a speed measurement; kept to explain the old README figures.",
        "environment": environment(device),
        "per_scale": {},
    }
    for scale in P.SCALES:
        m = models.thermal_imdn(scale)
        net = models.prepare(m, device)
        in_hw = P.speed_input_hw(scale)
        notebook_loop(net, in_hw, device)  # first pass: CUDA context, allocator
        r = notebook_loop(net, in_hw, device)
        r["input_hw"] = list(in_hw)
        res["per_scale"][str(scale)] = r
        print(
            f"x{scale}: {in_hw[1]}x{in_hw[0]} input, notebook-style FPS {r['fps_as_reported']:.1f}"
        )
    args.out.write_text(json.dumps(res, indent=1) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
