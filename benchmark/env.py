"""Record the software and hardware a run was made on."""

from __future__ import annotations

import platform
import subprocess

import cv2
import numpy as np
import torch

from . import protocol as P


def _run(cmd: list[str]) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=20, check=True)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def _cpu_model() -> str | None:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or None


def git_state() -> dict:
    root = str(P.REPO_ROOT)
    commit = _run(["git", "-C", root, "rev-parse", "HEAD"])
    status = _run(["git", "-C", root, "status", "--porcelain", "--untracked-files=no"])
    return {"commit": commit, "dirty": bool(status)}


def environment(device: torch.device) -> dict:
    env = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu": _cpu_model(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "device": str(device),
        "torch_num_threads": torch.get_num_threads(),
        "git": git_state(),
    }
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        env.update(
            {
                "gpu": torch.cuda.get_device_name(device),
                "gpu_memory_gb": round(props.total_memory / 1e9, 2),
                "gpu_compute_capability": f"{props.major}.{props.minor}",
                "cuda_runtime": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version(),
                "driver": _run(
                    ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
                ),
                "gpu_power_limit": _run(
                    ["nvidia-smi", "--query-gpu=power.limit", "--format=csv,noheader"]
                ),
            }
        )
    return env
