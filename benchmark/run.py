"""Single entry point: quality and speed under benchmark/protocol.py.

    python -m benchmark.run --scale 2 3 4                    # full run (GPU)
    python -m benchmark.run --scale 2 3 4 --sets repo17 tno-lwir
    python -m benchmark.run --quick --device cpu --random-weights   # CI smoke test

Writes results/benchmark.json and results/benchmark.md (quick or random-weight
runs go to results/benchmark-quick.* so they never overwrite the real result).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from . import data, models, report, timing
from . import protocol as P
from .env import environment
from .metrics import evaluate_pair


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="python -m benchmark.run",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--scale", type=int, nargs="+", default=list(P.SCALES), choices=P.SCALES)
    ap.add_argument("--sets", nargs="+", default=list(P.DEFAULT_SETS), choices=sorted(P.TEST_SETS))
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument(
        "--quick",
        action="store_true",
        help="few images, few timing iterations, small speed input (for CI)",
    )
    ap.add_argument(
        "--random-weights",
        action="store_true",
        help="randomly initialised thermal model (pipeline smoke test; no baselines "
        "that need weights)",
    )
    ap.add_argument("--no-speed", action="store_true", help="skip latency measurement")
    ap.add_argument(
        "--no-ablation",
        action="store_true",
        help="skip the degradation sensitivity check (protocol.ABLATION_DEGRADATIONS)",
    )
    ap.add_argument("--weights-dir", type=Path, default=P.WEIGHTS_DIR)
    ap.add_argument(
        "--flir-video-test-dir",
        type=Path,
        default=None,
        help="folder of FLIR ADAS v2 video_thermal_test frames (for --sets flir-video-test)",
    )
    ap.add_argument("--out", type=Path, default=None, help="results JSON path")
    return ap.parse_args(argv)


def select_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but CUDA is not available")
    return torch.device(name)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def deterministic(on: bool) -> None:
    """Quality pass: deterministic kernels. Speed pass: cuDNN autotuning."""
    torch.backends.cudnn.deterministic = on
    torch.backends.cudnn.benchmark = not on and P.SPEED["cudnn_benchmark"]
    torch.use_deterministic_algorithms(on, warn_only=True)
    # Strict fp32 in both passes (Ampere would otherwise use TF32 for convolutions).
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


def build_methods(scale: int, args: argparse.Namespace) -> tuple[list[models.Method], list[dict]]:
    methods = [models.bicubic(scale)]
    skipped = []
    if args.random_weights:
        skipped.append(
            {
                "method": "imdn_rgb_pretrained",
                "scale": scale,
                "reason": "--random-weights smoke test",
            }
        )
    else:
        try:
            methods.append(models.rgb_imdn(scale, args.weights_dir))
        except FileNotFoundError as e:
            skipped.append({"method": "imdn_rgb_pretrained", "scale": scale, "reason": str(e)})
    methods.append(models.thermal_imdn(scale, args.weights_dir, random_init=args.random_weights))
    return methods, skipped


def evaluate_quality(
    method: models.Method,
    pairs: list[tuple[str, np.ndarray, np.ndarray]],
    scale: int,
    device: torch.device,
    precision: str,
) -> dict:
    prepared = models.prepare(method, device) if method.kind == "torch" else None
    per_image = []
    for name, hr, lr in pairs:
        sr = models.apply(method, lr, device, precision, prepared)
        if sr.shape != hr.shape:
            raise RuntimeError(f"{method.key}: output {sr.shape} != HR {hr.shape} for {name}")
        m = evaluate_pair(sr, hr, scale)
        per_image.append({"image": name, **m})
    return {
        "psnr_mean": float(np.mean([r["psnr"] for r in per_image])),
        "ssim_mean": float(np.mean([r["ssim"] for r in per_image])),
        "n_images": len(per_image),
        "per_image": per_image,
    }


def measure_speed(
    args: argparse.Namespace, device: torch.device, precisions: list[str]
) -> tuple[list[dict], dict]:
    """Interleaved trials (trial -> scale -> precision) so that thermal drift of a
    laptop GPU hits every configuration alike; samples of all trials are pooled."""
    deterministic(False)
    cfg = P.QUICK if args.quick else P.SPEED
    warmup, iters, trials = cfg["warmup_iters"], cfg["timed_iters"], cfg["trials"]
    out_hw = P.QUICK["speed_output_hw"] if args.quick else P.SPEED_OUTPUT_HW
    telemetry: dict = {}
    nets = {}
    for scale in args.scale:
        method = models.thermal_imdn(scale, args.weights_dir, random_init=args.random_weights)
        nets[scale] = (method, models.prepare(method, device))
    samples: dict = {}
    for trial in range(trials):
        if device.type == "cuda" and not args.quick:
            telemetry[f"before_trial_{trial + 1}"] = timing.wait_for_cooldown(
                P.SPEED["cooldown_temp_c"], P.SPEED["cooldown_timeout_s"]
            )
        for scale, (_method, net) in nets.items():
            in_hw = (out_hw[0] // scale, out_hw[1] // scale)
            for prec in precisions:
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                m = timing.model_times(net, in_hw, 1, device, prec, warmup, iters, P.SEED + trial)
                e = timing.end_to_end_times(
                    net, in_hw, 1, device, prec, warmup, iters, P.SEED + trial
                )
                rec = samples.setdefault(
                    (scale, prec),
                    {"model": [], "e2e": [], "mt": [], "et": [], "mem": 0.0, "in_hw": in_hw},
                )
                rec["model"] += m
                rec["e2e"] += e
                rec["mt"].append(float(np.median(m)) * 1000)
                rec["et"].append(float(np.median(e)) * 1000)
                rec["mem"] = max(rec["mem"], timing.peak_memory_mb(device) or 0.0)
                print(
                    f"[speed trial {trial + 1}/{trials} x{scale} {prec}] model median "
                    f"{rec['mt'][-1]:.3f} ms, end-to-end median {rec['et'][-1]:.3f} ms",
                    flush=True,
                )
        if device.type == "cuda":
            telemetry[f"after_trial_{trial + 1}"] = timing.gpu_telemetry()
    entries = []
    for (scale, prec), rec in samples.items():
        in_hw = rec["in_hw"]
        entry = {
            "scale": scale,
            "method": "imdn_thermal",
            "precision": prec,
            "device": str(device),
            "batch": 1,
            "input_hw": list(in_hw),
            "output_hw": [in_hw[0] * scale, in_hw[1] * scale],
            "warmup_iters": warmup,
            "timed_iters": iters,
            "trials": trials,
            "model": timing.summarise(rec["model"], rec["mt"]),
            "end_to_end": timing.summarise(rec["e2e"], rec["et"]),
            "peak_memory_mb": rec["mem"] if device.type == "cuda" else None,
        }
        entries.append(entry)
        lat, e2e = entry["model"], entry["end_to_end"]
        print(
            f"[speed x{scale}] {prec}: {in_hw[1]}x{in_hw[0]} -> {entry['output_hw'][1]}x"
            f"{entry['output_hw'][0]}  model median {lat['median_ms']:.3f} ms (p95 "
            f"{lat['p95_ms']:.3f}) = {lat['fps']:.1f} FPS; end-to-end median "
            f"{e2e['median_ms']:.3f} ms = {e2e['fps']:.1f} FPS",
            flush=True,
        )
    return entries, telemetry


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if args.flir_video_test_dir:
        os.environ["FLIR_VIDEO_TEST_DIR"] = str(args.flir_video_test_dir)
    device = select_device(args.device)
    trial = args.quick or args.random_weights
    out = args.out or P.REPO_ROOT / "results" / (
        "benchmark-quick.json" if trial else "benchmark.json"
    )
    precisions = list(P.SPEED["precisions"]) if device.type == "cuda" else ["fp32"]
    t_start = time.time()

    seed_everything(P.SEED)
    deterministic(True)

    results: dict = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "command": "python -m benchmark.run " + " ".join(sys.argv[1:] if argv is None else argv),
        "quick": args.quick,
        "random_weights": args.random_weights,
        "protocol": P.protocol_dict(),
        "environment": environment(device),
        "test_sets": {},
        "methods": {},
        "quality": [],
        "speed": [],
        "skipped": [],
    }

    for set_name in args.sets:
        ts = P.TEST_SETS[set_name]
        images = data.list_images(set_name)
        if args.quick:
            images = images[: P.QUICK["max_images"]]
        results["test_sets"][set_name] = {
            "description": ts.description,
            "source": ts.source,
            "licence": ts.licence,
            "hr_note": ts.hr_note,
            "selection": ts.exclude_note or None,
            "n_images": len(images),
            "images": [{"name": im.name, "sha256": im.sha256} for im in images],
        }
        hr_full = [(im.name, data.load_gray_u8(im.path)) for im in images]

        for scale in args.scale:
            methods, skipped = build_methods(scale, args)
            results["skipped"].extend(s for s in skipped if s not in results["skipped"])
            for m in methods:
                results["methods"].setdefault(m.key, {"label": m.label, "per_scale": {}})
                results["methods"][m.key]["per_scale"][str(scale)] = m.info
            degradations = [P.PRIMARY_DEGRADATION]
            if not args.no_ablation and set_name in P.ABLATION_SETS:
                degradations += list(P.ABLATION_DEGRADATIONS)
            for deg in degradations:
                pairs = []
                for name, img in hr_full:
                    hr = data.modcrop(img, scale)
                    pairs.append((name, hr, data.degrade(hr, scale, deg)))
                for m in methods:
                    if m.kind == "opencv":
                        precs = ["uint8"]
                    elif m.key == "imdn_thermal" and deg == P.PRIMARY_DEGRADATION:
                        precs = precisions
                    else:
                        precs = ["fp32"]
                    for prec in precs:
                        q = evaluate_quality(
                            m, pairs, scale, device, "fp32" if prec == "uint8" else prec
                        )
                        results["quality"].append(
                            {
                                "set": set_name,
                                "scale": scale,
                                "degradation": deg,
                                "method": m.key,
                                "precision": prec,
                                "device": str(device),
                                **q,
                            }
                        )
                        print(
                            f"[{set_name} x{scale} {deg}] {m.key:<20} {prec:<5} "
                            f"PSNR {q['psnr_mean']:.3f} dB  SSIM {q['ssim_mean']:.4f}  "
                            f"(n={q['n_images']})",
                            flush=True,
                        )

    if not args.no_speed:
        results["speed"], results["speed_telemetry"] = measure_speed(args, device, precisions)

    results["elapsed_s"] = round(time.time() - t_start, 1)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=1)
        f.write("\n")
    md = out.with_suffix(".md")
    md.write_text(report.render_markdown(results))
    print(f"wrote {out} and {md}")
    return results


if __name__ == "__main__":
    main()
