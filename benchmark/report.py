"""Render results/benchmark.json into Markdown and keep the docs in sync.

    python -m benchmark.report            # rewrite results/benchmark.md and the
                                          # generated blocks in README.md and
                                          # docs/portfolio.md
    python -m benchmark.report --check    # exit 1 if any of them is stale (CI)

Generated blocks sit between ``<!-- BENCHMARK:<NAME>:START -->`` and
``<!-- BENCHMARK:<NAME>:END -->``. Numbers outside those blocks are not allowed
to be benchmark numbers: edit the protocol and rerun instead.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from . import protocol as P

RESULTS = P.REPO_ROOT / "results" / "benchmark.json"
TARGETS = (
    P.REPO_ROOT / "README.md",
    P.REPO_ROOT / "docs" / "portfolio.md",
    P.REPO_ROOT / "docs" / "claims-audit.md",
)

SHORT = {
    "bicubic": "Bicubic (OpenCV)",
    "imdn_rgb_pretrained": "IMDN, upstream RGB weights",
    "imdn_thermal": "**IMDN, thermal fine-tune (this repo)**",
}
SET_TITLE = {
    "repo17": "test-data/ (17 FLIR ADAS v2 frames, 320x256)",
    "tno-lwir": "TNO Image Fusion Dataset, thermal band (51 images, other sensors)",
    "flir-video-test": "FLIR ADAS v2 video_thermal_test (native 640x512)",
}


def _q(
    results: dict,
    set_name: str,
    scale: int,
    method: str,
    precision: str | None = None,
    degradation: str = P.PRIMARY_DEGRADATION,
):
    for e in results["quality"]:
        if (e["set"], e["scale"], e["method"], e["degradation"]) == (
            set_name,
            scale,
            method,
            degradation,
        ) and (precision is None or e["precision"] == precision):
            return e
    return None


def ablation_table(results: dict) -> str:
    rows = [e for e in results["quality"] if e["degradation"] != P.PRIMARY_DEGRADATION]
    if not rows:
        return "_Degradation sensitivity check not run._"
    lines = [
        "Same frames, same methods, but the LR input is made with an anti-aliased bicubic "
        "downscale (PIL, MATLAB-like) instead of the OpenCV bicubic used for training. "
        "This is a sensitivity check, not the headline protocol.",
        "",
        "| Scale | Method | PSNR (dB), OpenCV LR | PSNR (dB), anti-aliased LR |",
        "|:---:|:---|---:|---:|",
    ]
    for e in sorted(rows, key=lambda e: (e["set"], e["scale"])):
        prim = _q(
            results, e["set"], e["scale"], e["method"], None if e["method"] == "bicubic" else "fp32"
        )
        lines.append(
            f"| x{e['scale']} | {SHORT[e['method']]} | "
            f"{prim['psnr_mean']:.2f} | {e['psnr_mean']:.2f} |"
        )
    return "\n".join(lines)


def _s(results: dict, scale: int, precision: str):
    for e in results["speed"]:
        if e["scale"] == scale and e["precision"] == precision:
            return e
    return None


def _scales(results: dict) -> list[int]:
    return sorted({e["scale"] for e in results["quality"]} | {e["scale"] for e in results["speed"]})


def hardware(results: dict) -> str:
    env = results["environment"]
    if env.get("gpu"):
        cudnn = env.get("cudnn")
        cudnn_s = f"{cudnn // 10000}.{cudnn // 100 % 100}.{cudnn % 100}" if cudnn else "?"
        power = env.get("gpu_power_limit")
        power_s = f", power limit {power}" if power and "N/A" not in power else ""
        return (
            f"{env['gpu']} ({env.get('gpu_memory_gb')} GB{power_s}), driver "
            f"{env.get('driver')}, CUDA {env.get('cuda_runtime')}, cuDNN {cudnn_s}, "
            f"PyTorch {env['torch']}, Python {env['python']}"
        )
    return f"CPU {env.get('cpu')}, PyTorch {env['torch']}, Python {env['python']}"


def provenance(results: dict) -> str:
    g = results["environment"].get("git", {})
    commit = (g.get("commit") or "unknown")[:10]
    dirty = " (uncommitted changes)" if g.get("dirty") else ""
    return (
        f"Protocol v{results['protocol']['version']} (`benchmark/protocol.py`), "
        f"`{results['command']}`, commit `{commit}`{dirty}, {results['generated_utc']}."
    )


def warning(results: dict) -> str:
    if results.get("random_weights"):
        return "> **Smoke test with random weights: these numbers mean nothing.**\n\n"
    if results.get("quick"):
        return "> **Quick mode (few images / iterations): not a result.**\n\n"
    return ""


def quality_table(results: dict, set_name: str) -> str:
    ts = results["test_sets"][set_name]
    lines = [
        f"**{SET_TITLE.get(set_name, set_name)}**, n = {ts['n_images']}. {ts['hr_note']}",
        "",
        "| Scale | Method | PSNR (dB) | SSIM | PSNR gain vs bicubic (dB) |",
        "|:---:|:---|---:|---:|---:|",
    ]
    for scale in _scales(results):
        base = _q(results, set_name, scale, "bicubic")
        for key in ("bicubic", "imdn_rgb_pretrained", "imdn_thermal"):
            e = _q(results, set_name, scale, key, None if key == "bicubic" else "fp32")
            if e is None:
                continue
            gain = (
                "-"
                if key == "bicubic" or base is None
                else f"{e['psnr_mean'] - base['psnr_mean']:+.2f}"
            )
            lines.append(
                f"| x{scale} | {SHORT[key]} | {e['psnr_mean']:.2f} | "
                f"{e['ssim_mean']:.4f} | {gain} |"
            )
    fp16 = [
        (
            _q(results, set_name, s, "imdn_thermal", "fp32"),
            _q(results, set_name, s, "imdn_thermal", "fp16"),
        )
        for s in _scales(results)
    ]
    fp16 = [(a, b) for a, b in fp16 if a and b]
    if fp16:
        worst = max(abs(a["psnr_mean"] - b["psnr_mean"]) for a, b in fp16)
        bound = "less than 0.001" if worst < 0.001 else f"within {worst:.3f}"
        lines += [
            "",
            f"Thermal model with fp16 autocast: PSNR {bound} dB from fp32 at every scale.",
        ]
    return "\n".join(lines)


def speed_table(results: dict) -> str:
    if not results["speed"]:
        return "_Speed not measured in this run._"
    lines = [
        "| Scale | Input -> output (W x H) | Precision | Model latency median / p95 (ms) "
        "| Trial medians min-max (ms) | Model FPS | End-to-end median / p95 (ms) "
        "| End-to-end FPS |",
        "|:---:|:---:|:---:|---:|---:|---:|---:|---:|",
    ]
    for e in sorted(results["speed"], key=lambda e: (e["scale"], e["precision"] != "fp32")):
        i, o, m, x = e["input_hw"], e["output_hw"], e["model"], e["end_to_end"]
        tm = m.get("trial_medians_ms") or [m["median_ms"]]
        lines.append(
            f"| x{e['scale']} | {i[1]}x{i[0]} -> {o[1]}x{o[0]} | {e['precision']} | "
            f"{m['median_ms']:.2f} / {m['p95_ms']:.2f} | {min(tm):.2f}-{max(tm):.2f} | "
            f"{m['fps']:.0f} | {x['median_ms']:.2f} / {x['p95_ms']:.2f} | {x['fps']:.0f} |"
        )
    e0 = results["speed"][0]
    gpu = e0["device"].startswith("cuda")
    sync = "`torch.cuda.synchronize()` around every timed call, " if gpu else ""
    where = "on the GPU" if gpu else "in memory"
    lines += [
        "",
        f"Batch 1. {e0.get('trials', 1)} interleaved trials x ({e0['warmup_iters']} warm-up + "
        f"{e0['timed_iters']} timed iterations), samples pooled; {sync}FPS = 1000 / median. "
        f"*Model latency*: forward pass only, input already {where}. *End-to-end*: uint8 "
        "frame in host memory to uint8 frame in host memory (conversion, copies, forward, "
        "rounding). fp32 has TF32 disabled; fp16 is `torch.autocast(float16)` with float32 "
        "weights. Plain PyTorch eager mode (no TensorRT, CUDA graphs or torch.compile).",
        "",
        f"Hardware: {hardware(results)}.",
    ]
    cond = conditions(results)
    if cond:
        lines += ["", cond]
    return "\n".join(lines)


def conditions(results: dict) -> str:
    """Thermal / load state during timing (laptop GPUs throttle)."""
    tel = results.get("speed_telemetry") or {}
    trials = [v for k, v in tel.items() if k.startswith("after_trial") and v]
    if not trials:
        return ""
    temps = [float(t["temp_c"]) for t in trials]
    clocks = [int(t["sm_clock"].split()[0]) for t in trials]
    txt = (
        f"Conditions: GPU at {min(temps):.0f}-{max(temps):.0f} °C and SM clock "
        f"{min(clocks)}-{max(clocks)} MHz (max {trials[0]['sm_clock_max']}) at the end of "
        "the trials"
    )
    loads = [t["cpu_loadavg_1m"] for t in trials if "cpu_loadavg_1m" in t]
    if loads:
        txt += f", 1-min CPU load average {min(loads):.1f}-{max(loads):.1f}"
    starts = [
        float(v["telemetry"]["temp_c"])
        for k, v in tel.items()
        if k.startswith("before_trial") and v.get("telemetry")
    ]
    if starts:
        txt += f"; trials started at {min(starts):.0f}-{max(starts):.0f} °C"
    return txt + ". Treat the trial spread as the uncertainty of these timings."


def _headline_numbers(results: dict) -> dict | None:
    set_name = (
        "repo17" if "repo17" in results["test_sets"] else next(iter(results["test_sets"]), None)
    )
    if set_name is None:
        return None
    out = {"set": set_name, "rows": []}
    for s in _scales(results):
        t = _q(results, set_name, s, "imdn_thermal", "fp32")
        b = _q(results, set_name, s, "bicubic")
        sp32, sp16 = _s(results, s, "fp32"), _s(results, s, "fp16")
        if t and b:
            out["rows"].append({"scale": s, "t": t, "b": b, "fp32": sp32, "fp16": sp16})
    return out


def headline(results: dict) -> str:
    h = _headline_numbers(results)
    if not h or not h["rows"]:
        return "_No results._"
    r2 = next((r for r in h["rows"] if r["scale"] == 2), h["rows"][0])
    s = r2["scale"]
    txt = (
        f"{warning(results)}Measured with `python -m benchmark.run` (protocol v"
        f"{results['protocol']['version']}): at x{s} on the 17 frames in `test-data/`, the "
        f"fine-tuned model reaches **{r2['t']['psnr_mean']:.2f} dB PSNR / "
        f"{r2['t']['ssim_mean']:.4f} SSIM**, against {r2['b']['psnr_mean']:.2f} dB / "
        f"{r2['b']['ssim_mean']:.4f} for bicubic interpolation"
    )
    sp = r2["fp16"] or r2["fp32"]
    if sp:
        i, o = sp["input_hw"], sp["output_hw"]
        gpu = results["environment"].get("gpu") or "CPU"
        prec = "fp16 autocast" if sp["precision"] == "fp16" else sp["precision"]
        txt += (
            f". One {i[1]}x{i[0]} -> {o[1]}x{o[0]} frame takes **{sp['model']['median_ms']:.2f} ms "
            f"median** for the forward pass with {prec} on an {gpu} "
            f"({sp['model']['fps']:.0f} FPS; {sp['end_to_end']['median_ms']:.2f} ms, "
            f"{sp['end_to_end']['fps']:.0f} FPS including host-device copies and conversion)"
        )
        f32 = r2["fp32"]
        if f32 and f32 is not sp:
            m32 = f32["model"]
            txt += f"; in fp32 it takes {m32['median_ms']:.2f} ms ({m32['fps']:.0f} FPS)"
    return txt + "."


def _lat(sp: dict) -> str:
    return f"{sp['model']['median_ms']:.1f} ms median ({sp['model']['fps']:.0f} FPS, batch 1)"


def _prec(sp: dict) -> str:
    return "fp16 autocast" if sp["precision"] == "fp16" else sp["precision"]


def measurements(results: dict) -> str:
    h = _headline_numbers(results)
    if not h or not h["rows"]:
        return "_No results._"
    lines = [warning(results).strip()] if warning(results) else []
    rows = {r["scale"]: r for r in h["rows"]}
    gpu = (results["environment"].get("gpu") or "CPU").replace("NVIDIA GeForce ", "")
    tiles = []
    if 2 in rows:
        r = rows[2]
        tiles.append(
            f"- **x2 PSNR**: {r['t']['psnr_mean']:.2f} dB (bicubic {r['b']['psnr_mean']:.2f} dB) "
            "on 17 FLIR ADAS v2 frames, protocol v1.0"
        )
        sp = r["fp16"] or r["fp32"]
        if sp:
            tiles.append(f"- **x2 latency**: {_lat(sp)} for 320x256 -> 640x512, {_prec(sp)}, {gpu}")
    if 4 in rows and (rows[4]["fp16"] or rows[4]["fp32"]):
        sp = rows[4]["fp16"] or rows[4]["fp32"]
        tiles.append(f"- **x4 latency**: {_lat(sp)} for 160x128 -> 640x512, {_prec(sp)}, {gpu}")
    t = _q(results, "tno-lwir", 2, "imdn_thermal", "fp32")
    b = _q(results, "tno-lwir", 2, "bicubic")
    if t and b:
        tiles.append(
            f"- **Other cameras, x2**: {t['psnr_mean'] - b['psnr_mean']:+.2f} dB PSNR over "
            "bicubic on 51 TNO thermal images"
        )
    if tiles:
        lines += ["Suggested portfolio tiles (label: value, context):", "", *tiles, "", "All:", ""]
    lines += ["| Label | Value | Context |", "|:---|:---|:---|"]
    for r in h["rows"]:
        s = r["scale"]
        lines.append(
            f"| PSNR / SSIM, x{s} | {r['t']['psnr_mean']:.2f} dB / {r['t']['ssim_mean']:.4f} "
            f"| 17 frames in test-data/, bicubic {r['b']['psnr_mean']:.2f} dB / "
            f"{r['b']['ssim_mean']:.4f}, border {s} px, uint8 |"
        )
    for r in h["rows"]:
        for prec in ("fp32", "fp16"):
            sp = r[prec]
            if sp:
                i, o = sp["input_hw"], sp["output_hw"]
                lines.append(
                    f"| Latency, x{r['scale']} {prec} | {sp['model']['median_ms']:.2f} ms median "
                    f"(p95 {sp['model']['p95_ms']:.2f}), {sp['model']['fps']:.0f} FPS | "
                    f"{i[1]}x{i[0]} -> {o[1]}x{o[0]}, batch 1, forward only; end-to-end "
                    f"{sp['end_to_end']['median_ms']:.2f} ms |"
                )
    if "tno-lwir" in results["test_sets"]:
        for r in h["rows"]:
            t = _q(results, "tno-lwir", r["scale"], "imdn_thermal", "fp32")
            b = _q(results, "tno-lwir", r["scale"], "bicubic")
            if t and b:
                lines.append(
                    f"| Cross-sensor PSNR, x{r['scale']} | {t['psnr_mean']:.2f} dB "
                    f"(bicubic {b['psnr_mean']:.2f} dB) | 51 TNO thermal images, "
                    "other cameras |"
                )
    lines.append(f"\nHardware: {hardware(results)}.")
    return "\n".join(lines)


def _gains(results: dict, set_name: str) -> str:
    out = []
    for s in _scales(results):
        t = _q(results, set_name, s, "imdn_thermal", "fp32")
        b = _q(results, set_name, s, "bicubic")
        if t and b:
            out.append(f"{t['psnr_mean'] - b['psnr_mean']:+.2f}")
    return "/".join(out) + " dB"


def cv_bullets(results: dict) -> str:
    h = _headline_numbers(results)
    if not h or not h["rows"]:
        return "_No results._"
    scales = "/".join(f"x{r['scale']}" for r in h["rows"])
    text = (
        "- Adapted an RGB-pretrained IMDN super-resolution network (0.69M parameters) to "
        f"single-channel thermal images ({scales}) by fine-tuning on 10,697 FLIR ADAS v2 frames; "
        f"PSNR gain over bicubic of {_gains(results, h['set'])} on 17 validation-split frames"
    )
    if "tno-lwir" in results["test_sets"]:
        text += f" and {_gains(results, 'tno-lwir')} on 51 images from other thermal cameras (TNO)"
    b = [text + "."]
    r2 = next((r for r in h["rows"] if r["scale"] == 2), h["rows"][0])
    sp = r2["fp16"] or r2["fp32"]
    if sp:
        i, o = sp["input_hw"], sp["output_hw"]
        gpu = (results["environment"].get("gpu") or "CPU").replace("NVIDIA GeForce ", "")
        prec = "fp16 autocast" if sp["precision"] == "fp16" else sp["precision"]
        b.append(
            f"- Measured {sp['model']['median_ms']:.1f} ms median latency "
            f"({sp['model']['fps']:.0f} FPS) for {i[1]}x{i[0]} -> {o[1]}x{o[0]} frames with "
            f"{prec} on an {gpu} (PyTorch eager, batch 1, GPU-synchronised timing; "
            f"{sp['end_to_end']['median_ms']:.1f} ms including host-device transfers)."
        )
    b.append(
        "- Built a reproducible benchmark (single protocol file, metrics verified against "
        "scikit-image, CPU CI) and replaced unverifiable README claims with its output, "
        "including speed figures that had been timed without GPU synchronisation."
    )
    return (warning(results) + "\n".join(b)).strip()


def render_markdown(results: dict) -> str:
    parts = [
        "# Benchmark results",
        "",
        warning(results) + provenance(results),
        "",
        "## Quality",
        "",
    ]
    for set_name in results["test_sets"]:
        parts += [quality_table(results, set_name), ""]
    parts += ["## Degradation sensitivity (test-data/)", "", ablation_table(results), ""]
    parts += ["## Speed (thermal model)", "", speed_table(results), ""]
    if results.get("skipped"):
        parts += (
            ["## Skipped", ""]
            + [f"- {s['method']} x{s['scale']}: {s['reason']}" for s in results["skipped"]]
            + [""]
        )
    parts += [
        "## Weights evaluated",
        "",
        "| Method | Scale | Params | state-dict SHA-256 |",
        "|:---|:---:|---:|:---|",
    ]
    for key, m in results["methods"].items():
        for s, info in m["per_scale"].items():
            if "params" in info:
                parts.append(
                    f"| {key} | x{s} | {info['params']:,} | "
                    f"`{info.get('state_dict_sha256', info.get('weights', '-'))}` |"
                )
    return "\n".join(parts) + "\n"


# Figures from the README on main before this benchmark (commit 5a23feb). They are
# quoted, never recomputed; see docs/claims-audit.md for where each came from.
OLD = {
    2: {"psnr": "34.2", "ssim": "0.840", "fps": "270.6 (table) / 229.6 (report)"},
    3: {"psnr": "31.0", "ssim": "0.757", "fps": "256.1"},
    4: {
        "psnr": "29.6",
        "ssim": "0.713",
        "fps": "250.9",
        "note": "No x4 evaluation output exists in the repository, so how this figure "
        "was produced is unknown.",
    },
}
OLD_BICUBIC_X2 = {"psnr": "24.2", "ssim": "0.612"}


def claims_table(results: dict) -> str:
    """Old README figures next to what the benchmark measures (repo17, protocol v1)."""
    if "repo17" not in results["test_sets"]:
        return "_repo17 not in this run._"
    lines = [
        "| Figure | Old README | Measured now | Why they differ |",
        "|:---|:---|:---|:---|",
    ]
    why_q = (
        "Old: 1,100 unpublished validation frames at 640x512 that also picked the "
        "checkpoint, no border crop, unquantised output. New: 17 committed frames, "
        "protocol v1.0. Not the same images, so not directly comparable."
    )
    why_s = (
        "Old timing used time.time() without torch.cuda.synchronize(), which measures "
        "kernel launches, not GPU work."
    )
    for s in _scales(results):
        t = _q(results, "repo17", s, "imdn_thermal", "fp32")
        b = _q(results, "repo17", s, "bicubic")
        if not (t and b):
            continue
        o = OLD[s]
        lines.append(
            f"| x{s} PSNR / SSIM | {o['psnr']} dB / {o['ssim']} | {t['psnr_mean']:.2f} dB / "
            f"{t['ssim_mean']:.4f} (bicubic {b['psnr_mean']:.2f} dB / {b['ssim_mean']:.4f}) | "
            f"{o.get('note', why_q)} |"
        )
        why_q = "As above."
    b2 = _q(results, "repo17", 2, "bicubic")
    t2 = _q(results, "repo17", 2, "imdn_thermal", "fp32")
    if b2 and t2:
        lines.append(
            f"| x2 bicubic baseline | {OLD_BICUBIC_X2['psnr']} dB / {OLD_BICUBIC_X2['ssim']} | "
            f"{b2['psnr_mean']:.2f} dB / {b2['ssim_mean']:.4f} | No code, data or citation "
            "for the old row. |"
        )
        lines.append(
            f"| x2 gain over bicubic | +10.0 dB (34.2 vs 24.2) | "
            f"{t2['psnr_mean'] - b2['psnr_mean']:+.2f} dB | Same images and protocol for "
            "both methods now. |"
        )
    for s in _scales(results):
        f32, f16 = _s(results, s, "fp32"), _s(results, s, "fp16")
        if not f32:
            continue
        i, o = f32["input_hw"], f32["output_hw"]
        now = f"fp32 {f32['model']['fps']:.0f} FPS"
        if f16:
            now += f", fp16 {f16['model']['fps']:.0f} FPS"
        lines.append(
            f"| x{s} speed | {OLD[s]['fps']} FPS | {now} ({i[1]}x{i[0]} -> {o[1]}x{o[0]}, "
            f"forward only) | {why_s} |"
        )
        why_s = "As above."
    return "\n".join(lines)


def changed(results: dict) -> str:
    """Short 'what changed' bullets for the README."""
    t2 = _q(results, "repo17", 2, "imdn_thermal", "fp32")
    b2 = _q(results, "repo17", 2, "bicubic")
    f32, f16 = _s(results, 2, "fp32"), _s(results, 2, "fp16")
    if not (t2 and b2):
        return "_No results._"
    out = [
        f"- **Quality.** On the 17 committed frames the x2 model is "
        f"{t2['psnr_mean'] - b2['psnr_mean']:+.2f} dB PSNR above bicubic "
        f"({t2['psnr_mean']:.2f} vs {b2['psnr_mean']:.2f} dB). The old table implied about "
        "+10 dB, against a bicubic figure (24.2 dB) that has no traceable source. The old "
        "34.2 dB came from a validation split that is not published and also selected the "
        "checkpoint; it is no longer quoted.",
    ]
    if f32:
        txt = (
            f"- **Speed.** Timed with GPU synchronisation, x2 (320x256 -> 640x512) runs at "
            f"{f32['model']['fps']:.0f} FPS in fp32"
        )
        if f16:
            txt += f" and {f16['model']['fps']:.0f} FPS with fp16 autocast"
        txt += (
            " on an RTX 3070 Laptop GPU, not 229-271 FPS: the old figures timed kernel "
            "launches, not GPU work."
        )
        out.append(txt)
    out.append(
        "- **Removed** as unsupported: ~~First thermal super-resolution system~~, "
        "~~New SOTA for thermal SR~~, ~~15x faster than competitors~~, ~~First real-time 4x "
        "thermal SR~~, ~~Breakthrough method~~, ~~novel IMDN architecture~~, the 40x/24x "
        "parameter-reduction claims and the ESRGAN comparison row (never run here). Full list: "
        "[docs/claims-audit.md](docs/claims-audit.md)."
    )
    return "\n".join(out)


BLOCKS = {
    "HEADLINE": headline,
    "CLAIMS": claims_table,
    "CHANGED": changed,
    "QUALITY": lambda r: "\n\n".join(quality_table(r, s) for s in r["test_sets"]),
    "SPEED": speed_table,
    "ABLATION": ablation_table,
    "PROVENANCE": provenance,
    "MEASUREMENTS": measurements,
    "CV": cv_bullets,
}


def replace_blocks(text: str, results: dict) -> str:
    for name, fn in BLOCKS.items():
        pat = re.compile(
            rf"(<!-- BENCHMARK:{name}:START -->\n)(.*?)(<!-- BENCHMARK:{name}:END -->)", re.S
        )
        text = pat.sub(lambda m, fn=fn: m.group(1) + fn(results).rstrip() + "\n" + m.group(3), text)
    return text


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m benchmark.report",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--results", type=Path, default=RESULTS)
    ap.add_argument("--check", action="store_true", help="fail if generated content is stale")
    args = ap.parse_args(argv)
    results = json.loads(args.results.read_text())
    if (results.get("quick") or results.get("random_weights")) and args.results == RESULTS:
        print("refusing: results/benchmark.json comes from a quick/random-weight run")
        return 1
    stale = []
    outputs = {args.results.with_suffix(".md"): render_markdown(results)}
    for path in TARGETS:
        if path.exists():
            outputs[path] = replace_blocks(path.read_text(), results)
    for path, new in outputs.items():
        old = path.read_text() if path.exists() else None
        if old != new:
            stale.append(path)
            if not args.check:
                path.write_text(new)
    rel = [str(p.relative_to(P.REPO_ROOT)) for p in stale]
    if args.check:
        print("stale: " + ", ".join(rel) if stale else "generated content is up to date")
        return 1 if stale else 0
    print("updated: " + (", ".join(rel) if rel else "nothing (already up to date)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
