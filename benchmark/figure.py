"""Hero figure built only from results/benchmark.json.

    python -m benchmark.figure   # -> docs/figures/benchmark.png

The figure shows metrics, not thermal imagery: the test frames are street scenes
that may contain people, so no image crops are published.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import protocol as P  # noqa: E402
from .report import RESULTS, _q, _s, _scales  # noqa: E402

OUT = P.REPO_ROOT / "docs" / "figures" / "benchmark.png"

# Dark surface and ink from the author's site palette; the two series hues were
# checked with the dataviz palette validator (lightness band, chroma, CVD and
# normal-vision separation, contrast) against this surface.
SURFACE = "#121614"
INK = "#eef1ec"
INK_2 = "#b5bab3"  # ~74 % ink
INK_3 = "#868c86"  # ~52 % ink
HAIR = "#2a302c"
SERIES = ("#719f5d", "#c862b2")


def _style(ax, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(HAIR)
    ax.tick_params(colors=INK_3, length=0, labelsize=11)
    ax.yaxis.grid(True, color=HAIR, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=11)


def _bars(ax, groups, series, values, labels, fmt):
    n = len(series)
    width = 0.3
    for j, name in enumerate(series):
        xs = [i + (j - (n - 1) / 2) * (width + 0.025) for i in range(len(groups))]
        ys = [values[g][j] for g in groups]
        ax.bar(xs, ys, width=width, color=SERIES[j], label=name, linewidth=0)
        for x, y, lab in zip(xs, ys, [labels[g][j] for g in groups], strict=True):
            ax.text(
                x,
                y,
                fmt(lab),
                ha="center",
                va="bottom",
                color=INK,
                fontsize=10.5,
                fontweight="bold",
            )
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([f"x{g}" for g in groups], color=INK_2, fontsize=12)
    leg = ax.legend(loc="upper right", frameon=False, fontsize=10.5)
    for t in leg.get_texts():
        t.set_color(INK_2)


def render(results: dict, out: Path = OUT) -> Path:
    scales = _scales(results)
    fig, (a, b) = plt.subplots(1, 2, figsize=(12.8, 6.0), dpi=125)
    fig.patch.set_facecolor(SURFACE)

    sets = [s for s in ("repo17", "tno-lwir") if s in results["test_sets"]]
    names = {"repo17": "17 FLIR ADAS v2 frames", "tno-lwir": "51 TNO images (other cameras)"}
    gain = {}
    for s in scales:
        row = []
        for name in sets:
            t = _q(results, name, s, "imdn_thermal", "fp32")
            base = _q(results, name, s, "bicubic")
            row.append(t["psnr_mean"] - base["psnr_mean"])
        gain[s] = row
    _style(a, "PSNR gain over bicubic (dB)")
    _bars(a, scales, [names[s] for s in sets], gain, gain, lambda v: f"{v:+.2f}")
    a.set_ylim(0, max(max(v) for v in gain.values()) * 1.35)
    a.set_title(
        "Quality: fine-tuned IMDN vs bicubic",
        color=INK,
        fontsize=14,
        loc="left",
        fontweight="bold",
        pad=12,
    )

    precs = [p for p in ("fp32", "fp16") if _s(results, scales[0], p)]
    lat = {s: [_s(results, s, p)["model"]["median_ms"] for p in precs] for s in scales}
    fps = {s: [_s(results, s, p)["model"]["fps"] for p in precs] for s in scales}
    _style(b, "Forward latency, median (ms)")
    pretty = {"fp32": "fp32", "fp16": "fp16 autocast"}
    _bars(b, scales, [pretty[p] for p in precs], lat, fps, lambda v: f"{v:.0f} FPS")
    b.set_ylim(0, max(max(v) for v in lat.values()) * 1.3)
    b.set_title(
        "Speed: ~640x512 output, batch 1",
        color=INK,
        fontsize=14,
        loc="left",
        fontweight="bold",
        pad=12,
    )

    env = results["environment"]
    gpu = (env.get("gpu") or "CPU").replace("NVIDIA GeForce ", "")
    fig.suptitle(
        "Thermal super-resolution (IMDN), measured",
        x=0.012,
        ha="left",
        color=INK,
        fontsize=17,
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.012,
        0.885,
        f"Protocol v{results['protocol']['version']} (benchmark/protocol.py): OpenCV bicubic "
        "LR input, uint8 output, border crop = scale, mean of per-image PSNR.\n"
        f"Speed: {gpu}, PyTorch {env['torch'].split('+')[0]} eager, GPU-synchronised "
        f"timing, median over {_s(results, scales[0], precs[0]).get('trials', 1)} interleaved "
        "trials. Labels above latency bars: FPS = 1000 / median.",
        color=INK_3,
        fontsize=10,
        linespacing=1.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.86), w_pad=3)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m benchmark.figure", description=__doc__)
    ap.add_argument("--results", type=Path, default=RESULTS)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)
    print(render(json.loads(args.results.read_text()), args.out))


if __name__ == "__main__":
    main()
