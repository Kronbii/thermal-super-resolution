"""Export the fine-tuned weights as small, state-dict-only files for publishing.

    python -m benchmark.export_weights      # -> dist/weights/imdn_thermal_x{2,3,4}.pth

The training checkpoints also hold optimiser state (8.4 MB each); the exports
hold only the float32 tensors (about 2.8 MB each). Their state-dict SHA-256
equals the one recorded in results/benchmark.json, so a downloaded copy can be
checked against the benchmark. dist/ is git-ignored: attach the files to a
release instead of committing them. Put downloaded files in checkpoints/ and
the benchmark finds them by these names.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from . import protocol as P
from .data import sha256_file
from .models import load_state_dict, state_dict_digest

OUT = P.REPO_ROOT / "dist" / "weights"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m benchmark.export_weights", description=__doc__)
    ap.add_argument("--weights-dir", type=Path, default=P.WEIGHTS_DIR)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    lines = []
    for s in P.SCALES:
        sd = load_state_dict(args.weights_dir / P.THERMAL_WEIGHTS[s])
        dst = args.out / P.EXPORTED_WEIGHTS.format(s=s)
        torch.save(sd, dst)
        if state_dict_digest(load_state_dict(dst)) != state_dict_digest(sd):
            raise RuntimeError(f"round trip changed the weights for x{s}")
        lines.append(f"{sha256_file(dst)}  {dst.name}")
        print(f"x{s}: {dst} state_dict_sha256={state_dict_digest(sd)}")
    (args.out / "SHA256SUMS").write_text("\n".join(lines) + "\n")
    print(f"wrote {args.out / 'SHA256SUMS'}")


if __name__ == "__main__":
    main()
