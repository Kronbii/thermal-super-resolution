"""Test sets, LR synthesis and the optional TNO download."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from . import protocol as P

MANIFEST_DIR = Path(__file__).resolve().parent / "manifests"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_gray_u8(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"cannot read image: {path}")
    return img


def modcrop(img: np.ndarray, scale: int) -> np.ndarray:
    h, w = img.shape[:2]
    return img[: h - h % scale, : w - w % scale]


def degrade(hr: np.ndarray, scale: int, kind: str = P.PRIMARY_DEGRADATION) -> np.ndarray:
    """LR synthesis. The primary kind is identical to the training data preparation."""
    h, w = hr.shape[:2]
    if h % scale or w % scale:
        raise ValueError("modcrop the HR image before degrading it")
    if kind == "cv2_bicubic":
        return cv2.resize(hr, (w // scale, h // scale), interpolation=cv2.INTER_CUBIC)
    if kind == "pil_bicubic_antialias":
        from PIL import Image as PILImage

        return np.asarray(PILImage.fromarray(hr).resize((w // scale, h // scale), PILImage.BICUBIC))
    raise ValueError(f"unknown degradation {kind!r}")


@dataclass(frozen=True)
class Image:
    name: str
    path: Path
    sha256: str


def load_manifest(name: str) -> dict:
    with open(MANIFEST_DIR / f"{name.replace('-', '_')}.json") as f:
        return json.load(f)


def list_images(set_name: str) -> list[Image]:
    """Images of a test set, sorted by name, with integrity checks."""
    if set_name in (P.REPO17.name, P.TNO.name):
        manifest = load_manifest(set_name)
        root = P.REPO_ROOT / "test-data" if set_name == P.REPO17.name else P.TNO.root
        if not root.exists():
            raise FileNotFoundError(
                f"{set_name} is not downloaded; run: python -m benchmark.data --fetch {set_name}"
            )
        out = []
        for entry in manifest["files"]:
            path = root / entry["path"]
            if not path.exists():
                raise FileNotFoundError(f"{set_name}: missing {path}")
            digest = sha256_file(path)
            if digest != entry["sha256"]:
                raise RuntimeError(f"{set_name}: checksum mismatch for {path}")
            out.append(Image(entry["path"], path, digest))
        return out
    if set_name == P.FLIR_VIDEO_TEST.name:
        root = os.environ.get("FLIR_VIDEO_TEST_DIR")
        if not root or not Path(root).is_dir():
            raise FileNotFoundError(
                "flir-video-test needs --flir-video-test-dir (or FLIR_VIDEO_TEST_DIR) pointing "
                "at FLIR_ADAS_v2/video_thermal_test/data"
            )
        files = sorted(p for p in Path(root).iterdir() if p.suffix.lower() in {".jpg", ".png"})
        if not files:
            raise FileNotFoundError(f"no .jpg/.png frames in {root}")
        return [Image(f.name, f, sha256_file(f)) for f in files]
    raise KeyError(f"unknown test set {set_name!r}; known: {sorted(P.TEST_SETS)}")


def _download(url: str, dest: Path, size: int, retries: int = 8) -> None:
    """Resumable download with retries (the network may be slow or flaky)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, retries + 1):
        have = part.stat().st_size if part.exists() else 0
        if have >= size:
            break
        req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"})
        try:
            with urllib.request.urlopen(req, timeout=120) as r, open(part, "ab") as f:
                if have and r.status != 206:  # server ignored the range: restart
                    f.truncate(0)
                shutil.copyfileobj(r, f, 1 << 20)
        except OSError as e:  # URLError, timeouts, resets
            print(f"download attempt {attempt}/{retries} failed: {e}")
            time.sleep(min(60, 5 * attempt))
    if not part.exists() or part.stat().st_size != size:
        raise RuntimeError(f"download of {url} incomplete")
    part.rename(dest)


def fetch_tno() -> Path:
    d = P.TNO.download
    archive = P.DATA_CACHE / d["archive_name"]
    if not archive.exists():
        print(f"downloading {d['url']} ({d['size'] / 1e6:.0f} MB) -> {archive}")
        _download(d["url"], archive, d["size"])
    digest = sha256_file(archive)
    if digest != d["sha256"]:
        raise RuntimeError(f"{archive} sha256 {digest} != expected {d['sha256']}")
    if not P.TNO.root.exists():
        with zipfile.ZipFile(archive) as z:
            z.extractall(P.DATA_CACHE / d["extract_dir"])
    list_images(P.TNO.name)  # verifies every selected file
    print(f"tno-lwir ready: {P.TNO.root}")
    return P.TNO.root


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Fetch optional benchmark data.")
    ap.add_argument("--fetch", choices=[P.TNO.name], required=True)
    args = ap.parse_args()
    if args.fetch == P.TNO.name:
        fetch_tno()


if __name__ == "__main__":
    main()
