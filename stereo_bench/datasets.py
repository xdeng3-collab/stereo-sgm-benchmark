"""Stereo pairs to evaluate against.

Two sources. The synthetic random-dot stereogram exists so the whole pipeline
runs on a fresh clone with no downloads: its ground truth is exact by
construction, because the right image is built by warping the left one by a
disparity field that is chosen first. The Middlebury loader is for real data,
where ground truth is measured and therefore has gaps.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .metrics import INVALID


@dataclass(frozen=True)
class StereoPair:
    name: str
    left: np.ndarray                  # uint8, HxW
    right: np.ndarray                 # uint8, HxW
    truth: np.ndarray | None = None   # float32 disparity, NaN where unknown
    occlusion: np.ndarray | None = None  # True where the pixel is visible in both
    focal_px: float | None = None
    baseline_m: float | None = None

    @property
    def shape(self) -> tuple[int, int]:
        return self.left.shape


def random_dot_pair(
    width: int = 640,
    height: int = 480,
    *,
    seed: int = 0,
    max_disparity: int = 48,
    texture_scale: int = 2,
    blank_fraction: float = 0.0,
) -> StereoPair:
    """A synthetic pair with exactly known disparity.

    The scene is three fronto-parallel slabs plus a smooth ramp, which is enough
    structure to expose the two failure modes that matter: discontinuities at
    slab edges, and gradient bias on the ramp.

    ``blank_fraction`` paints a textureless rectangle. Low-texture regions are
    where stereo genuinely has no signal, and a benchmark that never includes
    one will overstate every matcher it tests.
    """
    rng = np.random.default_rng(seed)

    # Coarse noise upsampled to pixels: single-pixel noise is unrealistically
    # easy to match, since every window is unique.
    coarse = rng.integers(0, 256, size=(height // texture_scale + 1, width // texture_scale + 1), dtype=np.uint8)
    left = np.repeat(np.repeat(coarse, texture_scale, axis=0), texture_scale, axis=1)[:height, :width]

    truth = np.zeros((height, width), dtype=np.float32)
    thirds = height // 3
    truth[:thirds, :] = max_disparity * 0.25
    truth[thirds:2 * thirds, :] = max_disparity * 0.60
    truth[2 * thirds:, :] = max_disparity * 0.85
    ramp = np.linspace(0, max_disparity * 0.15, width, dtype=np.float32)
    truth += ramp[None, :]
    truth = np.clip(truth, 0, max_disparity - 1)

    if blank_fraction > 0:
        bh = int(height * blank_fraction)
        bw = int(width * blank_fraction)
        y0, x0 = height // 2 - bh // 2, width // 2 - bw // 2
        left[y0:y0 + bh, x0:x0 + bw] = 128

    # Right image: the pixel at (y, x - d) in the right view corresponds to
    # (y, x) in the left. Building it this way makes the disparity exact.
    #
    # Forward warping also produces real occlusion. Where disparity increases
    # along a row, several left pixels compete for one right column: the nearer
    # surface wins and the farther one is genuinely hidden in the right view.
    # Those pixels get no valid ground truth, because no stereo matcher can
    # recover a pixel that the second camera cannot see. Silently keeping them
    # would penalise every matcher for the one thing they cannot do.
    right = np.zeros_like(left)
    visible = np.zeros((height, width), dtype=bool)
    cols = np.arange(width)
    for y in range(height):
        shifted = cols - np.rint(truth[y]).astype(np.int32)
        inside = shifted >= 0

        # Ascending x means larger disparity writes later, so last-write-wins
        # already resolves in favour of the nearer surface. A pixel is visible
        # only if it is the final writer of its target column.
        winner = np.full(width, -1, dtype=np.int32)
        winner[shifted[inside]] = cols[inside]
        right[y, shifted[inside]] = left[y, cols[inside]]
        visible[y, winner[winner >= 0]] = True

    # Occluded pixels -- off the left edge, or hidden behind a nearer surface --
    # are excluded from scoring.
    truth_masked = truth.astype(np.float32).copy()
    truth_masked[~visible] = INVALID

    return StereoPair(
        name=f"random-dot-{width}x{height}-d{max_disparity}-s{seed}",
        left=left,
        right=right,
        truth=truth_masked,
        occlusion=visible,
        focal_px=700.0,
        baseline_m=0.12,
    )


def read_pfm(path: Path) -> np.ndarray:
    """Middlebury ground truth is distributed as PFM. Infinity means unknown."""
    with open(path, "rb") as handle:
        header = handle.readline().decode("ascii").rstrip()
        if header not in ("Pf", "PF"):
            raise ValueError(f"{path} is not a PFM file")
        channels = 3 if header == "PF" else 1

        line = handle.readline().decode("ascii")
        match = re.match(r"^(\d+)\s+(\d+)\s*$", line)
        if not match:
            raise ValueError(f"malformed PFM dimensions in {path}: {line!r}")
        width, height = int(match.group(1)), int(match.group(2))

        scale = float(handle.readline().decode("ascii").rstrip())
        endian = "<" if scale < 0 else ">"

        data = np.frombuffer(handle.read(), dtype=endian + "f4")
        data = data.reshape(height, width, channels) if channels == 3 else data.reshape(height, width)

    data = np.flipud(data).astype(np.float32)      # PFM rows run bottom-to-top
    data[np.isinf(data)] = INVALID
    return data


def load_middlebury(directory: Path) -> StereoPair:
    """Load a Middlebury 2014 scene directory (im0.png, im1.png, disp0.pfm)."""
    import cv2

    directory = Path(directory)
    left = cv2.imread(str(directory / "im0.png"), cv2.IMREAD_GRAYSCALE)
    right = cv2.imread(str(directory / "im1.png"), cv2.IMREAD_GRAYSCALE)
    if left is None or right is None:
        raise FileNotFoundError(f"expected im0.png and im1.png in {directory}")

    truth_path = directory / "disp0.pfm"
    truth = read_pfm(truth_path) if truth_path.exists() else None

    focal = baseline = None
    calib = directory / "calib.txt"
    if calib.exists():
        values = dict(
            line.split("=", 1) for line in calib.read_text().splitlines() if "=" in line
        )
        if "cam0" in values:
            focal = float(values["cam0"].strip("[] ").split()[0])
        if "baseline" in values:
            baseline = float(values["baseline"]) / 1000.0  # mm to m

    return StereoPair(
        name=directory.name, left=left, right=right, truth=truth,
        focal_px=focal, baseline_m=baseline,
    )
