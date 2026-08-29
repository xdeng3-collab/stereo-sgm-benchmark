"""Matcher adapters.

Every matcher exposes the same call signature and returns disparity in pixels
with NaN where it declined to estimate. Normalising that here is what makes the
comparison honest: OpenCV returns fixed-point disparity with its own invalid
sentinel, libSGM returns an integer map with a different one, and comparing raw
outputs would be comparing conventions rather than algorithms.

The libSGM adapter is defined but not importable without CUDA. It is here so the
parameter names line up with the OpenCV path, which is what lets a sweep run on
a laptop and then re-run unchanged on a GPU box.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from .metrics import INVALID


@dataclass(frozen=True)
class SGMParams:
    """Semi-global matching parameters, named as in Hirschmuller 2008.

    ``p1`` and ``p2`` are the smoothness penalties for disparity changes of one
    step and more than one step. Their ratio is what trades edge sharpness
    against streaking, and it is the first thing to sweep.
    """

    num_disparities: int = 64     # must be a multiple of 16 for OpenCV
    block_size: int = 5           # census / SAD window, odd
    p1: int | None = None         # default: 8 * block_size^2
    p2: int | None = None         # default: 32 * block_size^2
    uniqueness_ratio: int = 10    # margin the best match must win by, percent
    speckle_window: int = 100     # region size below which blobs are removed
    speckle_range: int = 2
    disp12_max_diff: int = 1      # left-right consistency tolerance, -1 disables
    pre_filter_cap: int = 31
    mode_hh: bool = False         # 8-path instead of OpenCV's default 5-path

    def resolved(self) -> tuple[int, int]:
        p1 = self.p1 if self.p1 is not None else 8 * self.block_size ** 2
        p2 = self.p2 if self.p2 is not None else 32 * self.block_size ** 2
        return p1, p2


@dataclass
class MatchResult:
    disparity: np.ndarray
    elapsed_s: float
    matcher: str
    params: dict[str, Any] = field(default_factory=dict)


class Matcher(Protocol):
    name: str

    def __call__(self, left: np.ndarray, right: np.ndarray) -> np.ndarray: ...


class OpenCVSGBM:
    """OpenCV's semi-global block matcher.

    Not the same algorithm as libSGM -- OpenCV aggregates over 5 paths by
    default and uses a Birchfield-Tomasi cost, where libSGM uses census over 4
    or 8 paths. It is a stand-in that runs anywhere, not a reference
    implementation, and results should not be reported as "SGM says".
    """

    name = "opencv-sgbm"

    def __init__(self, params: SGMParams | None = None) -> None:
        import cv2

        self.params = params or SGMParams()
        p1, p2 = self.params.resolved()
        if self.params.num_disparities % 16 != 0:
            raise ValueError("OpenCV requires num_disparities to be a multiple of 16")

        self._matcher = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=self.params.num_disparities,
            blockSize=self.params.block_size,
            P1=p1,
            P2=p2,
            disp12MaxDiff=self.params.disp12_max_diff,
            uniquenessRatio=self.params.uniqueness_ratio,
            speckleWindowSize=self.params.speckle_window,
            speckleRange=self.params.speckle_range,
            preFilterCap=self.params.pre_filter_cap,
            mode=cv2.STEREO_SGBM_MODE_HH if self.params.mode_hh else cv2.STEREO_SGBM_MODE_SGBM,
        )

    def __call__(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        raw = self._matcher.compute(left, right)
        # OpenCV returns disparity scaled by 16 in int16, with negative values
        # marking pixels it refused to estimate.
        disparity = raw.astype(np.float32) / 16.0
        disparity[raw < 0] = INVALID
        return disparity


class OpenCVBM:
    """Plain block matching. Present as the floor, not as a contender."""

    name = "opencv-bm"

    def __init__(self, params: SGMParams | None = None) -> None:
        import cv2

        self.params = params or SGMParams()
        self._matcher = cv2.StereoBM_create(
            numDisparities=self.params.num_disparities,
            blockSize=max(5, self.params.block_size | 1),
        )

    def __call__(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        raw = self._matcher.compute(left, right)
        disparity = raw.astype(np.float32) / 16.0
        disparity[raw < 0] = INVALID
        return disparity


class LibSGM:
    """Adapter for FiXSTARS libSGM (CUDA).

    Requires a build of libSGM with Python bindings and an NVIDIA GPU. The point
    of keeping the signature identical to OpenCVSGBM is that a sweep authored on
    a laptop re-runs unchanged on a GPU machine.
    """

    name = "libsgm"

    def __init__(self, params: SGMParams | None = None) -> None:
        try:
            import libsgm  # type: ignore  # noqa: F401
        except ImportError as exc:  # pragma: no cover - requires CUDA
            raise ImportError(
                "libSGM is not available. It needs a CUDA device and a build of "
                "https://github.com/fixstars/libSGM with Python bindings. Note "
                "that the upstream CMake pins the CUDA architecture; set it to "
                "your device's compute capability (T4 = sm_75, TX2 = sm_62)."
            ) from exc
        self.params = params or SGMParams()
        raise NotImplementedError(
            "Wire this to the libSGM binding once a GPU is available; the "
            "parameter mapping is p1/p2 -> P1/P2, uniqueness_ratio -> "
            "uniqueness, disp12_max_diff -> LR check."
        )


def time_match(matcher: Matcher, left: np.ndarray, right: np.ndarray, *, repeats: int = 3) -> MatchResult:
    """Run a matcher and report the best of ``repeats`` wall-clock timings.

    Best-of rather than mean: this is measuring the matcher, and the slow runs
    are measuring whatever else the machine was doing.
    """
    if repeats < 1:
        raise ValueError("repeats must be at least 1")

    matcher(left, right)  # warm caches and any lazy initialisation
    best = float("inf")
    disparity = None
    for _ in range(repeats):
        started = time.perf_counter()
        disparity = matcher(left, right)
        best = min(best, time.perf_counter() - started)

    params = getattr(matcher, "params", None)
    return MatchResult(
        disparity=disparity,
        elapsed_s=best,
        matcher=matcher.name,
        params=params.__dict__.copy() if params is not None else {},
    )
