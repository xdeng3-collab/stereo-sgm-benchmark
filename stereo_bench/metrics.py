"""Disparity error metrics.

Every metric here is masked twice: once by the ground truth's own validity mask,
and once by whether the matcher produced an estimate at that pixel at all. A
stereo matcher that declines to guess in a textureless region is behaving
correctly, and averaging its refusals in as zero-error or as huge-error both
misreport what happened. Density is reported alongside accuracy for the same
reason: the two trade against each other, and either one alone can be gamed.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np

# Disparity maps use NaN for "no estimate". Sentinel values like 0 or -1 collide
# with legitimate disparities at the far plane.
INVALID = np.nan


@dataclass(frozen=True)
class DisparityMetrics:
    """Accuracy of one disparity map against ground truth."""

    mae: float           # mean absolute error over jointly valid pixels, px
    rmse: float          # root mean squared error, px
    bad_0_5: float       # fraction with |error| > 0.5 px
    bad_1_0: float       # fraction with |error| > 1.0 px
    bad_3_0: float       # fraction with |error| > 3.0 px
    density: float       # fraction of GT-valid pixels the matcher estimated
    evaluated: int       # pixel count the metrics are computed over
    gt_valid: int        # pixel count with valid ground truth

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def valid_mask(disparity: np.ndarray) -> np.ndarray:
    """Pixels where a matcher actually produced an estimate."""
    return np.isfinite(disparity)


def evaluate(
    estimate: np.ndarray,
    truth: np.ndarray,
    *,
    truth_mask: np.ndarray | None = None,
) -> DisparityMetrics:
    """Compare an estimated disparity map against ground truth.

    ``truth_mask`` marks where ground truth is meaningful. Occluded pixels have
    no correct disparity, so scoring them would penalise a matcher for something
    no matcher can get right.
    """
    if estimate.shape != truth.shape:
        raise ValueError(f"shape mismatch: estimate {estimate.shape} vs truth {truth.shape}")

    gt_valid = valid_mask(truth)
    if truth_mask is not None:
        if truth_mask.shape != truth.shape:
            raise ValueError("truth_mask must match the disparity shape")
        gt_valid &= truth_mask.astype(bool)

    gt_count = int(gt_valid.sum())
    if gt_count == 0:
        raise ValueError("ground truth has no valid pixels to evaluate against")

    both = gt_valid & valid_mask(estimate)
    evaluated = int(both.sum())
    density = evaluated / gt_count

    if evaluated == 0:
        # The matcher estimated nothing where truth exists. Report that plainly
        # rather than emitting a zero error that reads as a perfect score.
        return DisparityMetrics(
            mae=float("nan"), rmse=float("nan"),
            bad_0_5=float("nan"), bad_1_0=float("nan"), bad_3_0=float("nan"),
            density=0.0, evaluated=0, gt_valid=gt_count,
        )

    error = np.abs(estimate[both].astype(np.float64) - truth[both].astype(np.float64))
    return DisparityMetrics(
        mae=float(error.mean()),
        rmse=float(np.sqrt((error ** 2).mean())),
        bad_0_5=float((error > 0.5).mean()),
        bad_1_0=float((error > 1.0).mean()),
        bad_3_0=float((error > 3.0).mean()),
        density=float(density),
        evaluated=evaluated,
        gt_valid=gt_count,
    )


def disparity_to_depth(disparity: np.ndarray, focal_px: float, baseline_m: float) -> np.ndarray:
    """Z = f * B / d, with non-positive disparity mapped to infinity.

    Disparity of zero is the plane at infinity, not an error, and dividing by it
    should produce ``inf`` rather than a warning and a garbage value.
    """
    if focal_px <= 0 or baseline_m <= 0:
        raise ValueError("focal length and baseline must be positive")
    out = np.full(disparity.shape, np.inf, dtype=np.float64)
    usable = valid_mask(disparity) & (disparity > 0)
    out[usable] = focal_px * baseline_m / disparity[usable]
    out[~valid_mask(disparity)] = np.nan
    return out
