"""Stereo disparity benchmarking: metrics, datasets, matcher adapters, sweeps."""

from .metrics import DisparityMetrics, evaluate, disparity_to_depth, INVALID
from .datasets import StereoPair, random_dot_pair, load_middlebury, read_pfm
from .matchers import SGMParams, OpenCVSGBM, OpenCVBM, LibSGM, time_match
from .sweep import grid, run_sweep, pareto_frontier, plot_frontier, write_csv

__all__ = [
    "DisparityMetrics", "evaluate", "disparity_to_depth", "INVALID",
    "StereoPair", "random_dot_pair", "load_middlebury", "read_pfm",
    "SGMParams", "OpenCVSGBM", "OpenCVBM", "LibSGM", "time_match",
    "grid", "run_sweep", "pareto_frontier", "plot_frontier", "write_csv",
]
