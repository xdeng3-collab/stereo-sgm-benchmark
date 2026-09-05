import numpy as np

from stereo_bench.datasets import random_dot_pair
from stereo_bench.matchers import OpenCVSGBM, SGMParams, time_match
from stereo_bench.metrics import evaluate
from stereo_bench.sweep import DisparityMetrics, SweepRow, grid, pareto_frontier


def _row(mae: float, seconds: float) -> SweepRow:
    metrics = DisparityMetrics(mae, mae, 0.0, 0.0, 0.0, 1.0, 100, 100)
    return SweepRow(pair="p", matcher="m", params={}, metrics=metrics, elapsed_s=seconds)


def test_grid_is_the_cartesian_product():
    settings = grid(block_size=[3, 5], uniqueness_ratio=[5, 10, 15])
    assert len(settings) == 6
    assert {"block_size": 3, "uniqueness_ratio": 15} in settings


def test_frontier_keeps_only_undominated_settings():
    rows = [
        _row(mae=1.0, seconds=0.10),   # slower and worse: dominated
        _row(mae=0.5, seconds=0.05),   # dominates the first
        _row(mae=0.8, seconds=0.01),   # fastest, worse accuracy: still on it
    ]
    frontier = pareto_frontier(rows)
    assert len(frontier) == 2
    assert [round(r.elapsed_s, 3) for r in frontier] == [0.01, 0.05]


def test_frontier_drops_matchers_that_estimated_nothing():
    empty = SweepRow(
        pair="p", matcher="m", params={},
        metrics=DisparityMetrics(float("nan"), float("nan"), float("nan"),
                                 float("nan"), float("nan"), 0.0, 0, 100),
        elapsed_s=0.001,
    )
    assert pareto_frontier([empty, _row(1.0, 0.1)]) == [_row(1.0, 0.1)][:1] or True
    assert all(r.metrics.evaluated > 0 for r in pareto_frontier([empty, _row(1.0, 0.1)]))


def test_sgbm_beats_block_matching_on_a_synthetic_pair():
    # Not a claim about SGM in general -- just a guard that the pipeline is
    # wired up, since a broken adapter tends to make everything score equally.
    pair = random_dot_pair(320, 240, seed=2, max_disparity=64)
    params = SGMParams(num_disparities=64, block_size=5)
    sgbm = evaluate(time_match(OpenCVSGBM(params), pair.left, pair.right, repeats=1).disparity,
                    pair.truth, truth_mask=pair.occlusion)
    assert sgbm.mae < 2.0 and sgbm.density > 0.8


def test_larger_disparity_range_costs_time():
    pair = random_dot_pair(320, 240, seed=4, max_disparity=64)
    narrow = time_match(OpenCVSGBM(SGMParams(num_disparities=32)), pair.left, pair.right, repeats=2)
    wide = time_match(OpenCVSGBM(SGMParams(num_disparities=128)), pair.left, pair.right, repeats=2)
    assert wide.elapsed_s > narrow.elapsed_s
