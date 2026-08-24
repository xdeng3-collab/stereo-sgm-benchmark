"""The metrics are the thing every number in this repo rests on, so they get the
most testing. A quiet bug here silently rewrites every result."""

import numpy as np
import pytest

from stereo_bench.metrics import INVALID, disparity_to_depth, evaluate


def test_identical_maps_score_perfectly():
    truth = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    m = evaluate(truth.copy(), truth)
    assert m.mae == 0.0 and m.rmse == 0.0
    assert m.bad_1_0 == 0.0 and m.density == 1.0


def test_pixels_the_matcher_declined_lower_density_not_accuracy():
    truth = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    estimate = np.array([[1.0, INVALID, 3.0, INVALID]], dtype=np.float32)
    m = evaluate(estimate, truth)
    # The two estimated pixels are exact; the refusals must not be scored as
    # zero error, which would be flattery, nor as huge error, which would be
    # punishment for correct behaviour.
    assert m.mae == 0.0
    assert m.density == 0.5
    assert m.evaluated == 2 and m.gt_valid == 4


def test_occluded_pixels_are_excluded_from_scoring():
    truth = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    estimate = np.array([[1.0, 2.0, 3.0, 99.0]], dtype=np.float32)
    visible = np.array([[True, True, True, False]])
    assert evaluate(estimate, truth, truth_mask=visible).mae == 0.0
    assert evaluate(estimate, truth).mae > 20.0  # without the mask, it is punished


def test_unknown_ground_truth_is_not_scored():
    truth = np.array([[1.0, INVALID, 3.0]], dtype=np.float32)
    estimate = np.array([[1.0, 500.0, 3.0]], dtype=np.float32)
    m = evaluate(estimate, truth)
    assert m.mae == 0.0 and m.gt_valid == 2


@pytest.mark.parametrize("threshold,field", [(0.5, "bad_0_5"), (1.0, "bad_1_0"), (3.0, "bad_3_0")])
def test_error_exactly_on_a_threshold_is_not_bad(threshold, field):
    # "bad" means strictly worse than the threshold. An error of exactly 1.0 px
    # counting as bad-1.0 would make every published comparison off by whatever
    # sits on the boundary.
    truth = np.zeros((1, 1), dtype=np.float32)
    assert getattr(evaluate(np.array([[threshold]], dtype=np.float32), truth), field) == 0.0
    just_over = np.array([[threshold + 1e-3]], dtype=np.float32)
    assert getattr(evaluate(just_over, truth), field) == 1.0


def test_matcher_that_estimates_nothing_reports_nan_not_zero():
    truth = np.array([[1.0, 2.0]], dtype=np.float32)
    m = evaluate(np.full((1, 2), INVALID, dtype=np.float32), truth)
    assert m.density == 0.0
    assert np.isnan(m.mae)  # a zero here would read as a perfect score


def test_shape_and_empty_truth_are_errors():
    with pytest.raises(ValueError):
        evaluate(np.zeros((2, 2)), np.zeros((3, 3)))
    with pytest.raises(ValueError):
        evaluate(np.zeros((1, 2)), np.full((1, 2), INVALID))


def test_depth_conversion_sends_zero_disparity_to_infinity():
    disparity = np.array([[10.0, 0.0, INVALID]], dtype=np.float32)
    depth = disparity_to_depth(disparity, focal_px=700.0, baseline_m=0.1)
    assert depth[0, 0] == pytest.approx(7.0)
    assert np.isinf(depth[0, 1])   # the plane at infinity, not an error
    assert np.isnan(depth[0, 2])   # no estimate stays no estimate
