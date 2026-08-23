"""The synthetic pair is the ground truth for everything else, so it has to be
exactly right: if its disparity field and its right image disagree, every
matcher scored against it is scored against a lie."""

import numpy as np

from stereo_bench.datasets import random_dot_pair


def test_right_image_matches_the_declared_disparity():
    pair = random_dot_pair(128, 96, seed=7, max_disparity=32)
    height, width = pair.shape
    checked = 0
    for y in range(height):
        for x in range(width):
            d = pair.truth[y, x]
            if not np.isfinite(d):
                continue
            shifted = x - int(round(float(d)))
            if shifted < 0:
                continue
            assert pair.right[y, shifted] == pair.left[y, x]
            checked += 1
    assert checked > 0.5 * height * width


def test_occluded_pixels_are_marked_invalid():
    pair = random_dot_pair(128, 96, seed=3, max_disparity=32)
    # The truth mask and the visibility mask must agree exactly: a pixel has
    # ground truth if and only if the right camera can see it.
    assert np.all(np.isfinite(pair.truth) == pair.occlusion)
    assert (~pair.occlusion).sum() > 0  # the leftmost columns are always occluded


def test_occlusion_includes_pixels_hidden_behind_a_nearer_surface():
    # Disparity rising along a row compresses several left pixels into one right
    # column. Only the nearest survives; the rest are truly invisible.
    pair = random_dot_pair(256, 192, seed=9, max_disparity=48)
    height, width = pair.shape
    hidden_beyond_the_edge = 0
    for y in range(height):
        for x in range(width):
            if pair.occlusion[y, x]:
                continue
            d = np.rint(float(np.nan_to_num(pair.truth[y, x]))).astype(int)
            if x - d >= 0:
                hidden_beyond_the_edge += 1
    assert hidden_beyond_the_edge > 0


def test_disparity_stays_inside_the_declared_range():
    pair = random_dot_pair(160, 120, seed=11, max_disparity=48)
    finite = pair.truth[np.isfinite(pair.truth)]
    assert finite.min() >= 0 and finite.max() < 48


def test_seed_is_the_only_source_of_randomness():
    a = random_dot_pair(64, 64, seed=5)
    b = random_dot_pair(64, 64, seed=5)
    assert np.array_equal(a.left, b.left) and np.array_equal(a.right, b.right)
    assert not np.array_equal(a.left, random_dot_pair(64, 64, seed=6).left)
