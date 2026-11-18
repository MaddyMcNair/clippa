"""
tests/tier2_boundaries/test_b12_tracking_boundaries.py
Feature 12: Face Centroid Tracking & Smoothing - Boundary & Edge Cases (5 cases)
"""

import pytest

try:
    import clippa.reframing as reframing
    HAS_CLIPPA = True
except ImportError:
    HAS_CLIPPA = False


@pytest.mark.tier2
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.reframing not implemented yet")
def test_tc_b12_01_extreme_left_clamping():
    """TC-B12-01: Face hugging extreme left edge (x=0) clamps crop window to crop_x >= 0."""
    smooth_fn = getattr(reframing, "smooth_face_trajectory", getattr(reframing, "smooth_trajectory", None))
    result = smooth_fn([0, 0, 0], src_w=1920, crop_w=608)
    for crop_x in result:
        assert crop_x >= 0


@pytest.mark.tier2
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.reframing not implemented yet")
def test_tc_b12_02_extreme_right_clamping():
    """TC-B12-02: Face hugging extreme right edge (x=1920) clamps so crop_x + crop_w <= 1920."""
    smooth_fn = getattr(reframing, "smooth_face_trajectory", getattr(reframing, "smooth_trajectory", None))
    result = smooth_fn([1920, 1920, 1920], src_w=1920, crop_w=608)
    for crop_x in result:
        assert crop_x + 608 <= 1920


@pytest.mark.tier2
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.reframing not implemented yet")
def test_tc_b12_03_intermittent_disappearance_interpolation():
    """TC-B12-03: Intermittent face disappearance maintains smooth trajectory interpolation."""
    track = [900, None, None, None, 940]
    smooth_fn = getattr(reframing, "smooth_face_trajectory", getattr(reframing, "smooth_trajectory", None))
    result = smooth_fn(track, src_w=1920, crop_w=608)
    assert None not in result
    assert len(result) == len(track)


@pytest.mark.tier2
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.reframing not implemented yet")
def test_tc_b12_04_shot_cut_resets_smoothing():
    """TC-B12-04: Shot change cut instantly resets smoothing accumulator."""
    smooth_fn = getattr(reframing, "smooth_face_trajectory", getattr(reframing, "smooth_trajectory", None))
    # Massive coordinate jump from 200 to 1800
    track = [200, 200, 1800, 1800]
    result = smooth_fn(track, src_w=1920, crop_w=608)
    assert len(result) == 4
    # The jump should produce a significant shift on frame 2 without NaN
    assert result[2] > result[1]


@pytest.mark.tier2
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.reframing not implemented yet")
def test_tc_b12_05_zigzag_motion_damped():
    """TC-B12-05: Zig-zag motion is heavily damped by exponential moving average."""
    zigzag = [900, 1100, 900, 1100, 900, 1100]
    smooth_fn = getattr(reframing, "smooth_face_trajectory", getattr(reframing, "smooth_trajectory", None))
    smoothed = smooth_fn(zigzag, src_w=1920, crop_w=608, alpha=0.15)
    assert len(smoothed) == len(zigzag)
    # Variance of smoothed trajectory is lower than raw
    mean_raw = sum(zigzag) / len(zigzag)
    mean_smooth = sum(smoothed) / len(smoothed)
    var_raw = sum((x - mean_raw)**2 for x in zigzag)
    var_smooth = sum((x - mean_smooth)**2 for x in smoothed)
    assert var_smooth < var_raw
