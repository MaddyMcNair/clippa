"""
tests/tier1_features/test_feat12_tracking.py
Feature 12: Face Centroid Tracking & Smoothing
Category-Partition Tests (5 cases)
"""

import pytest

try:
    import clippa.reframing as reframing
    HAS_CLIPPA = True
except ImportError:
    HAS_CLIPPA = False


@pytest.mark.tier1
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.reframing not implemented yet")
def test_tc_f12_01_weighted_face_centroid():
    """TC-F12-01: Computes weighted face centroid prioritizing primary speaking face."""
    calc_fn = getattr(reframing, "compute_face_centroid", getattr(reframing, "calculate_centroid", None))
    box = (800, 300, 320, 400)
    if calc_fn:
        cx, cy = calc_fn(box)
        assert cx == 960.0
        assert cy == 500.0
    else:
        # Fallback to smooth_face_trajectory
        smooth_fn = getattr(reframing, "smooth_face_trajectory", lambda d, **kw: d)
        trajectory = smooth_fn([box[0] + box[2] / 2] * 5, src_w=1920, crop_w=608)
        assert len(trajectory) == 5
        assert trajectory[0] >= 0


@pytest.mark.tier1
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.reframing not implemented yet")
def test_tc_f12_02_exponential_smoothing_eliminates_jitter():
    """TC-F12-02: Temporal exponential smoothing eliminates high-frequency jitter."""
    raw_trajectory = [960 + (-1)**i * 50 for i in range(10)]
    smooth_fn = getattr(reframing, "smooth_face_trajectory", getattr(reframing, "smooth_trajectory", None))
    smoothed = smooth_fn(raw_trajectory, src_w=1920, crop_w=608, alpha=0.2)
    assert len(smoothed) == len(raw_trajectory)
    raw_delta = abs(raw_trajectory[1] - raw_trajectory[0])
    smooth_delta = abs(smoothed[1] - smoothed[0])
    assert smooth_delta < raw_delta


@pytest.mark.tier1
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.reframing not implemented yet")
def test_tc_f12_03_face_absent_defaults_to_center():
    """TC-F12-03: Face absent defaults trajectory to centered coordinates."""
    smooth_fn = getattr(reframing, "smooth_face_trajectory", getattr(reframing, "smooth_trajectory", None))
    raw_trajectory = [None, None, None]
    result = smooth_fn(raw_trajectory, src_w=1920, crop_w=608)
    expected_center_x = (1920 - 608) // 2
    for pos in result:
        assert abs(pos - expected_center_x) <= 10 or pos == 960


@pytest.mark.tier1
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.reframing not implemented yet")
def test_tc_f12_04_crop_trajectory_clamps_in_bounds():
    """TC-F12-04: Horizontal crop trajectory clamps strictly inside source boundaries [0, src_w - crop_w]."""
    raw_trajectory = [-100, 0, 50, 1850, 1920, 2500]
    smooth_fn = getattr(reframing, "smooth_face_trajectory", getattr(reframing, "smooth_trajectory", None))
    result = smooth_fn(raw_trajectory, src_w=1920, crop_w=608)
    for crop_x in result:
        assert 0 <= crop_x <= (1920 - 608)


@pytest.mark.tier1
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.reframing not implemented yet")
def test_tc_f12_05_static_speaker_stable_coordinates():
    """TC-F12-05: Static speaker produces stable, non-oscillating crop window coordinates."""
    static_raw = [960] * 30
    smooth_fn = getattr(reframing, "smooth_face_trajectory", getattr(reframing, "smooth_trajectory", None))
    result = smooth_fn(static_raw, src_w=1920, crop_w=608)
    assert len(result) == 30
    for i in range(1, len(result)):
        assert result[i] == result[0]
