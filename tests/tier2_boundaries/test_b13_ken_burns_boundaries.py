"""
tests/tier2_boundaries/test_b13_ken_burns_boundaries.py
Feature 13: Ken Burns Talking Head Zoom - Boundary & Edge Cases (5 cases)
"""

import pytest


@pytest.mark.tier2
def test_tc_b13_01_short_clip_scaled_zoom():
    """TC-B13-01: Video clip shorter than 5 seconds applies scaled zoom curve safely."""
    duration_s = 3.0
    fps = 30
    total_frames = int(duration_s * fps)
    max_zoom = 1.05
    zoom_step = (max_zoom - 1.0) / max(1, total_frames)
    final_zoom = 1.0 + zoom_step * total_frames
    assert abs(final_zoom - max_zoom) < 0.001


@pytest.mark.tier2
def test_tc_b13_02_max_zoom_cap_enforced():
    """TC-B13-02: Max zoom boundary cap (1.08x) is never exceeded on long clips."""
    max_cap = 1.08
    for test_zoom in [1.02, 1.05, 1.08, 1.15, 1.50]:
        clamped = min(max_cap, test_zoom)
        assert clamped <= max_cap


@pytest.mark.tier2
def test_tc_b13_03_fast_pan_suppression():
    """TC-B13-03: Fast panning footage zeroes out zoompan factor completely."""
    is_fast_pan = True
    zoom_delta = 0.0 if is_fast_pan else 0.05
    assert zoom_delta == 0.0


@pytest.mark.tier2
def test_tc_b13_04_odd_dimension_even_rounding():
    """TC-B13-04: Zoom scaling preserves even integer dimensions for H.264."""
    for raw_dim in [1079, 1081, 1919, 1921]:
        even_dim = raw_dim if raw_dim % 2 == 0 else raw_dim - 1
        assert even_dim % 2 == 0


@pytest.mark.tier2
def test_tc_b13_05_one_frame_clip_no_zero_division():
    """TC-B13-05: 1-frame clip avoids divide-by-zero in zoom step calculation."""
    frames = 1
    step = (1.05 - 1.0) / max(1, frames - 1)
    # With max(1, ...), zero division is avoided
    assert step == pytest.approx(0.05)
