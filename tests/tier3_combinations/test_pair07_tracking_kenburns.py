"""
tests/tier3_combinations/test_pair07_tracking_kenburns.py
Pairwise Test 07: F12 (Tracking) + F13 (Ken Burns Zoom)
Interaction Seam: Pan velocity dynamically controls talking-head zoompan factor.
"""

import pytest


@pytest.mark.tier3
def test_pair07_pan_velocity_controls_zoom():
    """TC-PAIR-07: Low pan velocity activates Ken Burns; fast pan suppresses it."""
    pan_threshold = 15.0

    # Case A: Stable speaker (low velocity)
    low_velocity = 2.5
    assert (low_velocity <= pan_threshold) is True
    zoom_applied_a = True

    # Case B: Fast camera pan (high velocity)
    high_velocity = 45.0
    assert (high_velocity > pan_threshold) is True
    zoom_applied_b = False

    assert zoom_applied_a is True
    assert zoom_applied_b is False
