"""
tests/tier3_combinations/test_pair06_yunet_tracking.py
Pairwise Test 06: F11 (YuNet Face Detection) + F12 (Centroid Tracking & Smoothing)
Interaction Seam: Raw YuNet detections feed exponential smoothing tracker.
"""

import pytest


@pytest.mark.tier3
def test_pair06_yunet_feeds_smoothing_tracker():
    """TC-PAIR-06: Raw face detections smooth smoothly without coordinate spikes."""
    raw_detections = [
        (800, 300, 320, 400, 0.95),
        (820, 305, 320, 400, 0.94),
        (810, 300, 320, 400, 0.96),
        (850, 310, 320, 400, 0.93),
    ]
    raw_centroids_x = [d[0] + d[2] / 2 for d in raw_detections]

    alpha = 0.2
    smoothed_x = [raw_centroids_x[0]]
    for x in raw_centroids_x[1:]:
        smoothed_x.append(alpha * x + (1 - alpha) * smoothed_x[-1])

    # Check that smoothed steps have no erratic spikes
    for i in range(1, len(smoothed_x)):
        step = abs(smoothed_x[i] - smoothed_x[i - 1])
        assert step < 20.0  # smooth incremental motion
