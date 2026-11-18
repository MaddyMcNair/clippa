"""
tests/tier3_combinations/test_pair05_virality_teaser.py
Pairwise Test 05: F06 (Virality Detection) + F10 (Cold-Open Teaser)
Interaction Seam: Teaser timestamps are contained within parent clip interval.
"""

import pytest

from tests.fixtures.synthetic import make_sample_viral_segment


@pytest.mark.tier3
def test_pair05_teaser_nested_in_segment():
    """TC-PAIR-05: Teaser timestamps are strictly contained within parent viral segment."""
    seg = make_sample_viral_segment(15000, 48000)
    seg.hook_start_time_ms = 35000
    seg.hook_end_time_ms = 38500

    assert seg.start_time_ms <= seg.hook_start_time_ms
    assert seg.hook_end_time_ms <= seg.end_time_ms
    teaser_duration = (seg.hook_end_time_ms - seg.hook_start_time_ms) / 1000.0
    assert 2.0 <= teaser_duration <= 5.0
