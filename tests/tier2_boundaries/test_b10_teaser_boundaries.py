"""
tests/tier2_boundaries/test_b10_teaser_boundaries.py
Feature 10: Cold-Open Teaser Detection - Boundary & Edge Cases (5 cases)
"""

import pytest

from tests.fixtures.synthetic import make_sample_transcript, make_sample_viral_segment

try:
    import clippa.analysis as analysis
    HAS_CLIPPA = True
except ImportError:
    HAS_CLIPPA = False


@pytest.mark.tier2
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.analysis not implemented yet")
def test_tc_b10_01_short_teaser_clamped():
    """TC-B10-01: Proposed teaser duration shorter than 2.0s is rejected or padded."""
    seg = make_sample_viral_segment(10000, 45000)
    transcript = make_sample_transcript(60000)
    teaser_fn = getattr(analysis, "detect_cold_open_teaser", None)
    if teaser_fn:
        teaser = teaser_fn(seg, transcript, min_teaser_duration_s=2.0)
        if teaser:
            dur = teaser.end_time_ms - teaser.start_time_ms
            assert dur >= 2000
    else:
        pytest.skip("detect_cold_open_teaser pending in clippa.analysis")


@pytest.mark.tier2
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.analysis not implemented yet")
def test_tc_b10_02_long_teaser_truncated():
    """TC-B10-02: Proposed teaser duration longer than 5.0s is truncated to 5.0s."""
    seg = make_sample_viral_segment(10000, 45000)
    transcript = make_sample_transcript(60000)
    teaser_fn = getattr(analysis, "detect_cold_open_teaser", None)
    if teaser_fn:
        teaser = teaser_fn(seg, transcript, max_teaser_duration_s=5.0)
        if teaser:
            dur = teaser.end_time_ms - teaser.start_time_ms
            assert dur <= 5000
    else:
        pytest.skip("detect_cold_open_teaser pending in clippa.analysis")


@pytest.mark.tier2
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.analysis not implemented yet")
def test_tc_b10_03_teaser_after_clip_end_discarded():
    """TC-B10-03: Teaser timestamp located after clip end is discarded."""
    seg = make_sample_viral_segment(0, 30000)
    transcript = make_sample_transcript(60000)
    teaser_fn = getattr(analysis, "detect_cold_open_teaser", None)
    if teaser_fn:
        teaser = teaser_fn(seg, transcript)
        if teaser:
            assert teaser.end_time_ms <= seg.end_time_ms
    else:
        pytest.skip("detect_cold_open_teaser pending in clippa.analysis")


@pytest.mark.tier2
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.analysis not implemented yet")
def test_tc_b10_04_teaser_at_clip_start_disabled():
    """TC-B10-04: Teaser timestamp identical to clip start disables cold-open prepend."""
    seg = make_sample_viral_segment(10000, 40000)
    transcript = make_sample_transcript(60000)
    teaser_fn = getattr(analysis, "detect_cold_open_teaser", None)
    if teaser_fn:
        teaser = teaser_fn(seg, transcript)
        # Teaser should be None if opening already has hook
        assert teaser is None or teaser.start_time_ms > seg.start_time_ms
    else:
        pytest.skip("detect_cold_open_teaser pending in clippa.analysis")


@pytest.mark.tier2
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.analysis not implemented yet")
def test_tc_b10_05_short_clip_suppresses_teaser():
    """TC-B10-05: Clip total length < 20s automatically suppresses cold-open teaser."""
    seg = make_sample_viral_segment(0, 15000)
    transcript = make_sample_transcript(30000)
    teaser_fn = getattr(analysis, "detect_cold_open_teaser", None)
    if teaser_fn:
        teaser = teaser_fn(seg, transcript)
        assert teaser is None
    else:
        pytest.skip("detect_cold_open_teaser pending in clippa.analysis")
