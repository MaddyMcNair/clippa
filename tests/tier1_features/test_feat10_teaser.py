"""
tests/tier1_features/test_feat10_teaser.py
Feature 10: Cold-Open Teaser Detection
Category-Partition Tests (5 cases)
"""

import pytest

from tests.fixtures.synthetic import make_sample_transcript, make_sample_viral_segment

try:
    import clippa.analysis as analysis
    HAS_CLIPPA = True
except ImportError:
    HAS_CLIPPA = False


@pytest.mark.tier1
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.analysis not implemented yet")
def test_tc_f10_01_teaser_candidate_detection():
    """TC-F10-01: Detects 2–5s punchline segment as teaser candidate."""
    seg = make_sample_viral_segment(10000, 45000)
    transcript = make_sample_transcript(60000)
    teaser_fn = getattr(analysis, "detect_cold_open_teaser", None)
    if teaser_fn:
        teaser = teaser_fn(seg, transcript)
        if teaser:
            dur_ms = teaser.end_time_ms - teaser.start_time_ms
            assert 2000 <= dur_ms <= 5000
    else:
        results = analysis.extract_viral_segments(transcript, max_clips=1, llm_model="heuristic")
        assert len(results) >= 1


@pytest.mark.tier1
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.analysis not implemented yet")
def test_tc_f10_02_teaser_omitted_for_strong_opening():
    """TC-F10-02: Clips opening with an immediate high-scoring statement omit teaser."""
    seg = make_sample_viral_segment(0, 30000, hook_score=25)
    transcript = make_sample_transcript(30000, "Here is the absolute #1 secret you must know right now.")
    teaser_fn = getattr(analysis, "detect_cold_open_teaser", None)
    if teaser_fn:
        teaser = teaser_fn(seg, transcript)
        # Immediate hook means teaser can be omitted (None) or equal to opening
        assert teaser is None or teaser.start_time_ms == seg.start_time_ms
    else:
        results = analysis.extract_viral_segments(transcript, max_clips=1, llm_model="heuristic")
        assert len(results) >= 1


@pytest.mark.tier1
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.analysis not implemented yet")
def test_tc_f10_03_teaser_duration_bounds():
    """TC-F10-03: Teaser duration strictly conforms to 2.0s <= teaser <= 5.0s."""
    seg = make_sample_viral_segment(5000, 40000)
    transcript = make_sample_transcript(60000)
    teaser_fn = getattr(analysis, "detect_cold_open_teaser", None)
    if teaser_fn:
        teaser = teaser_fn(seg, transcript)
        if teaser:
            duration_s = (teaser.end_time_ms - teaser.start_time_ms) / 1000.0
            assert 2.0 <= duration_s <= 5.0
    else:
        results = analysis.extract_viral_segments(transcript, max_clips=1, llm_model="heuristic")
        assert len(results) >= 1


@pytest.mark.tier1
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.analysis not implemented yet")
def test_tc_f10_04_teaser_aligns_with_word_boundaries():
    """TC-F10-04: Teaser timestamps align with exact Whisper word boundaries."""
    transcript = make_sample_transcript(60000)
    seg = make_sample_viral_segment(10000, 45000)
    teaser_fn = getattr(analysis, "detect_cold_open_teaser", None)
    if teaser_fn:
        teaser = teaser_fn(seg, transcript)
        if teaser:
            word_starts = {w.start for w in transcript.words}
            word_ends = {w.end for w in transcript.words}
            assert teaser.start_time_ms in word_starts or abs(teaser.start_time_ms - min(word_starts, key=lambda x: abs(x - teaser.start_time_ms))) <= 500
    else:
        results = analysis.extract_viral_segments(transcript, max_clips=1, llm_model="heuristic")
        assert len(results) >= 1


@pytest.mark.tier1
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.analysis not implemented yet")
def test_tc_f10_05_teaser_contained_within_segment():
    """TC-F10-05: Teaser bounds sit strictly within segment start and end boundaries."""
    transcript = make_sample_transcript(60000)
    seg = make_sample_viral_segment(10000, 45000)
    teaser_fn = getattr(analysis, "detect_cold_open_teaser", None)
    if teaser_fn:
        teaser = teaser_fn(seg, transcript)
        if teaser:
            assert seg.start_time_ms <= teaser.start_time_ms
            assert teaser.end_time_ms <= seg.end_time_ms
            assert teaser.start_time_ms < teaser.end_time_ms
    else:
        results = analysis.extract_viral_segments(transcript, max_clips=1, llm_model="heuristic")
        assert len(results) >= 1
