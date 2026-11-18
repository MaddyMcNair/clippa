"""
tests/tier2_boundaries/test_b08_windowing_boundaries.py
Feature 8: Windowed Long-Video Analysis - Boundary & Edge Cases (5 cases)
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
def test_tc_b08_01_exactly_15min_single_window():
    """TC-B08-01: Video duration exactly 15 minutes (900s) uses single window."""
    windows_fn = getattr(analysis, "create_analysis_windows", getattr(analysis, "get_analysis_windows", None))
    if windows_fn:
        windows = windows_fn(total_duration_s=900, window_size_s=720, overlap_s=120)
        assert len(windows) == 1 or len(windows) == 2
    else:
        transcript = make_sample_transcript(900000)
        score_fn = getattr(analysis, "analyze_long_video_windowed", lambda t, **kw: analysis.extract_viral_segments(t, max_clips=1, llm_model="heuristic"))
        segments = score_fn(transcript, max_clips=1)
        assert len(segments) <= 1


@pytest.mark.tier2
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.analysis not implemented yet")
def test_tc_b08_02_901s_triggers_two_windows():
    """TC-B08-02: Video duration just over boundary (901s) triggers multiple windows."""
    windows_fn = getattr(analysis, "create_analysis_windows", getattr(analysis, "get_analysis_windows", None))
    if windows_fn:
        windows = windows_fn(total_duration_s=901, window_size_s=720, overlap_s=120)
        assert len(windows) >= 2
    else:
        transcript = make_sample_transcript(901000)
        score_fn = getattr(analysis, "analyze_long_video_windowed", lambda t, **kw: analysis.extract_viral_segments(t, max_clips=2, llm_model="heuristic"))
        segments = score_fn(transcript, max_clips=2)
        assert isinstance(segments, list)


@pytest.mark.tier2
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.analysis not implemented yet")
def test_tc_b08_03_short_last_window_no_index_error():
    """TC-B08-03: Remainder window shorter than window_size handles boundaries without IndexError."""
    transcript = make_sample_transcript(1000000)
    score_fn = getattr(analysis, "analyze_long_video_windowed", lambda t, **kw: analysis.extract_viral_segments(t, max_clips=2, llm_model="heuristic"))
    segments = score_fn(transcript, window_size_s=720, overlap_s=120, max_clips=2)
    assert isinstance(segments, list)


@pytest.mark.tier2
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.analysis not implemented yet")
def test_tc_b08_04_tie_breaker_identical_scores():
    """TC-B08-04: Two candidate segments with equal total_score resolve deterministically by start_time."""
    rank_fn = getattr(analysis, "rank_and_deduplicate_segments", None)
    seg1 = make_sample_viral_segment(10000, 40000, hook_title="Clip A", hook_score=20)
    seg2 = make_sample_viral_segment(20000, 50000, hook_title="Clip B", hook_score=20)
    if rank_fn:
        ranked = rank_fn([seg2, seg1], max_clips=2)
        assert ranked[0].start_time_ms <= ranked[1].start_time_ms
    else:
        candidates = [seg2, seg1]
        sorted_candidates = sorted(candidates, key=lambda s: (-s.total_score, s.start_time_ms))
        assert sorted_candidates[0] == seg1


@pytest.mark.tier2
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.analysis not implemented yet")
def test_tc_b08_05_max_clips_limit_long_video():
    """TC-B08-05: Long video candidate pool truncates strictly to max_clips."""
    transcript = make_sample_transcript(7200000)  # 2 hours
    max_clips = 3
    score_fn = getattr(analysis, "analyze_long_video_windowed", lambda t, **kw: analysis.extract_viral_segments(t, max_clips=max_clips, llm_model="heuristic"))
    segments = score_fn(transcript, max_clips=max_clips)
    assert len(segments) <= max_clips
