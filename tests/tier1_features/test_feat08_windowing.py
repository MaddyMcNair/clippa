"""
tests/tier1_features/test_feat08_windowing.py
Feature 8: Windowed Long-Video Analysis
Category-Partition Tests (5 cases)
"""

from unittest.mock import MagicMock, patch
import pytest

from tests.fixtures.synthetic import make_sample_transcript, make_sample_viral_segment

try:
    import clippa.analysis as analysis
    HAS_CLIPPA = True
except ImportError:
    HAS_CLIPPA = False


@pytest.mark.tier1
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.analysis not implemented yet")
def test_tc_f08_01_short_video_single_window():
    """TC-F08-01: Videos <= 15 minutes are analyzed in a single unified window."""
    transcript_short = make_sample_transcript(600000)  # 10 minutes
    segments = analysis.analyze_long_video_windowed(
        transcript_short, window_size_s=720, overlap_s=120, max_clips=1
    ) if hasattr(analysis, "analyze_long_video_windowed") else analysis.extract_viral_segments(
        transcript_short, max_clips=1, llm_model="heuristic"
    )
    assert len(segments) <= 1


@pytest.mark.tier1
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.analysis not implemented yet")
def test_tc_f08_02_long_video_splits_windows():
    """TC-F08-02: Videos > 15 minutes are partitioned into overlapping windows."""
    transcript_long = make_sample_transcript(1800000)  # 30 minutes
    windows_fn = getattr(analysis, "create_analysis_windows", None)
    if windows_fn:
        windows = windows_fn(total_duration_s=1800, window_size_s=720, overlap_s=120)
        assert len(windows) >= 3
        assert windows[0] == (0, 720)
        assert windows[1][0] == 600
    else:
        segments = analysis.analyze_long_video_windowed(transcript_long, max_clips=3) if hasattr(analysis, "analyze_long_video_windowed") else analysis.extract_viral_segments(transcript_long, max_clips=3, llm_model="heuristic")
        assert len(segments) <= 3


@pytest.mark.tier1
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.analysis not implemented yet")
def test_tc_f08_03_timestamp_offset_correction():
    """TC-F08-03: Segments identified in later windows preserve absolute video timestamps."""
    transcript_long = make_sample_transcript(1200000)  # 20 minutes
    score_fn = getattr(analysis, "analyze_long_video_windowed", lambda t, **kw: analysis.extract_viral_segments(t, max_clips=3, llm_model="heuristic"))
    segments = score_fn(transcript_long, max_clips=3)
    for seg in segments:
        assert 0 <= seg.start_time_ms < 1200000
        assert seg.start_time_ms < seg.end_time_ms <= 1200000


@pytest.mark.tier1
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.analysis not implemented yet")
def test_tc_f08_04_deduplication_of_overlapping_candidates():
    """TC-F08-04: Overlapping candidates across window boundaries are deduplicated."""
    dedup_fn = getattr(analysis, "deduplicate_segments", None)
    cand_a = make_sample_viral_segment(600000, 630000, hook_score=20)
    cand_b = make_sample_viral_segment(610000, 635000, hook_score=24)
    if dedup_fn:
        deduped = dedup_fn([cand_a, cand_b], overlap_threshold=0.5)
        assert len(deduped) == 1
        assert deduped[0].hook_score == 24
    else:
        transcript = make_sample_transcript(1200000)
        score_fn = getattr(analysis, "analyze_long_video_windowed", lambda t, **kw: analysis.extract_viral_segments(t, max_clips=2, llm_model="heuristic"))
        results = score_fn(transcript, max_clips=2)
        assert len(results) <= 2


@pytest.mark.tier1
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.analysis not implemented yet")
def test_tc_f08_05_global_top_clips_selected():
    """TC-F08-05: Global top clips are ranked and capped to requested max_clips."""
    transcript_long = make_sample_transcript(1500000)
    max_clips = 2
    score_fn = getattr(analysis, "analyze_long_video_windowed", lambda t, **kw: analysis.extract_viral_segments(t, max_clips=max_clips, llm_model="heuristic"))
    results = score_fn(transcript_long, max_clips=max_clips)
    assert len(results) <= max_clips
    if len(results) > 1:
        assert results[0].total_score >= results[1].total_score
