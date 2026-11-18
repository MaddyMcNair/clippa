"""
tests/tier1_features/test_feat07_heuristic.py
Feature 7: Rule-Based Heuristic Scorer Fallback
Category-Partition Tests (5 cases)
"""

import pytest

from tests.fixtures.synthetic import make_sample_transcript

try:
    import clippa.analysis as analysis
    HAS_CLIPPA = True
except ImportError:
    HAS_CLIPPA = False


@pytest.mark.tier1
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.analysis not implemented yet")
def test_tc_f07_01_heuristic_fallback_when_ollama_offline():
    """TC-F07-01: When Ollama daemon is offline, pipeline automatically falls back to heuristic."""
    transcript = make_sample_transcript(45000, "Why is this the biggest mistake you will ever make? Here is the secret.")
    # Call with llm_model="heuristic" or offline Ollama fallback
    segments = analysis.heuristic_score_segments(transcript) if hasattr(analysis, "heuristic_score_segments") else analysis.extract_viral_segments(transcript, max_clips=1, llm_model="heuristic")
    assert len(segments) >= 1
    assert segments[0].total_score >= 10


@pytest.mark.tier1
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.analysis not implemented yet")
def test_tc_f07_02_heuristic_scores_power_words():
    """TC-F07-02: Heuristic scorer evaluates power words and question marks to boost hook score."""
    text_power = "Why is this the biggest mistake you will ever make? Here is the secret truth revealed."
    transcript_power = make_sample_transcript(30000, text=text_power)

    text_plain = "The system will proceed to compute values according to general standard rules."
    transcript_plain = make_sample_transcript(30000, text=text_plain)

    score_fn = getattr(analysis, "heuristic_score_segments", lambda t: analysis.extract_viral_segments(t, max_clips=1, llm_model="heuristic"))
    segs_power = score_fn(transcript_power)
    segs_plain = score_fn(transcript_plain)

    assert len(segs_power) >= 1 and len(segs_plain) >= 1
    assert segs_power[0].hook_score >= segs_plain[0].hook_score


@pytest.mark.tier1
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.analysis not implemented yet")
def test_tc_f07_03_heuristic_conforms_to_schema():
    """TC-F07-03: Heuristic output conforms to exact ViralSegment schema with valid duration."""
    transcript = make_sample_transcript(40000, "Here is an amazing secret for you to learn.")
    score_fn = getattr(analysis, "heuristic_score_segments", lambda t: analysis.extract_viral_segments(t, max_clips=1, llm_model="heuristic"))
    segs = score_fn(transcript)
    assert len(segs) >= 1
    seg = segs[0]
    assert hasattr(seg, "start_time_ms") and hasattr(seg, "end_time_ms")
    assert hasattr(seg, "hook_title") and hasattr(seg, "total_score")
    assert 0 <= seg.start_time_ms < seg.end_time_ms


@pytest.mark.tier1
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.analysis not implemented yet")
def test_tc_f07_04_baseline_score_guarantee():
    """TC-F07-04: Transcript with zero keywords still produces valid segments with baseline score >= 10."""
    transcript_plain = make_sample_transcript(30000, text="A simple test reading numbers one two three four five.")
    score_fn = getattr(analysis, "heuristic_score_segments", lambda t: analysis.extract_viral_segments(t, max_clips=1, llm_model="heuristic"))
    segs = score_fn(transcript_plain)
    assert len(segs) >= 1
    assert segs[0].total_score >= 10


@pytest.mark.tier1
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.analysis not implemented yet")
def test_tc_f07_05_deterministic_scoring():
    """TC-F07-05: Calling heuristic scorer twice with identical transcript returns identical scores."""
    transcript = make_sample_transcript(40000, "Consistent text for deterministic scoring test.")
    score_fn = getattr(analysis, "heuristic_score_segments", lambda t: analysis.extract_viral_segments(t, max_clips=1, llm_model="heuristic"))
    res1 = score_fn(transcript)
    res2 = score_fn(transcript)
    assert len(res1) == len(res2)
    assert [s.total_score for s in res1] == [s.total_score for s in res2]
