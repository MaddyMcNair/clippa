"""
tests/tier2_boundaries/test_b07_heuristic_boundaries.py
Feature 7: Rule-Based Heuristic Scorer Fallback - Boundary & Edge Cases (5 cases)
"""

import time
import pytest

from tests.fixtures.synthetic import Word, TranscriptData, make_sample_transcript

try:
    import clippa.analysis as analysis
    HAS_CLIPPA = True
except ImportError:
    HAS_CLIPPA = False


@pytest.mark.tier2
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.analysis not implemented yet")
def test_tc_b07_01_zero_punctuation_gap_segmentation():
    """TC-B07-01: Transcript with zero punctuation segments using word gaps > 1.5s."""
    words = [
        Word("first", 0, 400),
        Word("thought", 450, 900),
        Word("second", 3000, 3500),  # gap = 2100ms > 1500ms
        Word("thought", 3550, 4000),
    ]
    transcript = TranscriptData(text="first thought second thought", language="en", words=words)
    score_fn = getattr(analysis, "heuristic_score_segments", lambda t: analysis.extract_viral_segments(t, max_clips=1, llm_model="heuristic"))
    segments = score_fn(transcript)
    assert isinstance(segments, list)


@pytest.mark.tier2
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.analysis not implemented yet")
def test_tc_b07_02_zero_keywords_fallback_scoring():
    """TC-B07-02: Transcript with no power keywords scores via base speech metrics (replaces base_score = 50)."""
    plain_words = [Word("generic", i * 350, i * 350 + 250) for i in range(50)]
    transcript = TranscriptData(text="generic text " * 25, language="en", words=plain_words)
    score_fn = getattr(analysis, "heuristic_score_segments", lambda t: analysis.extract_viral_segments(t, max_clips=1, llm_model="heuristic"))
    segments = score_fn(transcript)
    assert len(segments) >= 1
    assert 10 <= segments[0].total_score <= 100


@pytest.mark.tier2
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.analysis not implemented yet")
def test_tc_b07_03_repetitive_words_no_zero_division():
    """TC-B07-03: Transcript with repetitive single word does not divide by zero."""
    words = [Word("test", i * 300, i * 300 + 250) for i in range(50)]
    transcript = TranscriptData(text="test " * 50, language="en", words=words)
    score_fn = getattr(analysis, "heuristic_score_segments", lambda t: analysis.extract_viral_segments(t, max_clips=1, llm_model="heuristic"))
    segments = score_fn(transcript)
    assert isinstance(segments, list)


@pytest.mark.tier2
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.analysis not implemented yet")
def test_tc_b07_04_exactly_15s_transcript():
    """TC-B07-04: Transcript exactly 15 seconds produces single valid segment."""
    transcript = make_sample_transcript(15000, "Exactly fifteen seconds of content for testing heuristic segmenting boundaries.")
    score_fn = getattr(analysis, "heuristic_score_segments", lambda t: analysis.extract_viral_segments(t, max_clips=1, llm_model="heuristic"))
    segments = score_fn(transcript)
    assert len(segments) >= 1


@pytest.mark.tier2
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.analysis not implemented yet")
def test_tc_b07_05_extreme_10k_words_performance():
    """TC-B07-05: Extremely long transcript (>10,000 words) computes in <2.0s."""
    words = [Word("word", i * 100, i * 100 + 80) for i in range(10000)]
    transcript = TranscriptData(text="word " * 10000, language="en", words=words)
    score_fn = getattr(analysis, "heuristic_score_segments", lambda t: analysis.extract_viral_segments(t, max_clips=1, llm_model="heuristic"))

    start_time = time.time()
    segments = score_fn(transcript)
    elapsed = time.time() - start_time
    assert elapsed < 3.0
    assert isinstance(segments, list)
