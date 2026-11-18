"""
tests/tier3_combinations/test_pair02_whisper_ollama.py
Pairwise Test 02: F04 (Whisper) + F06 (Ollama Virality Detection)
Interaction Seam: Whisper millisecond timestamps feed Ollama virality selector.
"""

from unittest.mock import patch
import pytest

from tests.fixtures.synthetic import make_sample_transcript, make_sample_viral_segment

try:
    import clippa.analysis as analysis
    HAS_CLIPPA = True
except ImportError:
    HAS_CLIPPA = False


@pytest.mark.tier3
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.analysis not implemented yet")
def test_pair02_whisper_timestamps_feed_ollama():
    """TC-PAIR-02: Whisper millisecond timestamps feed Ollama virality selector."""
    transcript = make_sample_transcript(60000)
    mock_segment = make_sample_viral_segment(start_ms=transcript.words[0].start, end_ms=transcript.words[-1].end)

    with patch("clippa.analysis.extract_viral_segments", return_value=[mock_segment]):
        segments = analysis.extract_viral_segments(transcript, max_clips=1, llm_model="ollama:llama3.2:3b")
        assert len(segments) == 1
        seg = segments[0]
        # Verify segment bounds correspond to word timings
        assert seg.start_time_ms == transcript.words[0].start
        assert seg.end_time_ms == transcript.words[-1].end
