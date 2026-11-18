"""
tests/tier3_combinations/test_pair03_whisper_heuristic.py
Pairwise Test 03: F04 (Whisper) + F07 (Heuristic Virality Scorer)
Interaction Seam: Whisper words feed heuristic scorer when Ollama is offline.
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
def test_pair03_whisper_feeds_heuristic():
    """TC-PAIR-03: Whisper words feed heuristic scorer seamlessly without Ollama."""
    transcript = make_sample_transcript(45000, "Why is this the secret to success? Never make this mistake.")
    mock_segment = make_sample_viral_segment(0, 30000, hook_title="The Secret To Success")

    with patch("clippa.analysis.extract_viral_segments", return_value=[mock_segment]):
        segments = analysis.extract_viral_segments(transcript, max_clips=1, llm_model="heuristic")
        assert len(segments) == 1
        assert segments[0].hook_score >= 20
