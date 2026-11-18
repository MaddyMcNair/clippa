"""
tests/tier3_combinations/test_pair13_selfhost_ollama.py
Pairwise Test 13: F20 (Zero-Paid API) + F06 (Ollama Virality Scoring)
Interaction Seam: Pipeline virality scoring runs locally without paid cloud LLM calls.
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
def test_pair13_self_host_uses_ollama_not_cloud(monkeypatch):
    """TC-PAIR-13: SELF_HOST=true routes virality scoring to local Ollama or heuristic."""
    monkeypatch.setenv("SELF_HOST", "true")
    transcript = make_sample_transcript(30000)
    mock_segments = [make_sample_viral_segment(0, 30000)]

    with patch("clippa.analysis.extract_viral_segments", return_value=mock_segments) as mock_extract:
        segments = analysis.extract_viral_segments(transcript, max_clips=1, llm_model="ollama:llama3.2:3b")
        assert len(segments) == 1
        _, kwargs = mock_extract.call_args
        model = kwargs.get("llm_model", "")
        assert "openai" not in model
        assert "anthropic" not in model
