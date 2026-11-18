"""
tests/tier1_features/test_feat06_virality.py
Feature 6: Viral Highlight & Moment Detection
Category-Partition Tests (5 cases)
"""

import json
from unittest.mock import MagicMock, patch
import pytest

from tests.fixtures.synthetic import make_sample_transcript

try:
    import clippa.analysis as analysis
    HAS_CLIPPA = True
except ImportError:
    HAS_CLIPPA = False


def _mock_ollama_response(segments):
    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.read.return_value = json.dumps({"response": json.dumps({"segments": segments})}).encode("utf-8")
    return mock_resp


@pytest.mark.tier1
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.analysis not implemented yet")
def test_tc_f06_01_virality_4_dimensions():
    """TC-F06-01: Analysis evaluates segments across 4 dimensions (hook, engagement, value, shareability)."""
    transcript = make_sample_transcript(60000)
    mock_segments = [
        {
            "start_time_ms": 10000,
            "end_time_ms": 40000,
            "hook_title": "Huge Wealth Mistake",
            "hook_score": 24,
            "engagement_score": 23,
            "value_score": 22,
            "shareability_score": 25,
            "total_score": 94,
            "reasoning": "High-impact hook with strong emotional resonance.",
        }
    ]

    with patch("urllib.request.urlopen", return_value=_mock_ollama_response(mock_segments)):
        segments = analysis.extract_viral_segments(transcript, max_clips=1)
        assert len(segments) > 0
        seg = segments[0]
        assert 0 <= seg.hook_score <= 25
        assert 0 <= seg.engagement_score <= 25
        assert 0 <= seg.value_score <= 25
        assert 0 <= seg.shareability_score <= 25


@pytest.mark.tier1
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.analysis not implemented yet")
def test_tc_f06_02_total_score_equals_sum():
    """TC-F06-02: Total virality score equals sum of 4 dimensions (0–100 scale)."""
    transcript = make_sample_transcript(60000)
    mock_segments = [
        {
            "start_time_ms": 5000,
            "end_time_ms": 35000,
            "hook_title": "The Big Reveal",
            "hook_score": 20,
            "engagement_score": 22,
            "value_score": 21,
            "shareability_score": 25,
            "total_score": 88,
            "reasoning": "Engaging climax.",
        }
    ]

    with patch("urllib.request.urlopen", return_value=_mock_ollama_response(mock_segments)):
        segments = analysis.extract_viral_segments(transcript, max_clips=1)
        seg = segments[0]
        expected_sum = seg.hook_score + seg.engagement_score + seg.value_score + seg.shareability_score
        assert seg.total_score == expected_sum
        assert 0 <= seg.total_score <= 100


@pytest.mark.tier1
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.analysis not implemented yet")
def test_tc_f06_03_duration_bounds():
    """TC-F06-03: Selected viral segments strictly adhere to duration bounds: 15s <= duration <= 60s."""
    transcript = make_sample_transcript(120000)
    mock_segments = [
        {
            "start_time_ms": 10000,
            "end_time_ms": 40000,
            "hook_title": "Segment A",
            "hook_score": 22,
            "engagement_score": 20,
            "value_score": 21,
            "shareability_score": 20,
            "total_score": 83,
            "reasoning": "Reasoning A",
        },
        {
            "start_time_ms": 50000,
            "end_time_ms": 95000,
            "hook_title": "Segment B",
            "hook_score": 23,
            "engagement_score": 22,
            "value_score": 22,
            "shareability_score": 21,
            "total_score": 88,
            "reasoning": "Reasoning B",
        },
    ]

    with patch("urllib.request.urlopen", return_value=_mock_ollama_response(mock_segments)):
        segments = analysis.extract_viral_segments(transcript, max_clips=2)
        for seg in segments:
            duration_s = (seg.end_time_ms - seg.start_time_ms) / 1000.0
            assert 15.0 <= duration_s <= 60.0


@pytest.mark.tier1
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.analysis not implemented yet")
def test_tc_f06_04_segments_sorted_descending():
    """TC-F06-04: Multiple extracted segments are sorted in descending order of total virality score."""
    transcript = make_sample_transcript(120000)
    mock_segments = [
        {"start_time_ms": 0, "end_time_ms": 25000, "hook_title": "Low Score", "total_score": 60, "hook_score": 15, "engagement_score": 15, "value_score": 15, "shareability_score": 15, "reasoning": "Low"},
        {"start_time_ms": 30000, "end_time_ms": 60000, "hook_title": "High Score", "total_score": 95, "hook_score": 24, "engagement_score": 24, "value_score": 23, "shareability_score": 24, "reasoning": "High"},
        {"start_time_ms": 70000, "end_time_ms": 100000, "hook_title": "Med Score", "total_score": 80, "hook_score": 20, "engagement_score": 20, "value_score": 20, "shareability_score": 20, "reasoning": "Med"},
    ]

    with patch("urllib.request.urlopen", return_value=_mock_ollama_response(mock_segments)):
        segments = analysis.extract_viral_segments(transcript, max_clips=3)
        scores = [s.total_score for s in segments]
        assert scores == sorted(scores, reverse=True)


@pytest.mark.tier1
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.analysis not implemented yet")
def test_tc_f06_05_segment_metadata_completeness():
    """TC-F06-05: Each segment includes non-empty reasoning and valid start/end timestamps."""
    transcript = make_sample_transcript(60000)
    mock_segments = [
        {
            "start_time_ms": 10000,
            "end_time_ms": 35000,
            "hook_title": "Complete Segment",
            "hook_score": 22,
            "engagement_score": 22,
            "value_score": 22,
            "shareability_score": 22,
            "total_score": 88,
            "reasoning": "Clear emotional hook with punchline payoff.",
        }
    ]

    with patch("urllib.request.urlopen", return_value=_mock_ollama_response(mock_segments)):
        segments = analysis.extract_viral_segments(transcript, max_clips=1)
        for seg in segments:
            assert isinstance(seg.reasoning, str) and len(seg.reasoning) > 0
            assert isinstance(seg.hook_title, str) and len(seg.hook_title) > 0
            assert 0 <= seg.start_time_ms < seg.end_time_ms
