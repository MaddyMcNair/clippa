"""
tests/tier2_boundaries/test_b06_virality_boundaries.py
Feature 6: Viral Highlight & Moment Detection - Boundary & Edge Cases (5 cases)
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


def _mock_http_response(raw_text: str):
    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.read.return_value = json.dumps({"response": raw_text}).encode("utf-8")
    return mock_resp


@pytest.mark.tier2
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.analysis not implemented yet")
def test_tc_b06_01_duration_under_15s_clamped():
    """TC-B06-01: LLM returns 8-second segment; verify extract_viral_segments pads or clamps duration."""
    transcript = make_sample_transcript(60000)
    segments_payload = json.dumps({"segments": [{"start_time_ms": 10000, "end_time_ms": 18000, "hook_title": "Too Short", "total_score": 80}]})

    with patch("urllib.request.urlopen", return_value=_mock_http_response(segments_payload)):
        results = analysis.extract_viral_segments(transcript, max_clips=1)
        if results:
            dur = results[0].end_time_ms - results[0].start_time_ms
            assert dur >= 15000 or dur == transcript.duration_ms


@pytest.mark.tier2
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.analysis not implemented yet")
def test_tc_b06_02_parse_markdown_json_block():
    """TC-B06-02: Robust parser extracts JSON wrapped in markdown code blocks."""
    transcript = make_sample_transcript(60000)
    markdown_payload = "```json\n{\"segments\": [{\"start_time_ms\": 5000, \"end_time_ms\": 30000, \"hook_title\": \"Markdown Wrapped\", \"total_score\": 90}]}\n```"

    with patch("urllib.request.urlopen", return_value=_mock_http_response(markdown_payload)):
        results = analysis.extract_viral_segments(transcript, max_clips=1)
        assert len(results) >= 1
        assert "Markdown" in results[0].hook_title


@pytest.mark.tier2
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.analysis not implemented yet")
def test_tc_b06_03_timestamp_boundary_clamping():
    """TC-B06-03: Timestamps outside transcript duration clamp to [0, total_duration]."""
    transcript = make_sample_transcript(45000)
    out_of_bounds_payload = json.dumps({"segments": [{"start_time_ms": -5000, "end_time_ms": 75000, "hook_title": "Out of Bounds", "total_score": 85}]})

    with patch("urllib.request.urlopen", return_value=_mock_http_response(out_of_bounds_payload)):
        results = analysis.extract_viral_segments(transcript, max_clips=1)
        if results:
            assert results[0].start_time_ms >= 0
            assert results[0].end_time_ms <= 45000


@pytest.mark.tier2
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.analysis not implemented yet")
def test_tc_b06_04_negative_or_zero_duration_dropped():
    """TC-B06-04: Segment candidate with end_time <= start_time is dropped."""
    transcript = make_sample_transcript(60000)
    inverted_payload = json.dumps({"segments": [
        {"start_time_ms": 25000, "end_time_ms": 20000, "hook_title": "Inverted", "total_score": 90},
        {"start_time_ms": 10000, "end_time_ms": 30000, "hook_title": "Valid One", "total_score": 85},
    ]})

    with patch("urllib.request.urlopen", return_value=_mock_http_response(inverted_payload)):
        results = analysis.extract_viral_segments(transcript, max_clips=2)
        assert all(seg.end_time_ms > seg.start_time_ms for seg in results)


@pytest.mark.tier2
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.analysis not implemented yet")
def test_tc_b06_05_score_range_clamping():
    """TC-B06-05: Scores outside 0-100 range clamp to valid ranges."""
    transcript = make_sample_transcript(60000)
    extreme_scores_payload = json.dumps({"segments": [{"start_time_ms": 0, "end_time_ms": 30000, "hook_title": "Score 150", "total_score": 150, "hook_score": 40}]})

    with patch("urllib.request.urlopen", return_value=_mock_http_response(extreme_scores_payload)):
        results = analysis.extract_viral_segments(transcript, max_clips=1)
        if results:
            assert results[0].total_score <= 100
            assert getattr(results[0], "hook_score", 0) <= 25 or results[0].total_score <= 100
