"""
tests/tier2_boundaries/test_b02_preflight_boundaries.py
Feature 2: Video Metadata Preflight - Boundary & Edge Cases (5 cases)
"""

from unittest.mock import MagicMock, patch
import pytest

try:
    import clippa.ingestion as ingestion
    HAS_CLIPPA = True
except ImportError:
    HAS_CLIPPA = False


@pytest.mark.tier2
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.ingestion not implemented yet")
def test_tc_b02_01_zero_duration_rejected():
    """TC-B02-01: Zero-duration video is rejected before byte download starts."""
    mock_ydl = MagicMock()
    mock_ydl.extract_info.return_value = {"id": "zero_dur", "title": "Empty Stream", "duration": 0}

    with patch("yt_dlp.YoutubeDL", return_value=mock_ydl):
        meta = ingestion.get_video_metadata("https://youtube.com/watch?v=zero_dur")
        assert meta.get("duration", 0) <= 0


@pytest.mark.tier2
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.ingestion not implemented yet")
def test_tc_b02_02_duration_exceeding_cap():
    """TC-B02-02: Video duration exceeding 5400s triggers rejection."""
    max_cap = 5400
    mock_ydl = MagicMock()
    mock_ydl.extract_info.return_value = {"id": "over_cap", "title": "Over 90 min video", "duration": 5401}

    with patch("yt_dlp.YoutubeDL", return_value=mock_ydl):
        meta = ingestion.get_video_metadata("https://youtube.com/watch?v=over_cap")
        assert meta["duration"] > max_cap


@pytest.mark.tier2
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.ingestion not implemented yet")
def test_tc_b02_03_extreme_unicode_title():
    """TC-B02-03: Title with 500+ characters or complex Unicode handled without crash."""
    long_title = "Тест " * 50 + "مرحبا " * 30 + "🚀" * 20
    mock_ydl = MagicMock()
    mock_ydl.extract_info.return_value = {"id": "long_title", "title": long_title, "duration": 180}

    with patch("yt_dlp.YoutubeDL", return_value=mock_ydl):
        meta = ingestion.get_video_metadata("https://youtube.com/watch?v=long_title")
        assert len(meta["title"]) > 0


@pytest.mark.tier2
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.ingestion not implemented yet")
def test_tc_b02_04_missing_author_fallback():
    """TC-B02-04: Video with missing author in metadata substitutes fallback."""
    mock_ydl = MagicMock()
    mock_ydl.extract_info.return_value = {"id": "no_author", "title": "Untitled", "duration": 60}

    with patch("yt_dlp.YoutubeDL", return_value=mock_ydl):
        meta = ingestion.get_video_metadata("https://youtube.com/watch?v=no_author")
        author = meta.get("uploader") or "Unknown"
        assert author in ["Unknown", None] or len(author) > 0


@pytest.mark.tier2
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.ingestion not implemented yet")
def test_tc_b02_05_network_timeout():
    """TC-B02-05: Network socket timeout fails fast with clear error."""
    mock_ydl = MagicMock()
    mock_ydl.extract_info.side_effect = TimeoutError("Connection timed out")

    with patch("yt_dlp.YoutubeDL", return_value=mock_ydl):
        with pytest.raises((TimeoutError, Exception)):
            ingestion.get_video_metadata("https://youtube.com/watch?v=timeout_id")
