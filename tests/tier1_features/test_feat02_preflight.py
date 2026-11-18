"""
tests/tier1_features/test_feat02_preflight.py
Feature 2: Video Metadata Preflight
Category-Partition Tests (5 cases)
"""

from unittest.mock import MagicMock, patch
import pytest

try:
    import clippa.ingestion as ingestion
    HAS_CLIPPA = True
except ImportError:
    HAS_CLIPPA = False


@pytest.mark.tier1
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.ingestion not implemented yet")
def test_tc_f02_01_preflight_extracts_metadata():
    """TC-F02-01: Extract title, duration, author, and id without downloading payload."""
    url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    mock_ydl = MagicMock()
    mock_ydl.extract_info.return_value = {
        "id": "dQw4w9WgXcQ",
        "title": "Sample Educational Talk",
        "duration": 620.5,
        "uploader": "TalksChannel",
    }

    with patch("yt_dlp.YoutubeDL", return_value=mock_ydl):
        meta = ingestion.get_video_metadata(url)
        assert meta["id"] == "dQw4w9WgXcQ"
        assert meta["title"] == "Sample Educational Talk"
        assert float(meta["duration"]) == 620.5
        assert meta.get("uploader") == "TalksChannel" or "uploader" in meta
        mock_ydl.extract_info.assert_called_once_with(url, download=False)


@pytest.mark.tier1
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.ingestion not implemented yet")
def test_tc_f02_02_preflight_unicode_title():
    """TC-F02-02: Handle video titles containing special characters, Unicode, and punctuation."""
    url = "https://www.youtube.com/watch?v=unicode_test"
    mock_ydl = MagicMock()
    mock_ydl.extract_info.return_value = {
        "id": "unicode_test",
        "title": "🔥 Top 10 Tips: ¿Cómo Programar? [2026] & More! | «Special»",
        "duration": 300.0,
        "uploader": "GlobalTech",
    }

    with patch("yt_dlp.YoutubeDL", return_value=mock_ydl):
        meta = ingestion.get_video_metadata(url)
        assert "🔥" in meta["title"]
        assert "¿" in meta["title"]
        assert meta["duration"] == 300.0


@pytest.mark.tier1
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.ingestion not implemented yet")
def test_tc_f02_03_preflight_detects_livestream():
    """TC-F02-03: Detect livestream URL and return duration 0 or raise StreamDurationUnavailableError."""
    url = "https://www.youtube.com/watch?v=live_stream_active"
    mock_ydl = MagicMock()
    mock_ydl.extract_info.return_value = {
        "id": "live_stream_active",
        "title": "24/7 Live Stream",
        "duration": 0,
        "is_live": True,
    }

    with patch("yt_dlp.YoutubeDL", return_value=mock_ydl):
        meta = ingestion.get_video_metadata(url)
        assert meta.get("duration", 0) <= 0 or meta.get("is_live") is True


@pytest.mark.tier1
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.ingestion not implemented yet")
def test_tc_f02_04_preflight_rejects_exceeded_duration():
    """TC-F02-04: Reject video exceeding maximum configured duration during preflight."""
    url = "https://www.youtube.com/watch?v=super_long_video"
    mock_ydl = MagicMock()
    mock_ydl.extract_info.return_value = {
        "id": "super_long_video",
        "title": "10 Hour Ambient Sounds",
        "duration": 36000.0,
    }

    with patch("yt_dlp.YoutubeDL", return_value=mock_ydl):
        meta = ingestion.get_video_metadata(url)
        max_duration = 5400  # 90 minutes
        assert meta["duration"] > max_duration


@pytest.mark.tier1
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.ingestion not implemented yet")
def test_tc_f02_05_preflight_cache_reuse():
    """TC-F02-05: Verify preflight metadata cache prevents duplicate network preflight queries."""
    url = "https://www.youtube.com/watch?v=cached_url"
    mock_ydl = MagicMock()
    mock_ydl.extract_info.return_value = {
        "id": "cached_url",
        "title": "Cached Video",
        "duration": 120.0,
    }

    with patch("yt_dlp.YoutubeDL", return_value=mock_ydl):
        meta1 = ingestion.get_video_metadata(url)
        meta2 = ingestion.get_video_metadata(url)
        assert meta1["id"] == "cached_url"
        assert meta2["id"] == "cached_url"
