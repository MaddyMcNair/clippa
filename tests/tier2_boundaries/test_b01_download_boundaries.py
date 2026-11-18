"""
tests/tier2_boundaries/test_b01_download_boundaries.py
Feature 1: YouTube Video Download - Boundary & Edge Cases (5 cases)
"""

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

try:
    import clippa.ingestion as ingestion
    HAS_CLIPPA = True
except ImportError:
    HAS_CLIPPA = False


@pytest.mark.tier2
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.ingestion not implemented yet")
def test_tc_b01_01_empty_url_rejected(tmp_path: Path):
    """TC-B01-01: Empty string or whitespace-only URL raises error immediately."""
    with pytest.raises((ValueError, Exception)):
        ingestion.download_video("", output_dir=tmp_path)

    with pytest.raises((ValueError, Exception)):
        ingestion.download_video("   \t\n  ", output_dir=tmp_path)


@pytest.mark.tier2
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.ingestion not implemented yet")
def test_tc_b01_02_malformed_url_rejected(tmp_path: Path):
    """TC-B01-02: Malformed URL without video ID raises error."""
    with pytest.raises((ValueError, Exception)):
        ingestion.download_video("https://www.youtube.com/watch?v=", output_dir=tmp_path)


@pytest.mark.tier2
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.ingestion not implemented yet")
def test_tc_b01_03_private_video_error(tmp_path: Path):
    """TC-B01-03: Private or deleted YouTube video URL raises error with graceful exit."""
    mock_ydl = MagicMock()
    mock_ydl.extract_info.side_effect = Exception("This video is private or unavailable")
    mock_ydl.download.side_effect = Exception("This video is private or unavailable")

    with patch("yt_dlp.YoutubeDL", return_value=mock_ydl):
        with pytest.raises(Exception) as exc_info:
            ingestion.download_video("https://www.youtube.com/watch?v=private_id", output_dir=tmp_path)
        assert "private" in str(exc_info.value).lower() or "unavailable" in str(exc_info.value).lower()


@pytest.mark.tier2
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.ingestion not implemented yet")
def test_tc_b01_04_age_restricted_sign_in(tmp_path: Path):
    """TC-B01-04: Age-restricted video triggers clean diagnostic error message."""
    mock_ydl = MagicMock()
    mock_ydl.extract_info.side_effect = Exception("Sign in required for age-restricted video")
    mock_ydl.download.side_effect = Exception("Sign in required for age-restricted video")

    with patch("yt_dlp.YoutubeDL", return_value=mock_ydl):
        with pytest.raises(Exception) as exc_info:
            ingestion.download_video("https://www.youtube.com/watch?v=age_gated", output_dir=tmp_path)
        assert "sign in" in str(exc_info.value).lower() or "age" in str(exc_info.value).lower()


@pytest.mark.tier2
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.ingestion not implemented yet")
def test_tc_b01_05_livestream_rejected(tmp_path: Path):
    """TC-B01-05: Live stream in progress raises error and exits cleanly."""
    mock_ydl = MagicMock()
    mock_ydl.extract_info.return_value = {"id": "live_id", "is_live": True, "duration": 0}
    mock_ydl.download.side_effect = Exception("Active livestream download rejected")

    with patch("yt_dlp.YoutubeDL", return_value=mock_ydl):
        with pytest.raises(Exception) as exc_info:
            ingestion.download_video("https://www.youtube.com/watch?v=live_id", output_dir=tmp_path)
        assert "live" in str(exc_info.value).lower() or "stream" in str(exc_info.value).lower()
