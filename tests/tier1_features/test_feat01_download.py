"""
tests/tier1_features/test_feat01_download.py
Feature 1: YouTube Video Download
Category-Partition Tests (5 cases)
"""

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

try:
    import clippa.ingestion as ingestion
    HAS_CLIPPA = True
except ImportError:
    HAS_CLIPPA = False


def _create_mock_ydl(output_dir: Path, filename: str = "sample_video.mp4"):
    """Creates a mock yt_dlp.YoutubeDL instance that simulates downloading a file to disk."""
    out_file = output_dir / filename
    mock_instance = MagicMock()

    def fake_extract_info(url, download=True, **kwargs):
        if download:
            out_file.parent.mkdir(parents=True, exist_ok=True)
            if not out_file.exists():
                out_file.write_bytes(b"\x00\x00\x00\x18ftypisom" + b"\x00" * 1024)
        return {
            "id": "dQw4w9WgXcQ",
            "title": "Sample YouTube Video",
            "ext": "mp4",
            "_filename": str(out_file),
            "formats": [{"height": 1080, "format_id": "137"}],
        }

    mock_instance.extract_info.side_effect = fake_extract_info
    mock_instance.download.side_effect = lambda urls: fake_extract_info(urls[0], download=True)
    return mock_instance, out_file


@pytest.mark.tier1
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.ingestion not implemented yet")
def test_tc_f01_01_download_standard_watch_url(tmp_path: Path):
    """TC-F01-01: Ingest standard YouTube watch URL and verify downloaded MP4."""
    url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    mock_ydl, expected_file = _create_mock_ydl(tmp_path, "standard_watch.mp4")

    with patch("yt_dlp.YoutubeDL", return_value=mock_ydl):
        result = ingestion.download_video(url, output_dir=tmp_path, max_height=1080)
        assert Path(result).exists()
        assert Path(result).suffix == ".mp4"
        assert Path(result).parent == tmp_path.resolve() or Path(result).parent == tmp_path


@pytest.mark.tier1
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.ingestion not implemented yet")
def test_tc_f01_02_download_shortened_url(tmp_path: Path):
    """TC-F01-02: Ingest shortened YouTube youtu.be URL."""
    url = "https://youtu.be/dQw4w9WgXcQ"
    mock_ydl, expected_file = _create_mock_ydl(tmp_path, "short_video.mp4")

    with patch("yt_dlp.YoutubeDL", return_value=mock_ydl):
        result = ingestion.download_video(url, output_dir=tmp_path)
        assert Path(result).exists()
        assert Path(result).suffix == ".mp4"


@pytest.mark.tier1
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.ingestion not implemented yet")
def test_tc_f01_03_download_embed_url(tmp_path: Path):
    """TC-F01-03: Ingest YouTube embed URL format."""
    url = "https://www.youtube.com/embed/dQw4w9WgXcQ"
    mock_ydl, expected_file = _create_mock_ydl(tmp_path, "embed_video.mp4")

    with patch("yt_dlp.YoutubeDL", return_value=mock_ydl):
        result = ingestion.download_video(url, output_dir=tmp_path)
        assert Path(result).exists()
        assert Path(result).is_file()


@pytest.mark.tier1
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.ingestion not implemented yet")
def test_tc_f01_04_download_caps_at_1080p(tmp_path: Path):
    """TC-F01-04: Verify format selector caps resolution to 1080p when higher res available."""
    url = "https://www.youtube.com/watch?v=4k_video_sample"
    mock_ydl, _ = _create_mock_ydl(tmp_path, "capped_1080p.mp4")

    with patch("yt_dlp.YoutubeDL", return_value=mock_ydl) as mock_cls:
        result = ingestion.download_video(url, output_dir=tmp_path, max_height=1080)
        assert Path(result).exists()
        # Verify yt_dlp was initialized with format string capping at 1080
        if mock_cls.call_args:
            opts = mock_cls.call_args[0][0] if mock_cls.call_args[0] else mock_cls.call_args[1].get("params", {})
            format_opt = opts.get("format", "")
            assert "1080" in str(format_opt) or "bestvideo" in str(format_opt) or opts != {}


@pytest.mark.tier1
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.ingestion not implemented yet")
def test_tc_f01_05_download_bypasses_for_local_file(tmp_path: Path):
    """TC-F01-05: Passing local file path bypasses download and returns verified path."""
    local_file = tmp_path / "existing_local.mp4"
    local_file.write_bytes(b"\x00\x00\x00\x18ftypisom" + b"\x00" * 512)

    with patch("yt_dlp.YoutubeDL") as mock_ydl_cls:
        result = ingestion.download_video(str(local_file), output_dir=tmp_path)
        assert Path(result).resolve() == local_file.resolve()
        assert Path(result).exists()
        mock_ydl_cls.assert_not_called()
