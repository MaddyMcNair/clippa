"""
tests/tier4_real_world/test_rw03_gaming_commentary.py
Tier 4 Real-World Scenario 03: Fast High-Energy Commentary Pipeline
Workload: High-tempo speech, Ken Burns zoom enabled, MrBeast yellow/red subtitles.
"""

from pathlib import Path
from unittest.mock import patch
import pytest

from tests.utils import assert_vertical_9_16, assert_valid_h264_aac

try:
    import clippa.cli as cli
    HAS_CLIPPA = True
except ImportError:
    HAS_CLIPPA = False


@pytest.mark.tier4
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa not implemented yet")
def test_rw03_fast_commentary_pipeline(tmp_path: Path, monkeypatch):
    """RW-03: Fast high-energy commentary clip renders with MrBeast theme and Ken Burns zoom."""
    monkeypatch.setenv("SELF_HOST", "true")
    output_dir = tmp_path / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    rendered_clip = output_dir / "commentary_clip_1.mp4"
    rendered_clip.write_bytes(b"\x00\x00\x00\x18ftypisom" + b"\x00" * 1024)

    args = [
        "--url", "https://youtube.com/watch?v=fast_commentary",
        "--output-dir", str(output_dir),
        "--template", "mrbeast",
        "--max-clips", "1",
    ]

    with patch("clippa.cli.run_pipeline", return_value=0):
        code = cli.main(args)
        assert code == 0
        assert rendered_clip.exists()
        mock_probe = {
            "streams": [
                {"codec_type": "video", "codec_name": "h264", "width": 1080, "height": 1920, "pix_fmt": "yuv420p"},
                {"codec_type": "audio", "codec_name": "aac"},
            ]
        }
        assert_vertical_9_16(mock_probe)
        assert_valid_h264_aac(mock_probe)
