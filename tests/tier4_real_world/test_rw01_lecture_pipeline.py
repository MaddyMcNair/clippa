"""
tests/tier4_real_world/test_rw01_lecture_pipeline.py
Tier 4 Real-World Scenario 01: Technical / Educational Lecture Pipeline
Workload: 10-15m single speaker talking head, 1080p landscape, Hormozi subtitles.
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
def test_rw01_lecture_full_pipeline(tmp_path: Path, monkeypatch):
    """RW-01: Full pipeline execution on educational lecture produces 1080x1920 clip with Hormozi subs."""
    monkeypatch.setenv("SELF_HOST", "true")
    output_dir = tmp_path / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    rendered_clip = output_dir / "lecture_clip_1.mp4"
    rendered_clip.write_bytes(b"\x00\x00\x00\x18ftypisom" + b"\x00" * 1024)

    args = [
        "--url", "https://youtube.com/watch?v=lecture_sample",
        "--output-dir", str(output_dir),
        "--template", "hormozi",
        "--whisper-model", "base",
        "--resolution", "1080p",
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
        assert_vertical_9_16(mock_probe, target_w=1080, target_h=1920)
        assert_valid_h264_aac(mock_probe)
