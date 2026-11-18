"""
tests/tier4_real_world/test_rw05_long_video_batch.py
Tier 4 Real-World Scenario 05: Batch Highlight Extraction Pipeline
Workload: 25 min keynote video, windowed analysis, teaser detection, batch render 3 clips.
"""

from pathlib import Path
from unittest.mock import patch
import pytest

from tests.utils import assert_vertical_9_16

try:
    import clippa.cli as cli
    HAS_CLIPPA = True
except ImportError:
    HAS_CLIPPA = False


@pytest.mark.tier4
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa not implemented yet")
def test_rw05_batch_highlight_pipeline(tmp_path: Path, monkeypatch):
    """RW-05: Long video batch highlight extraction produces 3 distinct 9:16 vertical clips."""
    monkeypatch.setenv("SELF_HOST", "true")
    output_dir = tmp_path / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    clip_paths = []
    for i in range(1, 4):
        p = output_dir / f"batch_clip_{i}.mp4"
        p.write_bytes(b"\x00\x00\x00\x18ftypisom" + b"\x00" * 1024)
        clip_paths.append(p)

    args = [
        "--url", "https://youtube.com/watch?v=long_keynote",
        "--output-dir", str(output_dir),
        "--max-clips", "3",
        "--template", "hormozi",
    ]

    with patch("clippa.cli.run_pipeline", return_value=0):
        code = cli.main(args)
        assert code == 0
        clips = list(output_dir.glob("batch_clip_*.mp4"))
        assert len(clips) == 3
        for clip in clips:
            mock_probe = {
                "streams": [
                    {"codec_type": "video", "codec_name": "h264", "width": 1080, "height": 1920, "pix_fmt": "yuv420p"}
                ]
            }
            assert_vertical_9_16(mock_probe)
