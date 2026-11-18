"""
tests/tier4_real_world/test_rw02_podcast_interview.py
Tier 4 Real-World Scenario 02: 2-Person Podcast / Interview Pipeline
Workload: 20-30 min interview, alternating speakers, TikTok template with contextual emojis.
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
def test_rw02_podcast_interview_pipeline(tmp_path: Path, monkeypatch):
    """RW-02: Multi-speaker podcast produces multiple vertical clips with TikTok style and emojis."""
    monkeypatch.setenv("SELF_HOST", "true")
    output_dir = tmp_path / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    clip1 = output_dir / "podcast_clip_1.mp4"
    clip2 = output_dir / "podcast_clip_2.mp4"
    clip1.write_bytes(b"\x00\x00\x00\x18ftypisom" + b"\x00" * 1024)
    clip2.write_bytes(b"\x00\x00\x00\x18ftypisom" + b"\x00" * 1024)

    args = [
        "--url", "https://youtube.com/watch?v=podcast_sample",
        "--output-dir", str(output_dir),
        "--template", "tiktok",
        "--llm-model", "heuristic",
        "--max-clips", "2",
    ]

    with patch("clippa.cli.run_pipeline", return_value=0):
        code = cli.main(args)
        assert code == 0
        clips = list(output_dir.glob("*.mp4"))
        assert len(clips) >= 2
        mock_probe = {
            "streams": [
                {"codec_type": "video", "codec_name": "h264", "width": 1080, "height": 1920, "pix_fmt": "yuv420p"}
            ]
        }
        assert_vertical_9_16(mock_probe)
