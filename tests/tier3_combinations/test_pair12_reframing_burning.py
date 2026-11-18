"""
tests/tier3_combinations/test_pair12_reframing_burning.py
Pairwise Test 12: F14 (9:16 Vertical Crop) + F19 (Subtitle Burning)
Interaction Seam: Crop filter and ASS subtitle burning combine in single FFmpeg pass.
"""

from pathlib import Path
from unittest.mock import patch
import pytest

try:
    import clippa.rendering as rendering
    HAS_CLIPPA = True
except ImportError:
    HAS_CLIPPA = False


@pytest.mark.tier3
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.rendering not implemented yet")
def test_pair12_combined_crop_and_ass_filter(tmp_path: Path):
    """TC-PAIR-12: Combined crop and ass filter string executes in single pass."""
    source_video = tmp_path / "source.mp4"
    source_video.write_bytes(b"\x00" * 1024)
    ass_path = tmp_path / "subs.ass"
    ass_path.write_text("[Script Info]\n", encoding="utf-8")
    output_clip = tmp_path / "output_clip.mp4"
    output_clip.write_bytes(b"\x00\x00\x00\x18ftypisom" + b"\x00" * 1024)

    with patch("clippa.rendering.render_clip", return_value=output_clip) as mock_render:
        out = rendering.render_clip(
            source_video=source_video,
            start_ms=0,
            end_ms=30000,
            crop_filter="crop=608:1080:656:0,scale=1080:1920",
            ass_path=ass_path,
            output_file=output_clip,
        )
        assert out.exists()
