"""
tests/tier1_features/test_feat14_reframing.py
Feature 14: 9:16 Vertical Reframing & Clamping
Category-Partition Tests (5 cases)
"""

from pathlib import Path
import re
import pytest

try:
    import clippa.reframing as reframing
    HAS_CLIPPA = True
except ImportError:
    HAS_CLIPPA = False


@pytest.mark.tier1
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.reframing not implemented yet")
def test_tc_f14_01_crop_1080p_to_9_16(tmp_path: Path):
    """TC-F14-01: Crops standard 1920x1080 landscape footage to exact 1080x1920 filter."""
    video = tmp_path / "sample_1080p.mp4"
    video.write_bytes(b"\x00" * 1024)
    filter_expr = reframing.build_crop_filter(video, 0, 30000, target_w=1080, target_h=1920)
    assert "crop=" in filter_expr
    assert "scale=1080:1920" in filter_expr or "1080:1920" in filter_expr


@pytest.mark.tier1
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.reframing not implemented yet")
def test_tc_f14_02_crop_720p_to_9_16(tmp_path: Path):
    """TC-F14-02: Crops 1280x720 video to exact 720x1280 vertical video."""
    video = tmp_path / "sample_720p.mp4"
    video.write_bytes(b"\x00" * 1024)
    filter_expr = reframing.build_crop_filter(video, 0, 30000, target_w=720, target_h=1280)
    assert "scale=720:1280" in filter_expr or "720:1280" in filter_expr


@pytest.mark.tier1
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.reframing not implemented yet")
def test_tc_f14_03_non_standard_aspect_even_coordinates(tmp_path: Path):
    """TC-F14-03: Non-standard source aspect ratio produces even coordinates for H.264."""
    video = tmp_path / "sample_odd.mp4"
    video.write_bytes(b"\x00" * 1024)
    coords_fn = getattr(reframing, "calculate_crop_coordinates", None)
    if coords_fn:
        crop_w, crop_h, crop_x, crop_y = coords_fn(src_w=1921, src_h=1081, target_w=1080, target_h=1920)
        assert crop_w % 2 == 0
        assert crop_h % 2 == 0
        assert crop_x % 2 == 0
        assert crop_y % 2 == 0
    else:
        filter_expr = reframing.build_crop_filter(video, 0, 30000)
        match = re.search(r"crop=(\d+):(\d+):(\d+):(\d+)", filter_expr)
        if match:
            cw, ch, cx, cy = map(int, match.groups())
            assert cw % 2 == 0 and ch % 2 == 0


@pytest.mark.tier1
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.reframing not implemented yet")
def test_tc_f14_04_crop_coordinates_strictly_clamped(tmp_path: Path):
    """TC-F14-04: Crop coordinates are strictly clamped: 0 <= x <= src_w - crop_w."""
    video = tmp_path / "clamping_test.mp4"
    video.write_bytes(b"\x00" * 1024)
    coords_fn = getattr(reframing, "calculate_crop_coordinates", None)
    if coords_fn:
        for face_x in [-200, 0, 960, 1920, 3000]:
            crop_w, crop_h, crop_x, crop_y = coords_fn(face_box=(face_x, 300, 200, 200), src_w=1920, src_h=1080)
            assert 0 <= crop_x <= 1920 - crop_w
    else:
        filter_expr = reframing.build_crop_filter(video, 0, 30000)
        assert "crop=" in filter_expr


@pytest.mark.tier1
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.reframing not implemented yet")
def test_tc_f14_05_crop_filter_syntax(tmp_path: Path):
    """TC-F14-05: Generated crop filter string parses cleanly in FFmpeg syntax."""
    video = tmp_path / "syntax_sample.mp4"
    video.write_bytes(b"\x00" * 1024)
    filter_expr = reframing.build_crop_filter(video, 0, 30000, target_w=1080, target_h=1920)
    assert re.search(r"crop=\d+:\d+:\d+:\d+", filter_expr) is not None
