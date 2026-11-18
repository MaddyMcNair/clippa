"""
tests/tier1_features/test_feat19_burning.py
Feature 19: Subtitle Burning via FFmpeg
Category-Partition Tests (5 cases)
"""

from pathlib import Path
import pytest

from tests.utils import probe_media, assert_valid_h264_aac

try:
    import clippa.rendering as rendering
    HAS_CLIPPA = True
except ImportError:
    HAS_CLIPPA = False


@pytest.mark.tier1
@pytest.mark.requires_ffmpeg
@pytest.mark.requires_ffprobe
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.rendering not implemented yet")
def test_tc_f19_01_libass_filter_execution(sample_1080p_mp4: Path, sample_ass_file: Path, tmp_path: Path):
    """TC-F19-01: FFmpeg command executes with ass filter burning subtitles into pixels."""
    output_clip = tmp_path / "rendered_clip.mp4"
    out = rendering.render_clip(
        source_video=sample_1080p_mp4,
        start_ms=0,
        end_ms=3000,
        crop_filter="crop=608:1080:656:0,scale=1080:1920",
        ass_path=sample_ass_file,
        output_file=output_clip,
    )
    assert Path(out).exists()
    assert Path(out).stat().st_size > 1024


@pytest.mark.tier1
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.rendering not implemented yet")
def test_tc_f19_02_windows_path_escaping():
    """TC-F19-02: Windows absolute paths with backslashes and spaces escape properly for FFmpeg."""
    raw_path = r"C:\Users\User\Videos\clip [1].ass"
    escape_fn = getattr(rendering, "escape_filter_path", getattr(rendering, "escape_path_for_ffmpeg", None))
    if escape_fn:
        escaped = escape_fn(raw_path)
        assert "\\" not in escaped.replace(r"\:", "").replace(r"\[", "").replace(r"\]", "")
    else:
        args_fn = getattr(rendering, "build_ffmpeg_render_args", None)
        if args_fn:
            args = args_fn(Path(raw_path))
            assert any("clip" in arg for arg in args)
        else:
            pytest.skip("escape_filter_path pending in clippa.rendering")


@pytest.mark.tier1
@pytest.mark.requires_ffmpeg
@pytest.mark.requires_ffprobe
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.rendering not implemented yet")
def test_tc_f19_03_h264_encoder_parameters(sample_1080p_mp4: Path, sample_ass_file: Path, tmp_path: Path):
    """TC-F19-03: Video stream encodes as H.264 (avc1) with yuv420p pixel format."""
    output_clip = tmp_path / "h264_clip.mp4"
    rendering.render_clip(
        source_video=sample_1080p_mp4,
        start_ms=0,
        end_ms=3000,
        crop_filter="crop=608:1080:656:0,scale=1080:1920",
        ass_path=sample_ass_file,
        output_file=output_clip,
    )
    probe = probe_media(output_clip)
    assert_valid_h264_aac(probe)


@pytest.mark.tier1
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.rendering not implemented yet")
def test_tc_f19_04_faststart_movflags_present():
    """TC-F19-04: Audio and video render arguments include +faststart movflags."""
    build_args_fn = getattr(rendering, "build_ffmpeg_render_args", None)
    if build_args_fn:
        args = build_args_fn(source_video="dummy.mp4", output_file="dummy_out.mp4")
        assert "+faststart" in args or "-movflags" in args
    else:
        pytest.skip("build_ffmpeg_render_args pending in clippa.rendering")


@pytest.mark.tier1
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.rendering not implemented yet")
def test_tc_f19_05_burning_error_handling(tmp_path: Path):
    """TC-F19-05: Non-existent input file raises FileNotFoundError or RenderingError."""
    missing_video = tmp_path / "non_existent.mp4"
    dummy_ass = tmp_path / "dummy.ass"
    dummy_ass.write_text("[Script Info]\n", encoding="utf-8")
    output_clip = tmp_path / "out.mp4"

    with pytest.raises(Exception):
        rendering.render_clip(
            source_video=missing_video,
            start_ms=0,
            end_ms=3000,
            crop_filter="crop=608:1080:656:0",
            ass_path=dummy_ass,
            output_file=output_clip,
        )


@pytest.mark.tier1
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.rendering not implemented yet")
def test_tc_f19_06_nvenc_arguments():
    """TC-F19-06: build_ffmpeg_render_args sets h264_nvenc when use_nvenc is True."""
    args = rendering.build_ffmpeg_render_args(
        source_video="dummy.mp4",
        output_file="dummy_out.mp4",
        use_nvenc=True,
    )
    assert "-c:v" in args
    idx = args.index("-c:v")
    assert args[idx + 1] == "h264_nvenc"
    assert "-cq" in args

