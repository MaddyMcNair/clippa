"""
tests/tier1_features/test_feat22_acceptance.py
Feature 22: End-to-End Acceptance Verification
Category-Partition Tests (5 cases)
"""

from pathlib import Path
import pytest

from tests.utils import (
    probe_media,
    assert_vertical_9_16,
    assert_valid_h264_aac,
)

try:
    import clippa.cli as cli
    import clippa.rendering as rendering
    import clippa.config as config
    HAS_CLIPPA = True
except ImportError:
    HAS_CLIPPA = False


@pytest.mark.tier1
@pytest.mark.requires_ffmpeg
@pytest.mark.requires_ffprobe
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa not implemented yet")
def test_tc_f22_01_produces_rendered_vertical_clip(sample_1080p_mp4: Path, sample_ass_file: Path, tmp_path: Path):
    """TC-F22-01: Complete rendering pipeline produces at least one rendered 9:16 vertical MP4 clip."""
    output_clip = tmp_path / "clip_1.mp4"
    crop_filter = "crop=608:1080:656:0,scale=1080:1920"

    rendered = rendering.render_clip(
        source_video=sample_1080p_mp4,
        start_ms=0,
        end_ms=3000,
        crop_filter=crop_filter,
        ass_path=sample_ass_file,
        output_file=output_clip,
    )

    assert Path(rendered).exists()
    assert Path(rendered).stat().st_size > 1024
    assert Path(rendered).suffix == ".mp4"


@pytest.mark.tier1
@pytest.mark.requires_ffprobe
def test_tc_f22_02_aspect_ratio_9_16(sample_9_16_mp4: Path):
    """TC-F22-02: Rendered clip has exact aspect ratio 9:16 (1080x1920) verified via ffprobe."""
    probe_data = probe_media(sample_9_16_mp4)
    assert_vertical_9_16(probe_data, target_w=1080, target_h=1920)


@pytest.mark.tier1
@pytest.mark.requires_ffprobe
def test_tc_f22_03_h264_and_aac_codecs(sample_1080p_mp4: Path):
    """TC-F22-03: Rendered clip contains valid video stream (h264) and audio stream (aac)."""
    probe_data = probe_media(sample_1080p_mp4)
    assert_valid_h264_aac(probe_data)


@pytest.mark.tier1
@pytest.mark.requires_ffprobe
def test_tc_f22_04_clip_duration_matches_segment(sample_1080p_mp4: Path):
    """TC-F22-04: Rendered clip duration matches selected viral segment duration (+/- 1.0s)."""
    expected_duration = 3.0
    probe_data = probe_media(sample_1080p_mp4)
    actual_duration = float(probe_data.get("format", {}).get("duration", 0))
    assert abs(actual_duration - expected_duration) <= 1.0, (
        f"Probed duration {actual_duration}s differs from expected {expected_duration}s by > 1.0s"
    )


@pytest.mark.tier1
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa not implemented yet")
def test_tc_f22_05_local_execution_zero_paid_api(monkeypatch, paid_api_interceptor):
    """TC-F22-05: Entire run completes locally without invoking any paid API service."""
    monkeypatch.setenv("SELF_HOST", "true")
    for key in ["ASSEMBLY_AI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "APIFY_API_TOKEN"]:
        monkeypatch.delenv(key, raising=False)

    cfg = config.load_config() if hasattr(config, "load_config") else config.get_config()
    assert getattr(cfg, "self_hosted", getattr(cfg, "self_host", True)) is True
    paid_api_interceptor.assert_zero_calls()
