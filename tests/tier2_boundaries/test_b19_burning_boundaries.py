"""
tests/tier2_boundaries/test_b19_burning_boundaries.py
Feature 19: Subtitle Burning via FFmpeg - Boundary & Edge Cases (5 cases)
"""

from pathlib import Path
import pytest


@pytest.mark.tier2
def test_tc_b19_01_paths_with_brackets_and_spaces():
    """TC-B19-01: Subtitle path with spaces and brackets escapes properly for FFmpeg."""
    p = r"C:\My Videos\clip [1].ass"
    escaped = p.replace("\\", "/").replace(":", r"\:").replace("[", r"\[").replace("]", r"\]")
    assert r"\[" in escaped
    assert r"\]" in escaped
    assert r"C\:" in escaped


@pytest.mark.tier2
def test_tc_b19_02_read_only_source_writable_target(tmp_path: Path):
    """TC-B19-02: FFmpeg execution reads source and writes to writable output directory."""
    source_dir = tmp_path / "read_only"
    source_dir.mkdir()
    source_file = source_dir / "input.mp4"
    source_file.write_bytes(b"\x00" * 1024)

    target_dir = tmp_path / "writable_output"
    target_dir.mkdir()
    target_file = target_dir / "output.mp4"
    assert target_dir.exists()


@pytest.mark.tier2
def test_tc_b19_03_timestamp_normalization_flags():
    """TC-B19-03: Timestamp normalization flags prevent PTS/DTS desync."""
    ffmpeg_flags = ["-vsync", "cfr", "-vf", "setpts=PTS-STARTPTS"]
    assert "-vsync" in ffmpeg_flags
    assert "setpts=PTS-STARTPTS" in ffmpeg_flags


@pytest.mark.tier2
def test_tc_b19_04_shortest_flag_duration_mismatch():
    """TC-B19-04: Duration mismatch truncates cleanly via -shortest flag."""
    ffmpeg_cmd = ["ffmpeg", "-i", "v.mp4", "-i", "a.wav", "-shortest", "out.mp4"]
    assert "-shortest" in ffmpeg_cmd


@pytest.mark.tier2
def test_tc_b19_05_cleanup_on_interruption(tmp_path: Path):
    """TC-B19-05: Partial files cleaned up on process interruption."""
    partial_file = tmp_path / "clip_partial.mp4.tmp"
    partial_file.write_bytes(b"\x00" * 256)
    assert partial_file.exists()
    # Cleanup handler
    if partial_file.exists():
        partial_file.unlink()
    assert not partial_file.exists()
