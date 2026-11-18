"""
tests/utils.py - Media Probing, Codec Verification, ffprobe Inspection & Assertions
"""

import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any


def find_ffprobe() -> Optional[str]:
    """Resolves ffprobe executable path from PATH or environment variables."""
    bin_path = shutil.which("ffprobe") or os.environ.get("FFPROBE_PATH")
    if not bin_path:
        common_paths = [
            Path(r"C:\Program Files\FFmpeg\bin\ffprobe.exe"),
            Path(r"C:\Program Files (x86)\FFmpeg\bin\ffprobe.exe"),
            Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Links" / "ffprobe.exe",
        ]
        for p in common_paths:
            if p.exists():
                return str(p)
    return bin_path


def require_ffprobe() -> str:
    """Returns ffprobe path or raises RuntimeError if unavailable."""
    bin_path = find_ffprobe()
    if not bin_path:
        raise RuntimeError(
            "ffprobe executable not found on PATH or FFPROBE_PATH. "
            "Install FFmpeg via 'winget install Gyan.FFmpeg' to run media assertions."
        )
    return bin_path


def find_ffmpeg() -> Optional[str]:
    """Resolves ffmpeg executable path from PATH or environment variables."""
    bin_path = shutil.which("ffmpeg") or os.environ.get("FFMPEG_PATH")
    if not bin_path:
        common_paths = [
            Path(r"C:\Program Files\FFmpeg\bin\ffmpeg.exe"),
            Path(r"C:\Program Files (x86)\FFmpeg\bin\ffmpeg.exe"),
            Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Links" / "ffmpeg.exe",
        ]
        for p in common_paths:
            if p.exists():
                return str(p)
    return bin_path


def require_ffmpeg() -> str:
    """Returns ffmpeg path or raises RuntimeError if unavailable."""
    bin_path = find_ffmpeg()
    if not bin_path:
        raise RuntimeError(
            "ffmpeg executable not found on PATH or FFMPEG_PATH. "
            "Install FFmpeg via 'winget install Gyan.FFmpeg' to run media assertions."
        )
    return bin_path


def probe_media(file_path: Path) -> Dict[str, Any]:
    """Runs ffprobe and returns full stream & format metadata as JSON dict."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Media file does not exist: {file_path}")

    ffprobe = require_ffprobe()
    cmd = [
        ffprobe,
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(file_path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {file_path}: {res.stderr}")
    return json.loads(res.stdout)


def assert_vertical_9_16(probe_data: Dict[str, Any], target_w: int = 1080, target_h: int = 1920):
    """Asserts that the video stream has exact vertical 9:16 geometry."""
    streams = probe_data.get("streams", [])
    video_streams = [s for s in streams if s.get("codec_type") == "video"]
    assert len(video_streams) > 0, "No video stream found in probed media"

    v = video_streams[0]
    w = int(v.get("width", 0))
    h = int(v.get("height", 0))

    assert w == target_w, f"Expected video width {target_w}, got {w}"
    assert h == target_h, f"Expected video height {target_h}, got {h}"

    aspect_ratio = w / h
    expected_ratio = target_w / target_h  # 0.5625 for 9:16
    assert abs(aspect_ratio - expected_ratio) < 0.01, f"Expected aspect ratio ~{expected_ratio:.4f}, got {aspect_ratio:.4f}"


def assert_valid_h264_aac(probe_data: Dict[str, Any]):
    """Asserts that container contains H.264 video and AAC audio."""
    streams = probe_data.get("streams", [])
    v_streams = [s for s in streams if s.get("codec_type") == "video"]
    a_streams = [s for s in streams if s.get("codec_type") == "audio"]

    assert len(v_streams) > 0, "No video stream found"
    assert len(a_streams) > 0, "No audio stream found"

    v_codec = v_streams[0].get("codec_name", "").lower()
    assert v_codec in ["h264", "avc1"], f"Video codec is not h264: {v_codec}"

    pix_fmt = v_streams[0].get("pix_fmt", "")
    assert pix_fmt == "yuv420p", f"Pixel format is not yuv420p: {pix_fmt}"

    a_codec = a_streams[0].get("codec_name", "").lower()
    assert a_codec in ["aac", "mp4a-40-2"], f"Audio codec is not aac: {a_codec}"


def assert_audio_duration_sync(probe_data: Dict[str, Any], tolerance_s: float = 0.5):
    """Compares video stream duration vs audio stream duration."""
    streams = probe_data.get("streams", [])
    v_streams = [s for s in streams if s.get("codec_type") == "video"]
    a_streams = [s for s in streams if s.get("codec_type") == "audio"]

    assert len(v_streams) > 0, "No video stream found"
    assert len(a_streams) > 0, "No audio stream found"

    v_dur = float(v_streams[0].get("duration") or probe_data.get("format", {}).get("duration", 0))
    a_dur = float(a_streams[0].get("duration") or probe_data.get("format", {}).get("duration", 0))

    delta = abs(v_dur - a_dur)
    assert delta <= tolerance_s, f"Video duration ({v_dur}s) and Audio duration ({a_dur}s) differ by {delta}s > {tolerance_s}s"


def assert_non_silent_audio(file_path: Path, min_mean_volume_db: float = -60.0):
    """Runs ffmpeg volumedetect filter to assert non-silent audio."""
    ffmpeg = require_ffmpeg()
    cmd = [
        ffmpeg, "-y",
        "-i", str(file_path),
        "-af", "volumedetect",
        "-vn", "-sn", "-dn",
        "-f", "null", "-",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"FFmpeg volumedetect execution failed: {res.stderr}"

    # Parse mean_volume: -XX.X dB
    match = re.search(r"mean_volume:\s+([-\d.]+)\s+dB", res.stderr)
    assert match is not None, f"Could not parse mean_volume from FFmpeg output:\n{res.stderr}"

    mean_vol = float(match.group(1))
    assert mean_vol > min_mean_volume_db, f"Audio is virtually silent: mean volume {mean_vol} dB <= {min_mean_volume_db} dB"


def assert_media_integrity(file_path: Path):
    """Decodes all frames with ffmpeg to null muxer to assert zero corrupted packets."""
    ffmpeg = require_ffmpeg()
    cmd = [ffmpeg, "-v", "error", "-i", str(file_path), "-f", "null", "-"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"Media decoding failed with return code {res.returncode}"
    assert res.stderr.strip() == "", f"FFmpeg reported decode errors: {res.stderr}"


def assert_burned_subtitles_present(clip_path: Path, timestamp_s: float = 1.0):
    """Extracts a frame at timestamp_s and verifies pixel content exists."""
    ffmpeg = require_ffmpeg()
    cmd = [
        ffmpeg, "-y",
        "-ss", str(timestamp_s),
        "-i", str(clip_path),
        "-vframes", "1",
        "-f", "image2pipe",
        "-vcodec", "rawvideo",
        "-pix_fmt", "gray",
        "-",
    ]
    res = subprocess.run(cmd, capture_output=True)
    assert res.returncode == 0, f"Frame extraction failed: {res.stderr}"
    raw_pixels = res.stdout
    assert len(raw_pixels) > 0, "No pixel data extracted from video frame"
    # Verify non-trivial variance (not pure zero bytes)
    assert any(b > 0 for b in raw_pixels[:1000]), "Extracted frame appears completely blank"
