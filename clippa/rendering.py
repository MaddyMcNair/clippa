"""
clippa/rendering.py - FFmpeg Video Rendering & Subtitle Burning

Feature 19: Subtitle Burning via FFmpeg (libass)
Seam: Combines 9:16 crop filter (Feature 14) and ASS subtitles (Feature 15) in a single pass.
"""

from __future__ import annotations

import functools
import os
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import List, Optional, Union

# Bundled fonts directory
FONTS_DIR = Path(__file__).parent.parent / "backend" / "fonts"


@functools.lru_cache(maxsize=1)
def is_nvenc_available() -> bool:
    """
    Checks if FFmpeg supports NVIDIA NVENC hardware acceleration (h264_nvenc).
    Cached so it only runs once per process.
    """
    try:
        ffmpeg_bin = find_ffmpeg()
        res = subprocess.run(
            [
                ffmpeg_bin,
                "-f", "lavfi",
                "-i", "testsrc=duration=1:size=320x240:rate=30",
                "-pix_fmt", "yuv420p",
                "-c:v", "h264_nvenc",
                "-f", "null",
                "-",
            ],
            capture_output=True,
            timeout=5.0,
        )
        return res.returncode == 0
    except Exception:
        return False


# ==============================================================================
# Exceptions
# ==============================================================================

class RenderingError(Exception):
    """Base exception for all rendering failures in clippa."""
    pass


class FFmpegExecutionError(RenderingError):
    """Raised when an FFmpeg process exits with a non-zero return code."""

    def __init__(self, message: str, returncode: int = 1, cmd: Optional[List[str]] = None, stderr: str = ""):
        super().__init__(message)
        self.returncode = returncode
        self.cmd = cmd or []
        self.stderr = stderr


class SubtitleBurnError(RenderingError):
    """Raised when subtitle burning or libass parsing fails."""
    pass


class MediaNotFoundError(RenderingError, FileNotFoundError):
    """Raised when input media file or subtitle file does not exist."""
    pass


# ==============================================================================
# Utility & Path Escaping
# ==============================================================================

def find_ffmpeg() -> str:
    """Resolves ffmpeg executable path from PATH or common installation locations."""
    bin_path = shutil.which("ffmpeg") or os.environ.get("FFMPEG_PATH")
    if not bin_path:
        common_paths = [
            Path(r"C:\Program Files\FFmpeg\bin\ffmpeg.exe"),
            Path(r"C:\Program Files (x86)\FFmpeg\bin\ffmpeg.exe"),
            Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Links" / "ffmpeg.exe",
            Path(r"C:\ffmpeg\bin\ffmpeg.exe"),
        ]
        for p in common_paths:
            if p.exists():
                return str(p)
    return bin_path or "ffmpeg"


def escape_filter_path(path: Union[str, Path]) -> str:
    """
    Escapes a filesystem path for use inside FFmpeg filter options (like libass/subtitles).
    Converts Windows backslashes to forward slashes and escapes colons and brackets.
    """
    p_str = str(path).replace("\\", "/")
    # Escape colons (e.g. C: -> C\:)
    p_str = p_str.replace(":", r"\:")
    # Escape square brackets (e.g. [ -> \[, ] -> \])
    p_str = p_str.replace("[", r"\[").replace("]", r"\]")
    # Escape single quotes
    p_str = p_str.replace("'", r"\'")
    return p_str


escape_path_for_ffmpeg = escape_filter_path


# ==============================================================================
# Filter & Command Builder
# ==============================================================================

def build_ffmpeg_render_args(
    source_video: Union[str, Path] = "input.mp4",
    output_file: Union[str, Path] = "output.mp4",
    start_ms: Optional[int] = None,
    end_ms: Optional[int] = None,
    crop_filter: Optional[str] = None,
    ass_path: Optional[Union[str, Path]] = None,
    fonts_dir: Optional[Union[str, Path]] = None,
    crf: int = 22,
    preset: str = "fast",
    audio_bitrate: str = "192k",
    audio_sample_rate: int = 48000,
    use_nvenc: bool = False,
) -> List[str]:
    """
    Builds the complete FFmpeg command line arguments list for rendering.
    """
    cmd = [find_ffmpeg(), "-y"]

    # Input Seeking / Trimming
    if start_ms is not None:
        start_s = max(0.0, float(start_ms) / 1000.0)
        cmd.extend(["-ss", f"{start_s:.3f}"])

    if start_ms is not None and end_ms is not None:
        duration_s = max(0.0, (float(end_ms) - float(start_ms)) / 1000.0)
        cmd.extend(["-t", f"{duration_s:.3f}"])
    elif end_ms is not None:
        end_s = max(0.0, float(end_ms) / 1000.0)
        cmd.extend(["-to", f"{end_s:.3f}"])

    cmd.extend(["-i", str(source_video)])

    # Video Filtergraph Construction
    filters: List[str] = []

    # PTS normalization to guarantee subtitle timing sync
    filters.append("setpts=PTS-STARTPTS")

    if crop_filter:
        filters.append(crop_filter.strip())

    if ass_path:
        ass_esc = escape_filter_path(ass_path)
        f_dir = fonts_dir or (FONTS_DIR if FONTS_DIR.exists() else None)
        if f_dir and Path(f_dir).exists():
            fonts_esc = escape_filter_path(f_dir)
            filters.append(f"ass=filename='{ass_esc}':fontsdir='{fonts_esc}'")
        else:
            filters.append(f"ass=filename='{ass_esc}'")

    if filters:
        cmd.extend(["-vf", ",".join(filters)])

    # Video & Audio Encoding Parameters
    if use_nvenc:
        cmd.extend([
            "-c:v", "h264_nvenc",
            "-preset", "p4",
            "-cq", str(crf),
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", audio_bitrate,
            "-ar", str(audio_sample_rate),
            "-movflags", "+faststart",
            "-shortest",
            str(output_file),
        ])
    else:
        cmd.extend([
            "-c:v", "libx264",
            "-preset", preset,
            "-crf", str(crf),
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", audio_bitrate,
            "-ar", str(audio_sample_rate),
            "-movflags", "+faststart",
            "-shortest",
            str(output_file),
        ])

    return cmd


# ==============================================================================
# Primary Rendering Entry Point
# ==============================================================================

def render_clip(
    source_video: Union[str, Path],
    start_ms: int,
    end_ms: int,
    crop_filter: Optional[str] = None,
    ass_path: Optional[Union[str, Path]] = None,
    output_file: Union[str, Path] = "output.mp4",
    fonts_dir: Optional[Union[str, Path]] = None,
    crf: int = 22,
    preset: str = "fast",
    timeout: Optional[float] = 600.0,
    use_nvenc: bool = False,
) -> Path:
    """
    Renders a trimmed 9:16 vertical video clip with burned animated subtitles.

    Parameters:
        source_video: Path to input landscape/source video.
        start_ms: Integer millisecond clip start timestamp.
        end_ms: Integer millisecond clip end timestamp.
        crop_filter: Optional FFmpeg crop/scale expression (e.g. from clippa.reframing).
        ass_path: Optional Path to ASS subtitle file to burn into video frames.
        output_file: Destination path for rendered MP4.
        fonts_dir: Directory containing fonts for libass.
        crf: Constant Rate Factor (default 22).
        preset: x264 preset (default 'fast').
        timeout: Subprocess timeout in seconds.
        use_nvenc: Whether to use NVIDIA NVENC hardware acceleration (h264_nvenc).

    Returns:
        Path to verified rendered output MP4 file.
    """
    src = Path(source_video)
    if not src.exists():
        raise MediaNotFoundError(f"Source video file not found: {src}")

    if ass_path is not None:
        ass_p = Path(ass_path)
        if not ass_p.exists():
            raise MediaNotFoundError(f"Subtitle file not found: {ass_p}")

    out = Path(output_file).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    # Atomic temporary file in target directory
    tmp_out = out.with_name(f".{out.stem}_{os.getpid()}_{uuid.uuid4().hex[:6]}.tmp.mp4")

    # Auto-generate crop_filter if omitted
    if crop_filter is None:
        from clippa.reframing import build_crop_filter
        crop_filter = build_crop_filter(src, start_ms=start_ms, end_ms=end_ms)

    cmd = build_ffmpeg_render_args(
        source_video=src,
        output_file=tmp_out,
        start_ms=start_ms,
        end_ms=end_ms,
        crop_filter=crop_filter,
        ass_path=ass_path,
        fonts_dir=fonts_dir,
        crf=crf,
        preset=preset,
        use_nvenc=use_nvenc,
    )

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if proc.returncode != 0 and use_nvenc:
            # Graceful fallback to libx264 CPU encoding if NVENC fails
            cmd_fallback = build_ffmpeg_render_args(
                source_video=src,
                output_file=tmp_out,
                start_ms=start_ms,
                end_ms=end_ms,
                crop_filter=crop_filter,
                ass_path=ass_path,
                fonts_dir=fonts_dir,
                crf=crf,
                preset=preset,
                use_nvenc=False,
            )
            proc = subprocess.run(cmd_fallback, capture_output=True, text=True, timeout=timeout)

        if proc.returncode != 0:
            stderr = proc.stderr or ""
            if "Could not create a libass track" in stderr or "libass" in stderr.lower():
                raise SubtitleBurnError(f"libass subtitle burning failed: {stderr[-500:]}")
            raise FFmpegExecutionError(
                f"FFmpeg render command failed with exit code {proc.returncode}: {stderr[-500:]}",
                returncode=proc.returncode,
                cmd=cmd,
                stderr=stderr,
            )

        if not tmp_out.exists() or tmp_out.stat().st_size == 0:
            raise RenderingError(f"Rendered temporary file is missing or empty: {tmp_out}")

        # Atomic replacement with Windows retry loop for indexer / OneDrive locks
        replaced = False
        for attempt in range(5):
            try:
                if out.exists():
                    try:
                        out.unlink()
                    except OSError:
                        pass
                tmp_out.replace(out)
                replaced = True
                break
            except OSError:
                import time
                time.sleep(0.2 * (attempt + 1))

        if not replaced:
            import shutil
            try:
                shutil.copy2(str(tmp_out), str(out))
                try:
                    tmp_out.unlink()
                except OSError:
                    pass
            except Exception as copy_exc:
                raise RenderingError(f"Failed to finalize output file {out}: {copy_exc}") from copy_exc

        return out

    except Exception:
        try:
            tmp_out.unlink(missing_ok=True)
        except OSError:
            pass
        raise
