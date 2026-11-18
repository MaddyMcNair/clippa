"""
tests/fixtures/synthetic.py - High-Performance Procedural Media Fixture Generator & Mock Schemas
"""

import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Dict, Any


@dataclass
class Word:
    text: str
    start: int  # milliseconds
    end: int    # milliseconds
    confidence: float = 1.0


@dataclass
class Utterance:
    text: str
    start: int  # milliseconds
    end: int    # milliseconds
    words: List[Word] = field(default_factory=list)


@dataclass
class TranscriptData:
    text: str
    language: str
    words: List[Word]
    utterances: List[Utterance] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": 3,
            "text": self.text,
            "language": self.language,
            "language_code": self.language,
            "words": [asdict(w) for w in self.words],
            "utterances": [asdict(u) for u in self.utterances],
        }


@dataclass
class ViralSegment:
    start_time_ms: int
    end_time_ms: int
    hook_title: str
    hook_score: int
    engagement_score: int
    value_score: int
    shareability_score: int
    total_score: int
    reasoning: str
    hook_start_time_ms: Optional[int] = None
    hook_end_time_ms: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


def find_ffmpeg() -> Optional[str]:
    """Resolves ffmpeg executable path from PATH or environment variables."""
    bin_path = shutil.which("ffmpeg") or os.environ.get("FFMPEG_PATH")
    if not bin_path:
        # Check standard Windows paths
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
            "FFmpeg executable not found on PATH, FFMPEG_PATH, or standard locations. "
            "Install FFmpeg via 'winget install Gyan.FFmpeg' to run live media generation."
        )
    return bin_path


def generate_synthetic_mp4(
    output_path: Path,
    duration_s: float = 3.0,
    width: int = 1920,
    height: int = 1080,
    fps: int = 30,
    audio_freq: int = 1000,
    sample_rate: int = 16000,
) -> Path:
    """Generates standard H.264 / AAC MP4 video with sine audio."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = require_ffmpeg()

    cmd = [
        ffmpeg, "-y",
        "-f", "lavfi", "-i", f"testsrc2=size={width}x{height}:rate={fps}",
        "-f", "lavfi", "-i", f"sine=frequency={audio_freq}:sample_rate={sample_rate}",
        "-t", str(duration_s),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        str(output_path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"FFmpeg synthetic media generation failed: {res.stderr}")
    return output_path


def generate_moving_face_mp4(
    output_path: Path,
    duration_s: float = 4.0,
    width: int = 1920,
    height: int = 1080,
) -> Path:
    """Simulates moving talking-head box across horizontal landscape axis."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = require_ffmpeg()

    # Box moves from left (x=200) to right (x=1500) over 4 seconds
    drawbox_filter = "drawbox=x='200+t*325':y='350':w='240':h='300':color=yellow@1.0:t=fill"

    cmd = [
        ffmpeg, "-y",
        "-f", "lavfi", "-i", f"color=c=darkblue:size={width}x{height}:rate=30",
        "-vf", drawbox_filter,
        "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=16000",
        "-t", str(duration_s),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-movflags", "+faststart",
        str(output_path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"FFmpeg moving face generation failed: {res.stderr}")
    return output_path


def generate_silent_mp4(output_path: Path, duration_s: float = 3.0, width: int = 1920, height: int = 1080) -> Path:
    """Generates video with silent audio track (anullsrc)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = require_ffmpeg()

    cmd = [
        ffmpeg, "-y",
        "-f", "lavfi", "-i", f"testsrc2=size={width}x{height}:rate=30",
        "-f", "lavfi", "-i", "anullsrc=channel_layout=mono:sample_rate=16000",
        "-t", str(duration_s),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        str(output_path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"FFmpeg silent media generation failed: {res.stderr}")
    return output_path


def generate_video_no_audio(output_path: Path, duration_s: float = 3.0, width: int = 1920, height: int = 1080) -> Path:
    """Generates video stream only, without any audio stream."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = require_ffmpeg()

    cmd = [
        ffmpeg, "-y",
        "-f", "lavfi", "-i", f"testsrc2=size={width}x{height}:rate=30",
        "-t", str(duration_s),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-an",
        str(output_path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"FFmpeg no-audio video generation failed: {res.stderr}")
    return output_path


def generate_audio_only(output_path: Path, duration_s: float = 3.0, format: str = "mp3", sample_rate: int = 16000) -> Path:
    """Generates pure audio file with sine wave."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = require_ffmpeg()

    codec = "libmp3lame" if format == "mp3" else "pcm_s16le"
    cmd = [
        ffmpeg, "-y",
        "-f", "lavfi", "-i", f"sine=frequency=1000:sample_rate={sample_rate}",
        "-t", str(duration_s),
        "-ac", "1",
        "-c:a", codec,
        str(output_path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"FFmpeg audio-only generation failed: {res.stderr}")
    return output_path


def generate_aspect_ratio_mp4(output_path: Path, aspect: str = "9:16", duration_s: float = 3.0) -> Path:
    """Generates synthetic video in target aspect ratio (9:16, 4:3, 1:1, 21:9)."""
    dimensions = {
        "9:16": (1080, 1920),
        "4:3": (1440, 1080),
        "1:1": (1080, 1080),
        "21:9": (2560, 1080),
        "16:9": (1920, 1080),
    }
    w, h = dimensions.get(aspect, (1920, 1080))
    return generate_synthetic_mp4(output_path, duration_s=duration_s, width=w, height=h)


def generate_corrupt_media(output_path: Path, corrupt_type: str = "truncated") -> Path:
    """Generates corrupted media file for error handling verification."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if corrupt_type == "empty":
        output_path.write_bytes(b"")
    elif corrupt_type == "truncated":
        # Fake MP4 header truncated abruptly
        output_path.write_bytes(b"\x00\x00\x00\x18ftypisom\x00\x00\x02\x00")
    else:  # garbage
        output_path.write_bytes(b"NOT_A_VALID_MEDIA_FILE_CONTENT_GARBAGE_BYTES_123456789")
    return output_path


def make_sample_transcript(duration_ms: int = 3000, text: Optional[str] = None) -> TranscriptData:
    """Builds a deterministic sample transcript with accurate millisecond timings."""
    if text is None:
        words_list = [
            Word("This", 100, 350, 0.99),
            Word("is", 360, 500, 0.98),
            Word("an", 510, 650, 0.95),
            Word("automated", 660, 1100, 0.97),
            Word("synthetic", 1120, 1600, 0.99),
            Word("speech", 1610, 1950, 0.96),
            Word("transcript", 1960, 2400, 0.98),
            Word("for", 2410, 2600, 0.95),
            Word("viral", 2610, 2850, 0.99),
            Word("clipping", 2860, max(3000, duration_ms), 0.98),
        ]
        full_text = "This is an automated synthetic speech transcript for viral clipping."
    else:
        tokens = text.split()
        words_list = []
        interval = max(50, duration_ms // max(1, len(tokens)))
        cur = 0
        for token in tokens:
            nxt = min(duration_ms, cur + interval - 20)
            words_list.append(Word(token, cur, nxt, 0.98))
            cur += interval
        full_text = text

    utterance = Utterance(
        text=full_text,
        start=words_list[0].start if words_list else 0,
        end=words_list[-1].end if words_list else duration_ms,
        words=words_list,
    )

    return TranscriptData(
        text=full_text,
        language="en",
        words=words_list,
        utterances=[utterance],
    )


def make_sample_viral_segment(
    start_ms: int = 0,
    end_ms: int = 30000,
    hook_title: str = "The Viral Secret You Never Knew",
    hook_score: int = 24,
    engagement_score: int = 23,
    value_score: int = 22,
    shareability_score: int = 24,
) -> ViralSegment:
    """Builds a deterministic ViralSegment object satisfying all scoring and duration bounds."""
    total = hook_score + engagement_score + value_score + shareability_score
    return ViralSegment(
        start_time_ms=start_ms,
        end_time_ms=end_ms,
        hook_title=hook_title,
        hook_score=hook_score,
        engagement_score=engagement_score,
        value_score=value_score,
        shareability_score=shareability_score,
        total_score=total,
        reasoning="High hook score with immediate value proposition and emotional curiosity.",
        hook_start_time_ms=None,
        hook_end_time_ms=None,
    )


def make_sample_viral_segments(count: int = 3) -> List[ViralSegment]:
    """Builds multiple sorted viral segments."""
    segments = []
    base_titles = [
        "The #1 Secret To Massive Productivity",
        "Why Most People Fail In Business",
        "The Psychological Trick Everyone Uses",
    ]
    for i in range(count):
        title = base_titles[i % len(base_titles)]
        start_ms = i * 35000
        end_ms = start_ms + 30000
        score_base = 24 - i
        seg = make_sample_viral_segment(
            start_ms=start_ms,
            end_ms=end_ms,
            hook_title=title,
            hook_score=score_base,
            engagement_score=score_base,
            value_score=score_base,
            shareability_score=score_base,
        )
        segments.append(seg)
    return segments


def make_sample_ass_content(hook_title: Optional[str] = "The Viral Hook", template: str = "default") -> str:
    """Builds a valid ASS subtitle string for testing parser and burning."""
    content = [
        "[Script Info]",
        "ScriptType: v4.00+",
        "PlayResX: 1080",
        "PlayResY: 1920",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        "Style: Default,Arial,60,&H00FFFFFF,&H0000FFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,3,0,2,40,40,240,1",
    ]
    if hook_title:
        content.append(
            "Style: Hook,Arial,72,&H00FFFFFF,&H0000FFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,4,0,8,40,40,160,1"
        )
    content.extend([
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ])
    if hook_title:
        content.append(f"Dialogue: 1,0:00:00.00,0:00:03.50,Hook,,0,0,0,,{{\\fad(200,200)}}{hook_title}")
    content.append("Dialogue: 0,0:00:00.10,0:00:01.50,Default,,0,0,0,,{\\k25}This {\\k35}is {\\k40}viral")
    return "\n".join(content) + "\n"
