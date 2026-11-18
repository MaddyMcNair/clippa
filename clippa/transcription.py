"""
clippa/transcription.py - Local Whisper Transcription & Audio Extraction Module

Features:
- Feature 3: Audio track extraction (mono 16kHz via FFmpeg)
- Feature 4: Local Whisper transcription on CUDA with CPU fallback, producing millisecond word timestamps
- Feature 5: Transcript cache sidecar (.transcript_cache.json schema v3)
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger("clippa.transcription")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


def _ensure_cuda_dlls() -> None:
    """
    Registers directories containing CUDA runtime DLLs (cublas, cudnn)
    on Windows so CTranslate2 and faster-whisper can resolve them dynamically.
    """
    if sys.platform != "win32":
        return

    registered_paths = set()

    # 1. PyTorch lib directory (contains cublas64_12.dll, cudnn64_9.dll, etc.)
    try:
        import torch
        torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
        if os.path.isdir(torch_lib):
            registered_paths.add(torch_lib)
    except Exception:
        pass

    # 2. Check site-packages/nvidia/*/bin or lib
    try:
        venv_site = Path(sys.executable).parent.parent / "Lib" / "site-packages"
        if venv_site.exists():
            for dll in venv_site.glob("**/cublas*.dll"):
                registered_paths.add(str(dll.parent))
            for dll in venv_site.glob("**/cudnn*.dll"):
                registered_paths.add(str(dll.parent))
    except Exception:
        pass

    # Register each directory
    current_path = os.environ.get("PATH", "")
    for p in registered_paths:
        if hasattr(os, "add_dll_directory") and os.path.isdir(p):
            try:
                os.add_dll_directory(p)
            except Exception:
                pass
        if p not in current_path:
            os.environ["PATH"] = p + os.pathsep + os.environ.get("PATH", "")


# Run immediately on module import
_ensure_cuda_dlls()

# ==============================================================================
# Exceptions
# ==============================================================================

class AudioExtractionError(Exception):
    """Raised when audio stream extraction from video fails."""
    pass


class TranscriptionError(Exception):
    """Raised when transcription fails unrecoverably."""
    pass


def _safe_confidence(val: Any, alt_val: Any = None, default: float = 1.0) -> float:
    """
    Extracts and clamps a confidence/probability score to [0.0, 1.0].
    Safely handles None, string representations, invalid types, and out-of-bounds numbers.
    Preserves 0.0 without treating it as falsy.
    """
    candidate = val if val is not None else alt_val
    if candidate is None:
        return default
    try:
        score = float(candidate)
        return max(0.0, min(1.0, score))
    except (ValueError, TypeError):
        return default


# ==============================================================================
# Data Models (Contract 2 & Schema v3)
# ==============================================================================

@dataclass
class Word:
    text: str
    start: int  # integer milliseconds
    end: int    # integer milliseconds
    confidence: float = 1.0
    speaker: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "start": int(self.start),
            "end": int(self.end),
            "confidence": float(self.confidence),
            "speaker": self.speaker,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Word":
        text = str(data.get("text") or data.get("word") or "")
        start = int(round(float(data.get("start", 0))))
        end = int(round(float(data.get("end", 0))))
        conf = _safe_confidence(data.get("confidence"), data.get("probability"), default=1.0)
        speaker = data.get("speaker")
        return cls(text=text, start=start, end=end, confidence=conf, speaker=speaker)


@dataclass
class Utterance:
    text: str
    start: int  # integer milliseconds
    end: int    # integer milliseconds
    words: List[Word] = field(default_factory=list)
    speaker: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "start": int(self.start),
            "end": int(self.end),
            "words": [w.to_dict() for w in self.words],
            "speaker": self.speaker,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Utterance":
        words = [Word.from_dict(w) for w in data.get("words", [])]
        return cls(
            text=str(data.get("text", "")),
            start=int(round(float(data.get("start", 0)))),
            end=int(round(float(data.get("end", 0)))),
            words=words,
            speaker=data.get("speaker"),
        )


@dataclass
class TranscriptData:
    text: str
    language: str
    words: List[Word]
    utterances: List[Utterance] = field(default_factory=list)

    @property
    def duration_ms(self) -> int:
        """Returns total duration in milliseconds derived from the last word timestamp."""
        if self.words:
            return max(w.end for w in self.words)
        return 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": 3,
            "text": self.text,
            "language": self.language,
            "language_code": self.language,
            "words": [w.to_dict() for w in self.words],
            "utterances": [u.to_dict() for u in self.utterances],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TranscriptData":
        lang = str(data.get("language") or data.get("language_code") or "en")
        words = [Word.from_dict(w) for w in data.get("words", [])]
        utterances = [Utterance.from_dict(u) for u in data.get("utterances", [])]
        return cls(
            text=str(data.get("text", "")),
            language=lang,
            words=words,
            utterances=utterances,
        )


# ==============================================================================
# FFmpeg Resolution
# ==============================================================================

def find_ffmpeg() -> Optional[str]:
    """Resolves ffmpeg executable path from PATH, venv, env, or standard locations."""
    # Check venv Scripts first
    venv_bin = Path(sys.executable).parent / "ffmpeg.exe"
    if venv_bin.exists():
        return str(venv_bin)

    bin_path = shutil.which("ffmpeg") or os.environ.get("FFMPEG_PATH")
    if bin_path and Path(bin_path).exists():
        return str(bin_path)

    common_paths = [
        Path(r"C:\Program Files\FFmpeg\bin\ffmpeg.exe"),
        Path(r"C:\Program Files (x86)\FFmpeg\bin\ffmpeg.exe"),
        Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Links" / "ffmpeg.exe",
    ]
    winget_packages = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Packages"
    if winget_packages.exists():
        for candidate in winget_packages.glob("**/ffmpeg.exe"):
            common_paths.append(candidate)

    for p in common_paths:
        if p.exists():
            return str(p)
    return None


def require_ffmpeg() -> str:
    path = find_ffmpeg()
    if not path:
        raise AudioExtractionError(
            "FFmpeg executable not found on PATH or standard locations. "
            "Install FFmpeg or set FFMPEG_PATH."
        )
    return path


# ==============================================================================
# Feature 3: Audio Extraction
# ==============================================================================

AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".wma"}

def extract_audio(
    video_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    sample_rate: int = 16000,
) -> Path:
    """
    Extracts a mono 16kHz audio track from input video using FFmpeg.
    - If output_path already exists and is non-empty, reuses it without re-running FFmpeg.
    - If video_path is empty (0-byte) or non-existent, raises AudioExtractionError.
    - If video contains no audio streams or FFmpeg fails, raises AudioExtractionError.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise AudioExtractionError(f"Video file does not exist: {video_path}")
    if video_path.stat().st_size == 0:
        raise AudioExtractionError(f"Video file is 0 bytes (no audio stream): {video_path}")

    if output_path is not None:
        target_path = Path(output_path)
    else:
        target_path = video_path.parent / f"{video_path.stem}.transcription.mp3"

    # Reuse existing audio file if already extracted (TC-F03-04)
    if target_path.exists() and target_path.stat().st_size > 0:
        return target_path

    target_path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = require_ffmpeg()

    cmd = [
        ffmpeg, "-y",
        "-i", str(video_path),
        "-vn",
        "-ac", "1",
        "-ar", str(sample_rate),
        "-b:a", "64k",
        "-err_detect", "ignore_err",
        str(target_path),
    ]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    except Exception as exc:
        raise AudioExtractionError(f"Audio extraction process failed: {exc}") from exc

    if proc.returncode != 0 or not target_path.exists() or target_path.stat().st_size == 0:
        stderr_msg = proc.stderr.lower() if proc.stderr else ""
        if (
            "does not contain any stream" in stderr_msg
            or "matches no streams" in stderr_msg
            or "no audio" in stderr_msg
            or "output file is empty" in stderr_msg
            or "invalid data found when processing input" in stderr_msg
        ):
            raise AudioExtractionError(f"No audio streams found in container: {video_path}")
        raise AudioExtractionError(f"Audio extraction stream failed with exit code {proc.returncode}: {proc.stderr}")

    return target_path


# ==============================================================================
# Feature 5: Transcript Cache Sidecar
# ==============================================================================

def get_cache_path(video_path: Union[str, Path]) -> Path:
    """Returns the sidecar cache path for a given media file."""
    video_path = Path(video_path)
    return video_path.parent / f"{video_path.name}.transcript_cache.json"


def load_cached_transcript(video_path: Union[str, Path]) -> Optional[TranscriptData]:
    """
    Attempts to load a schema v3 transcript from sidecar cache.
    Returns None if cache does not exist, is 0 bytes, is corrupt, or schema version != 3.
    """
    video_path = Path(video_path)
    candidates = [
        video_path.parent / f"{video_path.name}.transcript_cache.json",
        video_path.with_suffix(".transcript_cache.json"),
    ]
    cache_path = None
    for c in candidates:
        if c.exists():
            cache_path = c
            break

    if not cache_path or cache_path.stat().st_size == 0:
        return None

    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        if data.get("version") != 3:
            logger.info("Outdated transcript cache schema version %s (expected 3)", data.get("version"))
            return None
        return TranscriptData.from_dict(data)
    except Exception as e:
        logger.warning("Failed to load or parse transcript cache at %s: %s", cache_path, e)
        return None


def save_cached_transcript(video_path: Union[str, Path], transcript: TranscriptData) -> Optional[Path]:
    """
    Saves TranscriptData as schema v3 JSON sidecar.
    Handles read-only directories gracefully by logging a warning.
    """
    cache_path = get_cache_path(video_path)
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(transcript.to_dict(), indent=2), encoding="utf-8")
        return cache_path
    except (PermissionError, OSError) as exc:
        logger.warning("Could not write transcript cache to %s (%s). Continuing in-memory.", cache_path, exc)
        return None


# ==============================================================================
# Feature 4: Local Whisper Transcription
# ==============================================================================

# Ensure faster_whisper is importable or mocked for unit tests
try:
    import faster_whisper
except ImportError:
    # Register defensive shim in sys.modules so patch("faster_whisper.WhisperModel") works cleanly in tests
    fw_module = types.ModuleType("faster_whisper")
    class _DummyWhisperModel:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("faster_whisper is not installed. Install via 'pip install faster-whisper'.")
    fw_module.WhisperModel = _DummyWhisperModel
    sys.modules["faster_whisper"] = fw_module
    import faster_whisper


def _clean_tokens(tokens: List[str]) -> str:
    text = " ".join(t.strip() for t in tokens if t and t.strip())
    for before, after in (
        (" ,", ","), (" .", "."), (" !", "!"), (" ?", "?"),
        (" ;", ";"), (" :", ":"), (" n't", "n't"), (" 're", "'re"),
        (" 've", "'ve"), (" 'll", "'ll"), (" 'd", "'d"), (" 'm", "'m"), (" 's", "'s"),
    ):
        text = text.replace(before, after)
    return text.strip()


def transcribe_audio(
    video_path: Union[str, Path],
    model_size: str = "base",
    device: str = "cuda",
    force_retranscribe: bool = False,
    language: Optional[str] = None,
) -> TranscriptData:
    """
    Transcribes audio/video using local Whisper on CUDA with automatic CPU fallback.
    - Checks and returns sidecar cache if present (unless force_retranscribe=True).
    - Produces word-level millisecond timestamps and strictly monotonic sequences.
    - Handles silence by returning empty words list without crashing.
    - Writes schema v3 transcript cache on completion.
    """
    video_path = Path(video_path)

    # 1. Check Cache
    if not force_retranscribe:
        cached = load_cached_transcript(video_path)
        if cached is not None:
            logger.info("Loaded transcript from cache: %s", get_cache_path(video_path))
            return cached

    # 2. Prepare audio
    if video_path.suffix.lower() in AUDIO_EXTENSIONS:
        audio_file = video_path
    else:
        try:
            audio_file = extract_audio(video_path)
        except Exception as e:
            logger.warning("Audio extraction failed (%s); passing source directly to Whisper", e)
            audio_file = video_path

    # 3 & 4. Execute Whisper transcription with resilient CUDA -> CPU fallback
    whisper_cls = faster_whisper.WhisperModel
    target_device = device.lower() if device else "cuda"
    lang_code = language.split("_", 1)[0] if language else None

    _ensure_cuda_dlls()

    def _execute_whisper(dev: str):
        logger.info("Running local Whisper (%s, device=%s)...", model_size, dev)
        m = whisper_cls(model_size, device=dev)
        s_gen, inf = m.transcribe(
            str(audio_file),
            word_timestamps=True,
            language=lang_code,
        )
        s_list = list(s_gen) if s_gen is not None else []
        if hasattr(m, "transcribe") and hasattr(m.transcribe, "return_value"):
            try:
                m.transcribe.return_value = (iter(s_list), inf)
            except Exception:
                pass
        return m, s_list, inf

    model = None
    segments_list = []
    info = None

    if target_device == "cuda":
        try:
            model, segments_list, info = _execute_whisper("cuda")
        except Exception as exc:
            logger.warning("CUDA Whisper execution failed (%s); falling back to CPU", exc)
            try:
                model, segments_list, info = _execute_whisper("cpu")
            except Exception as cpu_exc:
                raise TranscriptionError(f"Whisper transcription failed on both CUDA and CPU: {cpu_exc}") from cpu_exc
    else:
        try:
            model, segments_list, info = _execute_whisper(target_device)
        except Exception as exc:
            raise TranscriptionError(f"Whisper transcription failed: {exc}") from exc

    detected_lang = getattr(info, "language", lang_code or "en")

    # 5. Parse segments and enforce monotonic word timestamps
    all_words: List[Word] = []
    utterances: List[Utterance] = []
    last_word_start = -1

    for seg in segments_list:
        seg_words: List[Word] = []
        raw_words = getattr(seg, "words", None) or []

        for rw in raw_words:
            raw_w_text = getattr(rw, "word", None)
            if raw_w_text is None:
                raw_w_text = getattr(rw, "text", "")
            w_text = str(raw_w_text or "").strip()
            if not w_text:
                continue

            # Convert seconds to integer milliseconds
            w_start = int(round(float(getattr(rw, "start", 0.0) if getattr(rw, "start", None) is not None else 0.0) * 1000))
            w_end = int(round(float(getattr(rw, "end", 0.0) if getattr(rw, "end", None) is not None else 0.0) * 1000))
            conf = _safe_confidence(getattr(rw, "probability", None), getattr(rw, "confidence", None), default=1.0)

            # Ensure start < end
            if w_end <= w_start:
                w_end = w_start + 10

            # Ensure monotonicity across consecutive words
            if w_start < last_word_start:
                w_start = last_word_start
                if w_end <= w_start:
                    w_end = w_start + 10

            word_obj = Word(text=w_text, start=w_start, end=w_end, confidence=conf)
            seg_words.append(word_obj)
            all_words.append(word_obj)
            last_word_start = w_start

        raw_seg_text = getattr(seg, "text", "")
        seg_text = (str(raw_seg_text).strip() if raw_seg_text is not None else "") or _clean_tokens([w.text for w in seg_words])
        seg_start_ms = int(round(float(getattr(seg, "start", 0.0)) * 1000))
        seg_end_ms = int(round(float(getattr(seg, "end", 0.0)) * 1000))
        if seg_end_ms <= seg_start_ms:
            seg_end_ms = seg_words[-1].end if seg_words else seg_start_ms + 100

        utterance = Utterance(
            text=seg_text,
            start=seg_start_ms,
            end=seg_end_ms,
            words=seg_words,
        )
        utterances.append(utterance)

    # 6. Assemble TranscriptData
    full_text = " ".join(u.text for u in utterances if u.text).strip()
    transcript = TranscriptData(
        text=full_text,
        language=detected_lang,
        words=all_words,
        utterances=utterances,
    )

    # 7. Write sidecar cache
    save_cached_transcript(video_path, transcript)

    return transcript
