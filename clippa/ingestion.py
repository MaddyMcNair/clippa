"""
clippa/ingestion.py - YouTube Video Ingestion & Metadata Preflight Module

Features:
- Feature 1: YouTube video download via yt-dlp (best MP4 <=1080p, local file bypass)
- Feature 2: Video metadata preflight (title, duration, uploader, live detection, caching)
"""

from __future__ import annotations

import logging
import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from urllib.parse import parse_qs, unquote, urlparse
from urllib.request import url2pathname

import yt_dlp

logger = logging.getLogger("clippa.ingestion")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


# ==============================================================================
# Exceptions
# ==============================================================================

class VideoDownloadError(Exception):
    """Base exception for all video ingestion and download failures."""
    pass


class InvalidURLError(VideoDownloadError, ValueError):
    """Raised when URL is empty, whitespace, malformed, or missing a video ID."""
    pass


class VideoUnavailableError(VideoDownloadError):
    """Raised when video is deleted, private, or inaccessible."""
    pass


class VideoAgeRestrictedError(VideoDownloadError):
    """Raised when video is age-restricted and requires sign-in."""
    pass


class StreamDurationUnavailableError(VideoDownloadError):
    """Raised when video is an ongoing livestream with no finite duration."""
    pass


class VideoDurationExceededError(VideoDownloadError):
    """Raised when video duration exceeds max configured limit (e.g. 5400s)."""
    pass


class PreflightTimeoutError(VideoDownloadError, TimeoutError):
    """Raised on metadata preflight network timeout."""
    pass


# ==============================================================================
# Data Models (Feature 2)
# ==============================================================================

@dataclass
class VideoMetadata:
    """Metadata extracted during video preflight."""
    id: str
    title: str
    duration: float  # seconds
    uploader: Optional[str] = "Unknown"
    is_live: bool = False
    description: str = ""
    view_count: Optional[int] = None
    like_count: Optional[int] = None
    thumbnail: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    fps: Optional[float] = None
    ext: str = "mp4"
    heatmap: Optional[List[Dict[str, float]]] = None

    def __getitem__(self, key: str) -> Any:
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key)

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, value)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "duration": float(self.duration),
            "uploader": self.uploader,
            "is_live": self.is_live,
            "description": self.description,
            "view_count": self.view_count,
            "like_count": self.like_count,
            "thumbnail": self.thumbnail,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "ext": self.ext,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> VideoMetadata:
        return cls(
            id=str(data.get("id", "")),
            title=str(data.get("title", "")),
            duration=float(data.get("duration", 0.0) or 0.0),
            uploader=data.get("uploader") or data.get("channel") or "Unknown",
            is_live=bool(data.get("is_live", False)),
            description=str(data.get("description", "")),
            view_count=data.get("view_count"),
            like_count=data.get("like_count"),
            thumbnail=data.get("thumbnail"),
            width=data.get("width"),
            height=data.get("height"),
            fps=data.get("fps"),
            ext=str(data.get("ext", "mp4")),
        )


# ==============================================================================
# URL & Local Path Helpers
# ==============================================================================

_METADATA_CACHE: Dict[str, VideoMetadata] = {}


def clear_metadata_cache() -> None:
    """Clears the in-memory video metadata preflight cache."""
    _METADATA_CACHE.clear()


def resolve_local_video_path(path_or_url: Union[str, Path]) -> Optional[Path]:
    """
    Resolves an existing local file path or file:// URI to an absolute Path.
    Returns None if not an existing local file.
    """
    if isinstance(path_or_url, Path):
        return path_or_url.resolve() if path_or_url.is_file() else None

    if not isinstance(path_or_url, str) or not path_or_url.strip():
        return None

    raw = path_or_url.strip()
    if raw.startswith("file://"):
        try:
            parsed = urlparse(raw)
            p = Path(url2pathname(parsed.path))
            if p.is_file():
                return p.resolve()
        except Exception:
            pass

    try:
        p = Path(raw)
        if p.is_file():
            return p.resolve()
    except Exception:
        pass

    return None


def is_local_video(path_or_url: Union[str, Path]) -> bool:
    """Returns True if the path or URL refers to an existing local file."""
    return resolve_local_video_path(path_or_url) is not None


def extract_video_id(url: str) -> Optional[str]:
    """
    Extracts video ID from YouTube URL or returns None if invalid/absent.
    Supports standard 11-char IDs and synthetic test mock IDs.
    """
    if not isinstance(url, str) or not url.strip():
        return None

    raw = url.strip()
    if "://" not in raw:
        raw = f"https://{raw}"

    try:
        parsed = urlparse(raw)
        host = (parsed.hostname or "").lower().removeprefix("www.")
    except Exception:
        return None

    is_yt = (
        host in {"youtu.be", "youtube.com", "youtube-nocookie.com"}
        or host.endswith(".youtube.com")
        or host.endswith(".youtube-nocookie.com")
    )
    if not is_yt:
        return None

    parts = [p for p in parsed.path.split("/") if p]
    candidate: Optional[str] = None

    if host == "youtu.be":
        candidate = parts[0] if parts else None
    else:
        qs = parse_qs(parsed.query)
        v_list = qs.get("v", [])
        if v_list and v_list[0].strip():
            candidate = v_list[0].strip()
        elif len(parts) >= 2 and parts[0].lower() in {"embed", "shorts", "live", "v"}:
            candidate = parts[1].strip() if parts[1].strip() else None

    if candidate and len(candidate) >= 3:
        if re.match(r"^[A-Za-z0-9_-]+$", candidate):
            return candidate

    return None


def is_valid_youtube_url(url: str) -> bool:
    """Returns True if the given string is a valid YouTube URL."""
    return extract_video_id(url) is not None


def canonical_youtube_url(url: str) -> Optional[str]:
    """Returns standard https://www.youtube.com/watch?v=ID format or None."""
    vid = extract_video_id(url)
    return f"https://www.youtube.com/watch?v={vid}" if vid else None


def validate_video_duration(
    metadata_or_duration: Union[VideoMetadata, Dict[str, Any], float, int],
    max_duration: float = 5400.0,
    raise_on_error: bool = False,
) -> bool:
    """
    Validates that video duration is positive and within max_duration.
    """
    if isinstance(metadata_or_duration, (int, float)):
        dur = float(metadata_or_duration)
    elif isinstance(metadata_or_duration, VideoMetadata):
        dur = float(metadata_or_duration.duration)
    elif isinstance(metadata_or_duration, dict):
        dur = float(metadata_or_duration.get("duration", 0.0) or 0.0)
    else:
        dur = 0.0

    if dur <= 0:
        if raise_on_error:
            raise StreamDurationUnavailableError(f"Video duration is non-positive or unavailable ({dur}s).")
        return False

    if dur > max_duration:
        if raise_on_error:
            raise VideoDurationExceededError(
                f"Video duration ({dur:.1f}s) exceeds maximum allowed duration ({max_duration:.1f}s)."
            )
        return False

    return True


# ==============================================================================
# Feature 2: Video Metadata Preflight
# ==============================================================================

def _get_default_js_runtimes() -> Optional[Dict[str, Dict[str, str]]]:
    """Finds node or deno executable on host to enable JS extraction and YouTube heatmaps."""
    node_path = shutil.which("node") or shutil.which("deno")
    if not node_path:
        for candidate in [
            Path(r"C:\Program Files\nodejs\node.exe"),
            Path(r"C:\Program Files (x86)\nodejs\node.exe"),
            Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "node" / "node.exe",
        ]:
            if candidate.is_file():
                node_path = str(candidate)
                break
    if node_path:
        name = "deno" if "deno" in str(node_path).lower() else "node"
        return {name: {"path": str(node_path)}}
    return None


def get_video_metadata(
    url: str,
    timeout: int = 30,
    use_cache: bool = True,
) -> VideoMetadata:
    """
    Extracts video metadata via yt-dlp preflight without downloading payload.
    """
    if not isinstance(url, str) or not url.strip():
        raise InvalidURLError("URL cannot be empty or whitespace.")

    clean_url = url.strip()

    # 1. Check local file
    local_p = resolve_local_video_path(clean_url)
    if local_p is not None:
        return VideoMetadata(
            id=local_p.stem,
            title=local_p.stem,
            duration=0.0,
            uploader="Local",
            ext=local_p.suffix.lstrip(".") or "mp4",
        )

    # 2. Check cache
    if use_cache and clean_url in _METADATA_CACHE:
        return _METADATA_CACHE[clean_url]

    # 3. Validate URL
    vid_id = extract_video_id(clean_url)
    if not vid_id:
        raise InvalidURLError(f"Invalid or unsupported YouTube URL: '{url}'")

    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "extractaudio": False,
        "skip_download": True,
        "noplaylist": True,
        "socket_timeout": timeout,
        "nocheckcertificate": True,
        "http_headers": {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        },
    }
    js_rts = _get_default_js_runtimes()
    if js_rts:
        ydl_opts["js_runtimes"] = js_rts

    ydl = yt_dlp.YoutubeDL(ydl_opts)
    try:
        info = ydl.extract_info(url, download=False)
    except TimeoutError as exc:
        raise PreflightTimeoutError(f"Preflight network timeout for '{url}': {exc}") from exc
    except Exception as exc:
        msg = str(exc).lower()
        if "timeout" in msg:
            raise PreflightTimeoutError(f"Preflight network timeout for '{url}': {exc}") from exc
        if "private" in msg or "unavailable" in msg:
            raise VideoUnavailableError(f"Video is private or unavailable: {exc}") from exc
        if "sign in" in msg or "age" in msg:
            raise VideoAgeRestrictedError(f"Sign in required for age-restricted video: {exc}") from exc
        raise VideoDownloadError(f"Failed to fetch metadata for '{url}': {exc}") from exc

    if not info:
        raise VideoDownloadError(f"yt-dlp returned empty info for '{url}'")

    meta = VideoMetadata(
        id=str(info.get("id") or vid_id),
        title=str(info.get("title") or "Untitled"),
        duration=float(info.get("duration", 0.0) or 0.0),
        uploader=info.get("uploader") or info.get("channel") or "Unknown",
        is_live=bool(info.get("is_live", False)),
        description=str(info.get("description", "")),
        view_count=info.get("view_count"),
        like_count=info.get("like_count"),
        thumbnail=info.get("thumbnail"),
        width=info.get("width"),
        height=info.get("height"),
        fps=info.get("fps"),
        ext=str(info.get("ext", "mp4")),
        heatmap=info.get("heatmap"),
    )

    if use_cache:
        _METADATA_CACHE[clean_url] = meta

    return meta


# ==============================================================================
# Feature 1: YouTube Video Download
# ==============================================================================

def download_video(
    url: Union[str, Path],
    output_dir: Union[str, Path] = "./downloads",
    max_height: Optional[int] = 1080,
    format_spec: Optional[str] = None,
    max_duration: Optional[float] = None,
    timeout: int = 30,
) -> Path:
    """
    Downloads YouTube video using yt-dlp capped at max_height MP4.
    If url refers to an existing local video, bypasses download and returns verified path.
    """
    # 1. Validate non-empty
    if url is None:
        raise InvalidURLError("URL or file path cannot be None.")
    if isinstance(url, str) and not url.strip():
        raise InvalidURLError("URL or file path cannot be empty or whitespace.")

    # 2. Local file bypass (MUST precede yt_dlp instantiation)
    local_p = resolve_local_video_path(url)
    if local_p is not None:
        logger.info("Local video path detected; bypassing download: %s", local_p)
        return local_p

    # 3. Validate YouTube URL
    raw_url = str(url).strip()
    vid_id = extract_video_id(raw_url)
    if not vid_id:
        raise InvalidURLError(f"Invalid URL or non-existent file path: '{url}'")

    # 4. Prepare download directory and options
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Fast check: If a video matching this ID is already downloaded in out_dir, reuse it!
    # Avoids redundant multi-hundred MB downloads and Windows file lock collisions (WinError 32)
    for candidate in out_dir.glob(f"*[{vid_id}].*"):
        if candidate.is_file() and candidate.suffix.lower() in {".mp4", ".mkv", ".webm"} and candidate.stat().st_size > 100_000:
            logger.info("Found existing downloaded video matching ID [%s]: %s; reusing.", vid_id, candidate)
            return candidate.resolve()

    limit = 1080 if max_height is None else max(144, min(int(max_height), 1080))
    fmt = format_spec or f"bestvideo[height<={limit}][ext=mp4]+bestaudio[ext=m4a]/bestvideo[height<={limit}]+bestaudio/best[height<={limit}]/best"

    outtmpl = str(out_dir / f"%(title)s [%(id)s].%(ext)s")

    ydl_opts = {
        "outtmpl": outtmpl,
        "format": fmt,
        "merge_output_format": "mp4",
        "format_sort": ["res", "fps"],
        "writesubtitles": False,
        "writeautomaticsub": False,
        "noplaylist": True,
        "overwrites": True,
        "quiet": True,
        "no_warnings": False,
        "socket_timeout": timeout,
        "retries": 5,
        "fragment_retries": 5,
        "nocheckcertificate": True,
        "http_headers": {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        },
    }
    js_rts = _get_default_js_runtimes()
    if js_rts:
        ydl_opts["js_runtimes"] = js_rts

    ydl = yt_dlp.YoutubeDL(ydl_opts)
    try:
        info = ydl.extract_info(raw_url, download=True)
    except Exception as exc:
        msg = str(exc).lower()
        if "private" in msg or "unavailable" in msg:
            raise VideoUnavailableError(f"Video is private or unavailable: {exc}") from exc
        if "sign in" in msg or "age" in msg:
            raise VideoAgeRestrictedError(f"Sign in required for age-restricted video: {exc}") from exc
        if "live" in msg or "stream" in msg:
            raise StreamDurationUnavailableError(f"Active livestream download rejected: {exc}") from exc
        raise VideoDownloadError(f"Failed to download video from '{url}': {exc}") from exc

    if not info:
        raise VideoDownloadError(f"yt-dlp returned no metadata during download of '{url}'")

    # 5. Check livestream / non-positive duration
    if info.get("is_live") is True or (info.get("duration") is not None and info.get("duration") <= 0):
        raise StreamDurationUnavailableError(
            f"Active live stream in progress ({url}); livestream download is not supported."
        )

    # 6. Check duration cap
    if max_duration is not None and info.get("duration"):
        if float(info["duration"]) > max_duration:
            raise VideoDurationExceededError(
                f"Video duration ({info['duration']}s) exceeds maximum allowed duration ({max_duration}s)."
            )

    # 7. Locate output file
    target_video_path: Optional[Path] = None

    # Check _filename first (provided by mocks & direct downloads)
    candidate_filename = info.get("_filename")
    if candidate_filename and Path(candidate_filename).exists():
        target_video_path = Path(candidate_filename).resolve()

    # Check prepare_filename
    if target_video_path is None:
        try:
            prep = ydl.prepare_filename(info)
            if prep and isinstance(prep, (str, Path)):
                prepared = Path(prep)
                if prepared.exists():
                    target_video_path = prepared.resolve()
                elif prepared.with_suffix(".mp4").exists():
                    target_video_path = prepared.with_suffix(".mp4").resolve()
        except Exception:
            pass

    # Glob search in output_dir
    if target_video_path is None:
        matches = list(out_dir.glob(f"*{vid_id}*.mp4")) + list(out_dir.glob(f"*{vid_id}*.*"))
        if matches:
            target_video_path = matches[0].resolve()

    if target_video_path is None:
        mp4_files = sorted(out_dir.glob("*.mp4"), key=lambda f: f.stat().st_mtime, reverse=True)
        if mp4_files:
            target_video_path = mp4_files[0].resolve()

    if target_video_path is None or not target_video_path.exists():
        raise VideoDownloadError(f"Download completed for '{url}' but output MP4 file was not found in '{out_dir}'.")

    # 8. Save YouTube heatmap sidecar if available
    heatmap_data = info.get("heatmap")
    if heatmap_data and isinstance(heatmap_data, list):
        try:
            import json
            hm_path = target_video_path.parent / f"{target_video_path.stem}.heatmap.json"
            hm_path.write_text(json.dumps(heatmap_data, indent=2), encoding="utf-8")
            logger.info("Saved YouTube Most Replayed heatmap (%d points): %s", len(heatmap_data), hm_path.name)
        except Exception as e:
            logger.warning("Failed to save heatmap sidecar: %s", e)

    # 9. Save Video Metadata sidecar (title, uploader, duration)
    try:
        import json
        meta_dict = {
            "id": str(info.get("id") or vid_id),
            "title": str(info.get("title") or target_video_path.stem),
            "uploader": str(info.get("uploader") or info.get("channel") or "Unknown"),
            "duration": float(info.get("duration", 0.0) or 0.0),
        }
        meta_path = target_video_path.parent / f"{target_video_path.stem}.meta.json"
        meta_path.write_text(json.dumps(meta_dict, indent=2), encoding="utf-8")
        logger.info("Saved video metadata sidecar: %s", meta_path.name)
    except Exception as e:
        logger.warning("Failed to save metadata sidecar: %s", e)

    return target_video_path


def get_video_title(video_path: Union[str, Path], default: Optional[str] = None) -> Optional[str]:
    """
    Extracts video title from metadata sidecar, cache, or clean file stem.
    Strips trailing YouTube video IDs (e.g. [jJ-zWCw85qA]) and formats cleanly.
    """
    vp = Path(video_path)

    # 1. Check sidecar JSON files
    candidates = [
        vp.parent / f"{vp.stem}.meta.json",
        vp.parent / f"{vp.name}.meta.json",
        vp.with_suffix(".meta.json"),
    ]
    for c in candidates:
        if c.exists() and c.stat().st_size > 0:
            try:
                import json
                data = json.loads(c.read_text(encoding="utf-8"))
                if isinstance(data, dict) and data.get("title"):
                    raw_t = str(data["title"]).strip()
                    if raw_t:
                        return raw_t
            except Exception:
                pass

    # 2. Check metadata cache
    for meta in _METADATA_CACHE.values():
        if meta.title and (meta.id in vp.stem or meta.title in vp.stem):
            return meta.title

    # 3. Clean filename stem
    stem = vp.stem
    cleaned = re.sub(r"\s*\[[A-Za-z0-9_-]{8,15}\]$", "", stem).strip()
    if cleaned:
        cleaned = re.sub(r"^[0-9]+[_\s-]+", "", cleaned).strip()
        if "_" in cleaned and " " not in cleaned:
            cleaned = cleaned.replace("_", " ")
        return cleaned

    return default


def load_video_heatmap(video_path: Union[str, Path]) -> Optional[List[Dict[str, float]]]:
    """
    Loads YouTube Most Replayed heatmap data from sidecar JSON if available.
    """
    import json
    vp = Path(video_path)
    candidates = [
        vp.parent / f"{vp.stem}.heatmap.json",
        vp.parent / f"{vp.name}.heatmap.json",
        vp.with_suffix(".heatmap.json"),
    ]
    for c in candidates:
        if c.exists() and c.stat().st_size > 0:
            try:
                data = json.loads(c.read_text(encoding="utf-8"))
                if isinstance(data, list) and len(data) > 0:
                    return data
            except Exception:
                pass
    return None
