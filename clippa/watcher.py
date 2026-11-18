"""
clippa/watcher.py - YouTube Channel Observer & Hands-Free Autopilot Pipeline

Continuously monitors user-specified YouTube channels (e.g. @MrBeast, @IShowSpeed),
detects newly published videos, and automatically triggers CLIPPA's pipeline to
clip, reframe into 9:16 vertical Shorts, generate captions, and auto-upload directly
to the user's connected YouTube channel without requiring manual intervention.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("clippa.watcher")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

CONFIG_DIR = Path.home() / ".clippa"
WATCHLIST_FILE = CONFIG_DIR / "watchlist.json"
HISTORY_FILE = CONFIG_DIR / "history.json"
SETTINGS_FILE = CONFIG_DIR / "autopilot_settings.json"


def ensure_config_dir() -> Path:
    """Ensures ~/.clippa directory exists."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    return CONFIG_DIR


# ---------------------------------------------------------------------------
# Channel Resolution & Video Fetching
# ---------------------------------------------------------------------------

def normalize_channel_url(identifier: str) -> str:
    """
    Normalizes a channel handle, name, or URL into a YouTube channel videos URL.
    Examples:
      - '@MrBeast' -> 'https://www.youtube.com/@MrBeast/videos'
      - 'IShowSpeed' -> 'https://www.youtube.com/@IShowSpeed/videos'
      - 'https://www.youtube.com/@MrBeast' -> 'https://www.youtube.com/@MrBeast/videos'
      - 'https://www.youtube.com/channel/UC...' -> 'https://www.youtube.com/channel/UC.../videos'
    """
    s = identifier.strip()
    if not s.startswith("http://") and not s.startswith("https://"):
        if not s.startswith("@"):
            s = f"@{s}"
        return f"https://www.youtube.com/{s}/videos"

    # Ensure /videos tab is targeted for latest upload stream
    if not s.endswith("/videos"):
        s = s.rstrip("/") + "/videos"
    return s


def resolve_channel_info(identifier: str) -> Dict[str, Any]:
    """
    Resolves YouTube channel details using lightweight yt-dlp metadata extraction.
    Returns:
      {
        "id": channel_id,
        "name": channel_name,
        "handle": "@handle",
        "url": channel_url,
        "avatar": avatar_url,
        "added_at": float
      }
    """
    import yt_dlp  # type: ignore

    url = normalize_channel_url(identifier)
    ydl_opts = {
        "extract_flat": True,
        "playlistend": 1,
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            res = ydl.extract_info(url, download=False)
            if not res:
                raise ValueError(f"Could not extract channel info for '{identifier}'")

            channel_name = res.get("channel") or res.get("uploader") or identifier
            uploader_id = res.get("uploader_id") or ""
            handle = uploader_id if uploader_id.startswith("@") else f"@{uploader_id or channel_name.replace(' ', '')}"
            channel_id = res.get("id") or uploader_id
            channel_url = res.get("channel_url") or res.get("uploader_url") or f"https://www.youtube.com/{handle}"

            # Extract avatar thumbnail if present
            avatar = ""
            thumbnails = res.get("thumbnails", [])
            for thumb in thumbnails:
                if isinstance(thumb, dict) and "avatar" in thumb.get("id", ""):
                    avatar = thumb.get("url", "")
                    break
            if not avatar and thumbnails:
                # Pick the highest quality square thumbnail or first available
                for thumb in reversed(thumbnails):
                    if isinstance(thumb, dict) and thumb.get("width") == thumb.get("height"):
                        avatar = thumb.get("url", "")
                        break
                if not avatar and isinstance(thumbnails[0], dict):
                    avatar = thumbnails[0].get("url", "")

            return {
                "id": channel_id,
                "name": channel_name,
                "handle": handle,
                "url": channel_url,
                "avatar": avatar,
                "added_at": time.time(),
            }
    except Exception as exc:
        logger.error("Failed to resolve channel info for '%s': %s", identifier, exc)
        raise


def fetch_recent_channel_videos(channel_identifier: str, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Fetches recent videos from a channel using fast flat extraction (takes ~1s, 0 quota).
    """
    import yt_dlp  # type: ignore

    url = normalize_channel_url(channel_identifier)
    ydl_opts = {
        "extract_flat": True,
        "playlistend": limit,
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
    }

    videos: List[Dict[str, Any]] = []
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            res = ydl.extract_info(url, download=False)
            if not res:
                return []

            entries = res.get("entries", [])
            for entry in entries:
                if not entry:
                    continue
                v_id = entry.get("id")
                if not v_id:
                    continue

                v_title = entry.get("title", f"Video {v_id}")
                v_url = entry.get("url")
                if not v_url or not v_url.startswith("http"):
                    v_url = f"https://www.youtube.com/watch?v={v_id}"

                v_thumb = ""
                thumbs = entry.get("thumbnails", [])
                if thumbs and isinstance(thumbs, list) and isinstance(thumbs[-1], dict):
                    v_thumb = thumbs[-1].get("url", "")

                videos.append({
                    "id": v_id,
                    "title": v_title,
                    "url": v_url,
                    "duration": entry.get("duration", 0),
                    "thumbnail": v_thumb,
                })
        return videos
    except Exception as exc:
        logger.error("Failed fetching recent videos for '%s': %s", channel_identifier, exc)
        return []


# ---------------------------------------------------------------------------
# Watchlist Persistence (~/.clippa/watchlist.json)
# ---------------------------------------------------------------------------

def get_watchlist() -> List[Dict[str, Any]]:
    """Returns list of observed channels."""
    ensure_config_dir()
    if not WATCHLIST_FILE.exists():
        return []
    try:
        data = json.loads(WATCHLIST_FILE.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
    except Exception as exc:
        logger.warning("Error reading watchlist: %s", exc)
    return []


def save_watchlist(watchlist: List[Dict[str, Any]]) -> None:
    """Saves the observed channels list."""
    ensure_config_dir()
    WATCHLIST_FILE.write_text(json.dumps(watchlist, indent=2), encoding="utf-8")


def add_channel_to_watchlist(identifier: str) -> Dict[str, Any]:
    """Resolves and adds a channel to the watchlist if not already present."""
    info = resolve_channel_info(identifier)
    current = get_watchlist()

    # Check for existing
    for item in current:
        if (
            item.get("id") == info["id"]
            or item.get("handle", "").lower() == info["handle"].lower()
            or item.get("url", "").lower() == info["url"].lower()
        ):
            return item

    current.append(info)
    save_watchlist(current)
    logger.info("Added channel '%s' (%s) to watchlist.", info["name"], info["handle"])
    return info


def remove_channel_from_watchlist(channel_id_or_handle: str) -> bool:
    """Removes a channel from the watchlist by id or handle."""
    current = get_watchlist()
    target = channel_id_or_handle.strip().lower()
    filtered = [
        item for item in current
        if item.get("id", "").lower() != target
        and item.get("handle", "").lower() != target
    ]
    if len(filtered) != len(current):
        save_watchlist(filtered)
        logger.info("Removed channel '%s' from watchlist.", channel_id_or_handle)
        return True
    return False


# ---------------------------------------------------------------------------
# History Ledger (~/.clippa/history.json)
# ---------------------------------------------------------------------------

def get_history() -> Dict[str, Any]:
    """Returns the history ledger of processed videos."""
    ensure_config_dir()
    if not HISTORY_FILE.exists():
        return {"processed": {}}
    try:
        data = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "processed" in data:
            return data
    except Exception as exc:
        logger.warning("Error reading history ledger: %s", exc)
    return {"processed": {}}


def save_history(history: Dict[str, Any]) -> None:
    """Persists history ledger."""
    ensure_config_dir()
    HISTORY_FILE.write_text(json.dumps(history, indent=2), encoding="utf-8")


def is_video_processed(video_id: str) -> bool:
    """Checks whether a video ID has already been clipped / processed."""
    h = get_history()
    return video_id in h.get("processed", {})


def record_video_history(
    video_id: str,
    status: str,
    title: str = "",
    channel_name: str = "",
    channel_handle: str = "",
    url: str = "",
    clips: Optional[List[str]] = None,
    shorts_urls: Optional[List[str]] = None,
    error: Optional[str] = None,
) -> None:
    """Records or updates processing status in history ledger."""
    h = get_history()
    processed = h.setdefault("processed", {})
    entry = processed.get(video_id, {})

    entry.update({
        "id": video_id,
        "status": status,
        "title": title or entry.get("title", ""),
        "channel_name": channel_name or entry.get("channel_name", ""),
        "channel_handle": channel_handle or entry.get("channel_handle", ""),
        "url": url or entry.get("url", f"https://www.youtube.com/watch?v={video_id}"),
        "updated_at": time.time(),
    })
    if "created_at" not in entry:
        entry["created_at"] = time.time()
    if clips is not None:
        entry["clips"] = clips
    if shorts_urls is not None:
        entry["shorts_urls"] = shorts_urls
        entry["uploaded"] = bool(shorts_urls)
    if error is not None:
        entry["error"] = error

    processed[video_id] = entry
    save_history(h)


# ---------------------------------------------------------------------------
# Autopilot Settings (~/.clippa/autopilot_settings.json)
# ---------------------------------------------------------------------------

DEFAULT_SETTINGS: Dict[str, Any] = {
    "enabled": False,
    "poll_interval_seconds": 600,  # 10 minutes
    "auto_upload": True,
    "upload_privacy": "scheduled",
    "template": "default",
    "max_clips": 3,
    "max_duration": 120,
    "nvenc": True,
}


def get_autopilot_settings() -> Dict[str, Any]:
    """Loads autopilot configuration settings."""
    ensure_config_dir()
    if not SETTINGS_FILE.exists():
        return dict(DEFAULT_SETTINGS)
    try:
        data = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        res = dict(DEFAULT_SETTINGS)
        res.update(data)
        return res
    except Exception:
        return dict(DEFAULT_SETTINGS)


def save_autopilot_settings(settings: Dict[str, Any]) -> Dict[str, Any]:
    """Saves autopilot configuration settings."""
    ensure_config_dir()
    current = get_autopilot_settings()
    current.update(settings)
    SETTINGS_FILE.write_text(json.dumps(current, indent=2), encoding="utf-8")
    return current


# ---------------------------------------------------------------------------
# Autopilot Daemon
# ---------------------------------------------------------------------------

class AutopilotDaemon:
    """
    Background worker thread that regularly polls watched YouTube channels,
    detects new videos, and automatically triggers the clipping + uploading pipeline.
    """

    _instance: Optional[AutopilotDaemon] = None

    @classmethod
    def get_instance(cls, output_dir: Optional[Path] = None, temp_dir: Optional[Path] = None) -> AutopilotDaemon:
        if cls._instance is None:
            cls._instance = cls(output_dir=output_dir, temp_dir=temp_dir)
        return cls._instance

    def __init__(self, output_dir: Optional[Path] = None, temp_dir: Optional[Path] = None) -> None:
        self.output_dir = Path(output_dir or "outputs").resolve()
        self.temp_dir = Path(temp_dir or "temp").resolve()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._is_polling = False
        self._last_poll_time: Optional[float] = None
        self._activity_feed: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    def start(self) -> None:
        """Starts the background monitoring daemon."""
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._run_loop, daemon=True, name="AutopilotDaemon")
            self._thread.start()
            logger.info("AutopilotDaemon started in background.")

    def stop(self) -> None:
        """Stops the monitoring daemon."""
        with self._lock:
            self._stop_event.set()
            if self._thread and self._thread.is_alive():
                self._thread.join(timeout=3)
            self._thread = None
            logger.info("AutopilotDaemon stopped.")

    def log_activity(self, message: str, level: str = "info", details: Optional[Dict[str, Any]] = None) -> None:
        """Appends an event to the in-memory activity feed."""
        entry = {
            "message": message,
            "level": level,
            "timestamp": time.time(),
            "details": details or {},
        }
        with self._lock:
            self._activity_feed.insert(0, entry)
            # Keep last 50 events in memory
            if len(self._activity_feed) > 50:
                self._activity_feed = self._activity_feed[:50]
        logger.info("[Autopilot Activity] %s", message)

    def get_activity_feed(self) -> List[Dict[str, Any]]:
        """Returns the recent autopilot activity feed."""
        with self._lock:
            return list(self._activity_feed)

    def is_running(self) -> bool:
        """Checks if daemon thread is currently active."""
        return self._thread is not None and self._thread.is_alive()

    def poll_once(self) -> List[Dict[str, Any]]:
        """
        Executes a single check across all channels in the watchlist.
        Detects unclipped videos and processes them automatically.
        """
        if self._is_polling:
            logger.info("Poll already in progress, skipping overlapping run.")
            return []

        self._is_polling = True
        self._last_poll_time = time.time()
        processed_in_this_pass: List[Dict[str, Any]] = []

        try:
            watchlist = get_watchlist()
            settings = get_autopilot_settings()

            if not watchlist:
                self.log_activity("No channels in watchlist to observe.", level="debug")
                return []

            self.log_activity(f"Checking {len(watchlist)} watched channels for new uploads...")

            for ch in watchlist:
                ch_name = ch.get("name", "Unknown")
                ch_handle = ch.get("handle", "")
                ch_url = ch.get("url", "") or ch_handle

                videos = fetch_recent_channel_videos(ch_url, limit=3)
                for vid in videos:
                    vid_id = vid["id"]
                    vid_title = vid["title"]
                    vid_url = vid["url"]

                    # Deduplication check
                    if is_video_processed(vid_id):
                        continue

                    # Found a fresh video!
                    self.log_activity(
                        f"🚨 New upload detected from {ch_name} ({ch_handle}): '{vid_title}'",
                        level="alert",
                        details={"channel": ch_name, "video_id": vid_id, "url": vid_url},
                    )

                    # Mark in history as processing to prevent race conditions
                    record_video_history(
                        video_id=vid_id,
                        status="processing",
                        title=vid_title,
                        channel_name=ch_name,
                        channel_handle=ch_handle,
                        url=vid_url,
                    )

                    # Check if sequential ClippingQueueManager worker is active
                    from clippa.queue_manager import ClippingQueueManager
                    qm = ClippingQueueManager.get_instance(output_dir=self.output_dir, temp_dir=self.temp_dir)
                    if qm._worker_thread and qm._worker_thread.is_alive():
                        # Live server mode: delegate to unified sequential queue
                        self.log_activity(
                            f"📋 Queued '{vid_title}' for clipping & auto-upload in sequential queue...",
                            details={"video_id": vid_id},
                        )
                        item = qm.add_to_queue(
                            video_id=vid_id,
                            title=vid_title,
                            url=vid_url,
                            channel=ch_name,
                            thumbnail=vid.get("thumbnail", ""),
                            duration=vid.get("duration", 0),
                            template=settings.get("template", "default"),
                            max_clips=settings.get("max_clips", 3),
                            max_duration=settings.get("max_duration", 120),
                            auto_upload=settings.get("auto_upload", True),
                            upload_privacy=settings.get("upload_privacy", "scheduled"),
                            nvenc=settings.get("nvenc", True),
                        )
                        processed_in_this_pass.append({
                            "video_id": vid_id,
                            "title": vid_title,
                            "channel": ch_name,
                            "queue_id": item.id,
                        })
                        continue

                    # Execute clipping pipeline (direct mode, e.g. tests)
                    try:
                        self.log_activity(
                            f"🎬 Auto-clipping '{vid_title}' (template: {settings.get('template', 'default')})...",
                            details={"video_id": vid_id},
                        )

                        from clippa.cli import run_pipeline

                        # Snapshot existing clips to find newly created ones
                        before_clips = set(self.output_dir.glob("*.mp4"))

                        exit_code = run_pipeline(
                            url=vid_url,
                            output_dir=self.output_dir,
                            temp_dir=self.temp_dir,
                            template=settings.get("template", "default"),
                            max_clips=settings.get("max_clips", 3),
                            max_duration=settings.get("max_duration", 120),
                            nvenc=settings.get("nvenc", True),
                            auto_upload=settings.get("auto_upload", True),
                            upload_privacy=settings.get("upload_privacy", "scheduled"),
                        )

                        after_clips = set(self.output_dir.glob("*.mp4"))
                        new_clips = sorted(list(after_clips - before_clips), key=lambda p: p.stat().st_mtime)

                        clip_names = [p.name for p in new_clips]
                        shorts_urls: List[str] = []

                        for p in new_clips:
                            meta_file = p.with_name(f"{p.stem}.meta.json")
                            if meta_file.exists():
                                try:
                                    m_data = json.loads(meta_file.read_text(encoding="utf-8"))
                                    yt = m_data.get("youtube", {})
                                    if yt.get("uploaded") and yt.get("url"):
                                        shorts_urls.append(yt["url"])
                                except Exception:
                                    pass

                        if exit_code == 0:
                            record_video_history(
                                video_id=vid_id,
                                status="completed",
                                title=vid_title,
                                channel_name=ch_name,
                                channel_handle=ch_handle,
                                url=vid_url,
                                clips=clip_names,
                                shorts_urls=shorts_urls,
                            )
                            msg = f"✅ Completed clipping for '{vid_title}' ({len(clip_names)} clips generated)."
                            if shorts_urls:
                                msg += f" 🚀 Published {len(shorts_urls)} Shorts to YouTube!"
                            self.log_activity(msg, level="success", details={"shorts": shorts_urls})
                            processed_in_this_pass.append({
                                "video_id": vid_id,
                                "title": vid_title,
                                "channel": ch_name,
                                "clips": clip_names,
                                "shorts": shorts_urls,
                            })
                        else:
                            record_video_history(
                                video_id=vid_id,
                                status="failed",
                                error="Pipeline exited with non-zero status code",
                            )
                            self.log_activity(
                                f"❌ Pipeline failed for '{vid_title}'",
                                level="error",
                                details={"video_id": vid_id},
                            )

                    except Exception as pipe_err:
                        logger.exception("Pipeline run failed for video %s: %s", vid_id, pipe_err)
                        record_video_history(
                            video_id=vid_id,
                            status="failed",
                            error=str(pipe_err),
                        )
                        self.log_activity(
                            f"❌ Error processing '{vid_title}': {pipe_err}",
                            level="error",
                            details={"error": str(pipe_err)},
                        )

            return processed_in_this_pass
        finally:
            self._is_polling = False

    def _run_loop(self) -> None:
        """Internal daemon loop."""
        logger.info("AutopilotDaemon loop started.")
        while not self._stop_event.is_set():
            try:
                settings = get_autopilot_settings()
                if settings.get("enabled", False):
                    self.poll_once()

                interval = max(60, int(settings.get("poll_interval_seconds", 600)))
                # Sleep in increments of 2s to respond promptly to stop requests
                for _ in range(interval // 2):
                    if self._stop_event.is_set():
                        break
                    time.sleep(2)
            except Exception as loop_err:
                logger.error("Error in Autopilot daemon loop: %s", loop_err)
                time.sleep(10)
        logger.info("AutopilotDaemon loop ended.")

    def get_status(self) -> Dict[str, Any]:
        """Returns the current state of the Autopilot daemon."""
        settings = get_autopilot_settings()
        watchlist = get_watchlist()
        history = get_history()
        return {
            "daemon_alive": self.is_running(),
            "enabled": settings.get("enabled", False),
            "is_polling": self._is_polling,
            "last_poll_time": self._last_poll_time,
            "poll_interval_seconds": settings.get("poll_interval_seconds", 600),
            "watched_channels_count": len(watchlist),
            "total_processed_videos": len(history.get("processed", {})),
            "settings": settings,
        }
