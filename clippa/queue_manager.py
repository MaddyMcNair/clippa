"""
clippa/queue_manager.py - Sequential Prioritized Video Queue & Auto-Upload Engine

Allows users to:
1. Browse YouTuber video feeds directly in the app.
2. Add videos to a processing queue ("clip this video first, then this, then this").
3. Re-order queue items (move up/down) or prioritize to front.
4. Sequentially process queued videos in the background and automatically upload
   each rendered 9:16 Short to YouTube without manual approval.
5. Persist queue state to ~/.clippa/queue.json across app restarts.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("clippa.queue")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

CONFIG_DIR = Path.home() / ".clippa"
QUEUE_FILE = CONFIG_DIR / "queue.json"


def ensure_config_dir() -> Path:
    """Ensures ~/.clippa directory exists."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    return CONFIG_DIR


class QueueItem:
    """Represents a single video task in the clipping & auto-upload queue."""

    def __init__(
        self,
        video_id: str,
        title: str,
        url: str,
        channel: str = "",
        thumbnail: str = "",
        duration: int = 0,
        template: str = "default",
        max_clips: int = 3,
        max_duration: int = 120,
        auto_upload: bool = True,
        upload_privacy: str = "scheduled",
        nvenc: bool = True,
        item_id: Optional[str] = None,
        status: str = "pending",
        created_at: Optional[float] = None,
        started_at: Optional[float] = None,
        finished_at: Optional[float] = None,
        message: str = "Waiting in queue...",
        result_clips: Optional[List[str]] = None,
        shorts_urls: Optional[List[str]] = None,
        error: Optional[str] = None,
    ):
        self.id = item_id or f"q_{uuid.uuid4().hex[:8]}"
        self.video_id = video_id
        self.title = title
        self.url = url
        self.channel = channel
        self.thumbnail = thumbnail
        self.duration = duration
        self.template = template
        self.max_clips = max_clips
        self.max_duration = max_duration
        self.auto_upload = auto_upload
        self.upload_privacy = upload_privacy
        self.nvenc = nvenc
        self.status = status  # 'pending', 'processing', 'completed', 'failed'
        self.created_at = created_at or time.time()
        self.started_at = started_at
        self.finished_at = finished_at
        self.message = message
        self.result_clips = result_clips or []
        self.shorts_urls = shorts_urls or []
        self.error = error

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "video_id": self.video_id,
            "title": self.title,
            "url": self.url,
            "channel": self.channel,
            "thumbnail": self.thumbnail,
            "duration": self.duration,
            "template": self.template,
            "max_clips": self.max_clips,
            "max_duration": self.max_duration,
            "auto_upload": self.auto_upload,
            "upload_privacy": self.upload_privacy,
            "nvenc": self.nvenc,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "message": self.message,
            "result_clips": self.result_clips,
            "shorts_urls": self.shorts_urls,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> QueueItem:
        return cls(
            item_id=data.get("id"),
            video_id=data.get("video_id", ""),
            title=data.get("title", ""),
            url=data.get("url", ""),
            channel=data.get("channel", ""),
            thumbnail=data.get("thumbnail", ""),
            duration=data.get("duration", 0),
            template=data.get("template", "default"),
            max_clips=data.get("max_clips", 3),
            max_duration=data.get("max_duration", 120),
            auto_upload=data.get("auto_upload", True),
            upload_privacy=data.get("upload_privacy", "public"),
            nvenc=data.get("nvenc", True),
            status=data.get("status", "pending"),
            created_at=data.get("created_at"),
            started_at=data.get("started_at"),
            finished_at=data.get("finished_at"),
            message=data.get("message", "Waiting in queue..."),
            result_clips=data.get("result_clips", []),
            shorts_urls=data.get("shorts_urls", []),
            error=data.get("error"),
        )


class ClippingQueueManager:
    """Manages sequential prioritized clipping and auto-uploading of videos."""

    _instance: Optional[ClippingQueueManager] = None

    @classmethod
    def get_instance(cls, output_dir: Optional[Path] = None, temp_dir: Optional[Path] = None) -> ClippingQueueManager:
        if cls._instance is None:
            cls._instance = cls(output_dir=output_dir, temp_dir=temp_dir)
        return cls._instance

    def __init__(self, output_dir: Optional[Path] = None, temp_dir: Optional[Path] = None):
        self.output_dir = Path(output_dir or "outputs").resolve()
        self.temp_dir = Path(temp_dir or "temp").resolve()
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None
        self._current_item: Optional[QueueItem] = None
        self._pending_queue: List[QueueItem] = []
        self._completed_history: List[QueueItem] = []
        self._load_from_disk()

    def _load_from_disk(self) -> None:
        """Loads persistent queue state from disk."""
        ensure_config_dir()
        if not QUEUE_FILE.exists():
            return
        try:
            with open(QUEUE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            with self._lock:
                self._pending_queue = [
                    QueueItem.from_dict(d) for d in data.get("pending", [])
                ]
                # If an item was interrupted mid-process, put it back to pending
                curr = data.get("current")
                if curr:
                    interrupted = QueueItem.from_dict(curr)
                    interrupted.status = "pending"
                    interrupted.message = "Resumed after app restart"
                    self._pending_queue.insert(0, interrupted)

                self._completed_history = [
                    QueueItem.from_dict(d) for d in data.get("completed", [])[:50]
                ]
        except Exception as exc:
            logger.warning("Failed loading queue from disk: %s", exc)

    def _save_to_disk(self) -> None:
        """Persists current queue state to disk."""
        ensure_config_dir()
        try:
            with self._lock:
                payload = {
                    "current": self._current_item.to_dict() if self._current_item else None,
                    "pending": [item.to_dict() for item in self._pending_queue],
                    "completed": [item.to_dict() for item in self._completed_history[:50]],
                }
            with open(QUEUE_FILE, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except Exception as exc:
            logger.warning("Failed saving queue to disk: %s", exc)

    def start_worker(self) -> None:
        """Starts background queue processing thread."""
        with self._lock:
            if self._worker_thread and self._worker_thread.is_alive():
                return
            self._stop_event.clear()
            self._worker_thread = threading.Thread(
                target=self._worker_loop, daemon=True, name="QueueWorkerThread"
            )
            self._worker_thread.start()
            logger.info("ClippingQueueManager worker started.")

    def stop_worker(self) -> None:
        """Stops background queue worker."""
        with self._lock:
            self._stop_event.set()
            if self._worker_thread and self._worker_thread.is_alive():
                self._worker_thread.join(timeout=3)
            self._worker_thread = None
            logger.info("ClippingQueueManager worker stopped.")

    def add_to_queue(
        self,
        video_id: str,
        title: str,
        url: str,
        channel: str = "",
        thumbnail: str = "",
        duration: int = 0,
        template: str = "default",
        max_clips: int = 3,
        max_duration: int = 120,
        auto_upload: bool = True,
        upload_privacy: str = "scheduled",
        nvenc: bool = True,
        prioritize_first: bool = False,
    ) -> QueueItem:
        """
        Adds a video to the processing queue.
        If prioritize_first is True, inserts at the top of the pending queue.
        """
        with self._lock:
            # Check if this video is already in pending queue or currently processing
            if self._current_item and self._current_item.video_id == video_id:
                return self._current_item

            for existing in self._pending_queue:
                if existing.video_id == video_id:
                    if prioritize_first:
                        self._pending_queue.remove(existing)
                        self._pending_queue.insert(0, existing)
                        self._save_to_disk()
                    return existing

            item = QueueItem(
                video_id=video_id,
                title=title,
                url=url,
                channel=channel,
                thumbnail=thumbnail,
                duration=duration,
                template=template,
                max_clips=max_clips,
                max_duration=max_duration,
                auto_upload=auto_upload,
                upload_privacy=upload_privacy,
                nvenc=nvenc,
            )

            if prioritize_first:
                self._pending_queue.insert(0, item)
            else:
                self._pending_queue.append(item)

            self._save_to_disk()
            logger.info("Added video '%s' to clipping queue (position %d)", title, len(self._pending_queue))
            return item

    def remove_from_queue(self, item_id: str) -> bool:
        """Removes a video from the pending queue."""
        with self._lock:
            for idx, item in enumerate(self._pending_queue):
                if item.id == item_id:
                    self._pending_queue.pop(idx)
                    self._save_to_disk()
                    logger.info("Removed item '%s' from queue.", item_id)
                    return True
        return False

    def move_item(self, item_id: str, direction: str) -> bool:
        """
        Moves a pending queue item 'up' (higher priority) or 'down' (lower priority).
        """
        with self._lock:
            for idx, item in enumerate(self._pending_queue):
                if item.id == item_id:
                    if direction == "up" and idx > 0:
                        self._pending_queue[idx], self._pending_queue[idx - 1] = (
                            self._pending_queue[idx - 1],
                            self._pending_queue[idx],
                        )
                        self._save_to_disk()
                        return True
                    elif direction == "down" and idx < len(self._pending_queue) - 1:
                        self._pending_queue[idx], self._pending_queue[idx + 1] = (
                            self._pending_queue[idx + 1],
                            self._pending_queue[idx],
                        )
                        self._save_to_disk()
                        return True
                    break
        return False

    def clear_pending(self) -> int:
        """Clears all pending items from queue."""
        with self._lock:
            count = len(self._pending_queue)
            self._pending_queue.clear()
            self._save_to_disk()
            return count

    def retry_item(self, item_id: str) -> Optional[QueueItem]:
        """Moves a failed item from completed history back to the front of the pending queue."""
        with self._lock:
            for idx, item in enumerate(self._completed_history):
                if item.id == item_id or item.video_id == item_id:
                    self._completed_history.pop(idx)
                    item.status = "pending"
                    item.error = None
                    item.message = "Retrying in queue..."
                    self._pending_queue.insert(0, item)
                    self._save_to_disk()
                    logger.info("Retrying item '%s' (%s).", item.id, item.title)
                    return item
        return None

    def get_state(self) -> Dict[str, Any]:
        """Returns the complete snapshot of the queue system."""
        with self._lock:
            return {
                "is_running": self._worker_thread is not None and self._worker_thread.is_alive(),
                "current": self._current_item.to_dict() if self._current_item else None,
                "pending": [item.to_dict() for item in self._pending_queue],
                "completed": [item.to_dict() for item in self._completed_history[:25]],
                "pending_count": len(self._pending_queue),
            }

    def _worker_loop(self) -> None:
        """Sequential queue execution loop."""
        logger.info("QueueWorkerThread started.")
        while not self._stop_event.is_set():
            item_to_process: Optional[QueueItem] = None

            with self._lock:
                if not self._current_item and self._pending_queue:
                    item_to_process = self._pending_queue.pop(0)
                    self._current_item = item_to_process
                    item_to_process.status = "processing"
                    item_to_process.started_at = time.time()
                    item_to_process.message = "Ingesting video & processing with AI..."
                    self._save_to_disk()

            if not item_to_process:
                time.sleep(1)
                continue

            # Process the item
            self._process_item(item_to_process)

            with self._lock:
                if item_to_process.status in ("completed", "failed"):
                    self._completed_history.insert(0, item_to_process)
                self._current_item = None
                self._save_to_disk()

            time.sleep(1)

        logger.info("QueueWorkerThread ended.")

    def _process_item(self, item: QueueItem) -> None:
        """Executes pipeline for a single queue item and handles auto-upload."""
        logger.info("Starting processing for queue item: '%s' (%s)", item.title, item.url)
        try:
            from clippa.cli import run_pipeline

            before_clips = set(self.output_dir.glob("*.mp4"))

            item.message = "Running Whisper transcription and YuNet face tracking..."
            self._save_to_disk()

            exit_code = run_pipeline(
                url=item.url,
                output_dir=self.output_dir,
                temp_dir=self.temp_dir,
                template=item.template,
                max_clips=item.max_clips,
                max_duration=item.max_duration,
                nvenc=item.nvenc,
                auto_upload=item.auto_upload,
                upload_privacy=item.upload_privacy,
            )

            after_clips = set(self.output_dir.glob("*.mp4"))
            new_clips = sorted(list(after_clips - before_clips), key=lambda p: p.stat().st_mtime)

            clip_names = [p.name for p in new_clips]
            shorts_urls: List[str] = []
            scheduled_info: List[str] = []

            for p in new_clips:
                meta_file = p.with_name(f"{p.stem}.meta.json")
                if meta_file.exists():
                    try:
                        m_data = json.loads(meta_file.read_text(encoding="utf-8"))
                        yt = m_data.get("youtube", {})
                        if yt.get("uploaded") and yt.get("url"):
                            shorts_urls.append(yt["url"])
                            if yt.get("scheduled") and yt.get("publish_at_local"):
                                scheduled_info.append(yt["publish_at_local"])
                    except Exception:
                        pass

            item.finished_at = time.time()
            item.result_clips = clip_names
            item.shorts_urls = shorts_urls

            if exit_code == 0:
                item.status = "completed"
                msg = f"Successfully rendered {len(clip_names)} clips."
                if scheduled_info:
                    msg += f" 📅 Scheduled {len(scheduled_info)} Shorts on YouTube (starting {scheduled_info[0]})!"
                elif shorts_urls:
                    msg += f" 🚀 Published {len(shorts_urls)} Shorts to YouTube!"
                item.message = msg
                logger.info("Queue item completed: %s", msg)

                # Record in watcher history deduplication
                try:
                    from clippa.watcher import record_video_history
                    record_video_history(
                        video_id=item.video_id,
                        status="completed",
                        title=item.title,
                        channel_name=item.channel,
                        url=item.url,
                        clips=clip_names,
                        shorts_urls=shorts_urls,
                    )
                except Exception:
                    pass
            else:
                item.status = "failed"
                item.error = "Pipeline exited with an error code"
                item.message = "Pipeline failed. Check server logs."
                logger.error("Queue item failed for video: %s", item.title)

        except Exception as exc:
            logger.exception("Error processing queue item '%s': %s", item.title, exc)
            item.status = "failed"
            item.error = str(exc)
            item.finished_at = time.time()
            item.message = f"Error: {exc}"
