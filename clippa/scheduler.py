"""
clippa/scheduler.py - Intelligent YouTube Scheduled Uploads & Anti-Shadowban Engine

Features:
- Enforces a safe daily publishing cap (default: 4 videos/day) to prevent YouTube spam shadowbans.
- Evenly spaces video releases across peak engagement hours (default: 10:00, 14:00, 18:00, 21:00 local time).
- Intraday slot rollover: skips past slots for today and cascades excess clips to tomorrow (and Day+2).
- Native YouTube Data API v3 integration: formats UTC ISO 8601 timestamps for `publishAt`
  so YouTube cloud servers publish videos automatically without requiring CLIPPA to run 24/7.
- Persists schedule calendar in ~/.clippa/schedule.json.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, time as dtime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("clippa.scheduler")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

CONFIG_DIR = Path.home() / ".clippa"
SCHEDULE_FILE = CONFIG_DIR / "schedule.json"


def ensure_config_dir() -> Path:
    """Ensures ~/.clippa directory exists."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    return CONFIG_DIR


class PublishScheduler:
    """
    Manages daily video publishing slots and converts local time slots
    into UTC ISO 8601 timestamps for YouTube Data API v3 publishAt.
    """

    DEFAULT_SLOTS = ["10:00", "14:00", "18:00", "21:00"]
    DEFAULT_MAX_DAILY_POSTS = 4

    _instance: Optional[PublishScheduler] = None

    @classmethod
    def get_instance(cls) -> PublishScheduler:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self, schedule_file: Optional[Path] = None, config_file: Optional[Path] = None):
        self.schedule_file = schedule_file or config_file or SCHEDULE_FILE
        self._load()

    def _load(self) -> None:
        """Loads schedule from disk or initializes defaults."""
        self.schedule_file.parent.mkdir(parents=True, exist_ok=True)
        self.max_daily_posts: int = self.DEFAULT_MAX_DAILY_POSTS
        self.slot_times: List[str] = list(self.DEFAULT_SLOTS)
        self.scheduled_slots: List[Dict[str, Any]] = []

        if self.schedule_file.exists():
            try:
                with open(self.schedule_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.max_daily_posts = int(data.get("max_daily_posts", self.DEFAULT_MAX_DAILY_POSTS))
                self.slot_times = list(data.get("slot_times", self.DEFAULT_SLOTS))
                self.scheduled_slots = list(data.get("scheduled_slots", []))
            except Exception as exc:
                logger.warning("Failed to load %s, using defaults: %s", self.schedule_file, exc)

    def _save(self) -> None:
        """Persists schedule state to disk."""
        ensure_config_dir()
        try:
            payload = {
                "max_daily_posts": self.max_daily_posts,
                "slot_times": self.slot_times,
                "scheduled_slots": self.scheduled_slots,
            }
            with open(self.schedule_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except Exception as exc:
            logger.warning("Failed to save schedule to %s: %s", self.schedule_file, exc)

    def get_slots_for_date(self, target_date: date) -> List[Dict[str, Any]]:
        """Returns all scheduled slots allocated on a specific calendar date (YYYY-MM-DD)."""
        date_str = target_date.isoformat()
        return [s for s in self.scheduled_slots if s.get("date") == date_str]

    def get_next_available_slot(
        self,
        min_buffer_minutes: int = 15,
        reference_time: Optional[datetime] = None,
    ) -> Tuple[datetime, str]:
        """
        Calculates the next open publishing slot.
        Cascades from today to tomorrow to subsequent days until an open slot is found.

        Returns:
            Tuple[datetime, str]:
                - datetime: Local timezone-aware datetime of the slot.
                - str: UTC ISO 8601 string formatted for YouTube API publishAt (e.g. '2026-11-15T08:30:00Z').
        """
        now = (reference_time or datetime.now().astimezone()).astimezone()
        current_date = now.date()

        # Search forward day by day (up to 365 days max)
        for _ in range(365):
            existing_slots = self.get_slots_for_date(current_date)
            allocated_times = {s.get("time") for s in existing_slots}

            # Check if this day still has capacity
            if len(existing_slots) < self.max_daily_posts:
                for slot_str in self.slot_times:
                    if slot_str in allocated_times:
                        continue

                    hour, minute = map(int, slot_str.split(":"))
                    candidate = datetime.combine(
                        current_date,
                        dtime(hour, minute),
                        tzinfo=now.tzinfo,
                    )

                    # Slot must be at least min_buffer_minutes in the future
                    if candidate > now + timedelta(minutes=min_buffer_minutes):
                        # Convert to UTC ISO 8601 for YouTube publishAt
                        utc_dt = candidate.astimezone(timezone.utc)
                        iso_utc = utc_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
                        return candidate, iso_utc

            # Advance to next day
            current_date += timedelta(days=1)

        # Fallback to tomorrow 10:00 AM
        fallback = datetime.combine(now.date() + timedelta(days=1), dtime(10, 0), tzinfo=now.tzinfo)
        return fallback, fallback.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def record_scheduled_slot(
        self,
        slot_dt: datetime,
        video_id: str,
        clip_name: str,
        title: str,
        youtube_url: str = "",
    ) -> Dict[str, Any]:
        """
        Records a newly scheduled upload slot in the persistent schedule ledger.
        """
        local_dt = slot_dt.astimezone()
        utc_dt = slot_dt.astimezone(timezone.utc)
        iso_utc = utc_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

        entry = {
            "id": f"slot_{local_dt.strftime('%Y%m%d_%H%M')}_{video_id[:6]}",
            "date": local_dt.date().isoformat(),
            "time": local_dt.strftime("%H:%M"),
            "publish_at_local": local_dt.strftime("%Y-%m-%d %I:%M %p"),
            "publish_at_utc": iso_utc,
            "video_id": video_id,
            "clip_name": clip_name,
            "title": title,
            "url": youtube_url,
            "scheduled_at": datetime.now().astimezone().isoformat(),
        }

        # Keep sorted by date and time
        self.scheduled_slots.append(entry)
        self.scheduled_slots.sort(key=lambda s: (s.get("date", ""), s.get("time", "")))
        self._save()
        logger.info(
            "Scheduled '%s' on YouTube for %s (%s UTC)",
            title,
            entry["publish_at_local"],
            iso_utc,
        )
        return entry

    def get_schedule_overview(self) -> Dict[str, Any]:
        """
        Returns structured calendar overview of upcoming releases for the dashboard UI.
        """
        now = datetime.now().astimezone()
        today_str = now.date().isoformat()
        tomorrow_str = (now.date() + timedelta(days=1)).isoformat()

        today_slots = [s for s in self.scheduled_slots if s.get("date") == today_str]
        tomorrow_slots = [s for s in self.scheduled_slots if s.get("date") == tomorrow_str]
        future_slots = [s for s in self.scheduled_slots if s.get("date") > tomorrow_str]

        # Group all upcoming slots by date
        by_date: Dict[str, List[Dict[str, Any]]] = {}
        for s in self.scheduled_slots:
            d = s.get("date", "Unknown")
            if d not in by_date:
                by_date[d] = []
            by_date[d].append(s)

        next_slot_info = None
        try:
            next_dt, next_utc = self.get_next_available_slot(reference_time=now)
            next_slot_info = {
                "local_formatted": next_dt.strftime("%b %d, %I:%M %p"),
                "iso_utc": next_utc,
            }
        except Exception as err:
            logger.debug("Could not calculate next slot: %s", err)

        return {
            "daily_limit": self.max_daily_posts,
            "slot_times": self.slot_times,
            "today_count": len(today_slots),
            "tomorrow_count": len(tomorrow_slots),
            "next_slot": next_slot_info,
            "today": {
                "date": today_str,
                "count": len(today_slots),
                "slots": today_slots,
                "remaining_capacity": max(0, self.max_daily_posts - len(today_slots)),
            },
            "tomorrow": {
                "date": tomorrow_str,
                "count": len(tomorrow_slots),
                "slots": tomorrow_slots,
                "remaining_capacity": max(0, self.max_daily_posts - len(tomorrow_slots)),
            },
            "total_scheduled": len(self.scheduled_slots),
            "calendar_by_date": by_date,
        }

    def update_settings(
        self,
        max_daily_posts: Optional[int] = None,
        slot_times: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Updates daily limits and slot hours."""
        if max_daily_posts is not None and max_daily_posts > 0:
            self.max_daily_posts = int(max_daily_posts)
        if slot_times is not None and len(slot_times) > 0:
            # Validate HH:MM format
            valid = []
            for t in slot_times:
                parts = t.strip().split(":")
                if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                    valid.append(f"{int(parts[0]):02d}:{int(parts[1]):02d}")
            if valid:
                self.slot_times = sorted(valid)

        self._save()
        return {
            "max_daily_posts": self.max_daily_posts,
            "slot_times": self.slot_times,
        }

    def clear_schedule(self) -> int:
        """Clears all scheduled slots."""
        count = len(self.scheduled_slots)
        self.scheduled_slots = []
        self._save()
        return count
