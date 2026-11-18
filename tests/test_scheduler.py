"""
tests/test_scheduler.py - Unit tests for PublishScheduler (Anti-Shadowban YouTube Scheduling)
"""

from datetime import datetime, timezone, timedelta
from pathlib import Path
import pytest

from clippa.scheduler import PublishScheduler


def test_default_slots_and_cap(tmp_path):
    sched_file = tmp_path / "test_sched.json"
    scheduler = PublishScheduler(schedule_file=sched_file)

    assert scheduler.slot_times == ["10:00", "14:00", "18:00", "21:00"]
    assert scheduler.max_daily_posts == 4


def test_custom_settings(tmp_path):
    sched_file = tmp_path / "test_sched.json"
    scheduler = PublishScheduler(schedule_file=sched_file)

    scheduler.update_settings(max_daily_posts=3, slot_times=["09:00", "15:00", "20:00"])
    assert scheduler.slot_times == ["09:00", "15:00", "20:00"]
    assert scheduler.max_daily_posts == 3

    # Ensure persisted and reloaded from disk
    reloaded = PublishScheduler(schedule_file=sched_file)
    assert reloaded.slot_times == ["09:00", "15:00", "20:00"]
    assert reloaded.max_daily_posts == 3


def test_intraday_slot_selection_morning(tmp_path):
    sched_file = tmp_path / "test_sched.json"
    scheduler = PublishScheduler(schedule_file=sched_file)

    # Local morning 8:00 AM (UTC+5:30)
    ist = timezone(timedelta(hours=5, minutes=30))
    mock_now = datetime(2026, 11, 15, 8, 0, 0, tzinfo=ist)

    slot_dt, iso_utc = scheduler.get_next_available_slot(reference_time=mock_now)
    assert slot_dt.date().isoformat() == "2026-11-15"
    assert slot_dt.strftime("%H:%M") == "10:00"
    assert iso_utc.endswith("Z")
    # 10:00 IST is 04:30 UTC
    assert iso_utc == "2026-11-15T04:30:00Z"


def test_intraday_slot_selection_skips_past_and_buffer(tmp_path):
    sched_file = tmp_path / "test_sched.json"
    scheduler = PublishScheduler(schedule_file=sched_file)

    ist = timezone(timedelta(hours=5, minutes=30))
    # Local 10:10 AM (10:00 is in the past, <15min buffer) -> next should be 14:00
    mock_now = datetime(2026, 11, 15, 10, 10, 0, tzinfo=ist)

    slot_dt, iso_utc = scheduler.get_next_available_slot(reference_time=mock_now)
    assert slot_dt.date().isoformat() == "2026-11-15"
    assert slot_dt.strftime("%H:%M") == "14:00"
    # 14:00 IST is 08:30 UTC
    assert iso_utc == "2026-11-15T08:30:00Z"


def test_multi_day_rollover_when_today_full(tmp_path):
    sched_file = tmp_path / "test_sched.json"
    scheduler = PublishScheduler(schedule_file=sched_file)

    ist = timezone(timedelta(hours=5, minutes=30))
    mock_now = datetime(2026, 11, 15, 9, 0, 0, tzinfo=ist)

    # Record all 4 daily slots for today
    for s_time in ["10:00", "14:00", "18:00", "21:00"]:
        h, m = map(int, s_time.split(":"))
        slot_dt = datetime(2026, 11, 15, h, m, 0, tzinfo=ist)
        scheduler.record_scheduled_slot(
            slot_dt=slot_dt,
            video_id=f"vid_{s_time}",
            clip_name=f"clip_{s_time}.mp4",
            title=f"Title {s_time}",
        )

    # Next available slot should now automatically roll over to tomorrow at 10:00 AM
    next_dt, iso_utc = scheduler.get_next_available_slot(reference_time=mock_now)
    assert next_dt.date().isoformat() == "2026-11-16"
    assert next_dt.strftime("%H:%M") == "10:00"
    assert iso_utc == "2026-11-16T04:30:00Z"


def test_consecutive_recording_advances_slots(tmp_path):
    sched_file = tmp_path / "test_sched.json"
    scheduler = PublishScheduler(schedule_file=sched_file)

    ist = timezone(timedelta(hours=5, minutes=30))
    mock_now = datetime(2026, 11, 15, 8, 0, 0, tzinfo=ist)

    # Slot 1 -> 10:00
    s1_dt, _ = scheduler.get_next_available_slot(reference_time=mock_now)
    assert s1_dt.strftime("%H:%M") == "10:00"
    scheduler.record_scheduled_slot(s1_dt, "vid1", "clip1.mp4", "Clip 1")

    # Slot 2 -> 14:00
    s2_dt, _ = scheduler.get_next_available_slot(reference_time=mock_now)
    assert s2_dt.strftime("%H:%M") == "14:00"
    scheduler.record_scheduled_slot(s2_dt, "vid2", "clip2.mp4", "Clip 2")

    # Slot 3 -> 18:00
    s3_dt, _ = scheduler.get_next_available_slot(reference_time=mock_now)
    assert s3_dt.strftime("%H:%M") == "18:00"
    scheduler.record_scheduled_slot(s3_dt, "vid3", "clip3.mp4", "Clip 3")

    # Slot 4 -> 21:00
    s4_dt, _ = scheduler.get_next_available_slot(reference_time=mock_now)
    assert s4_dt.strftime("%H:%M") == "21:00"
    scheduler.record_scheduled_slot(s4_dt, "vid4", "clip4.mp4", "Clip 4")

    # Slot 5 -> Tomorrow 10:00
    s5_dt, _ = scheduler.get_next_available_slot(reference_time=mock_now)
    assert s5_dt.date().isoformat() == "2026-11-16"
    assert s5_dt.strftime("%H:%M") == "10:00"


def test_schedule_overview_and_clear(tmp_path):
    sched_file = tmp_path / "test_sched.json"
    scheduler = PublishScheduler(schedule_file=sched_file)

    ist = timezone(timedelta(hours=5, minutes=30))
    now = datetime.now(ist)
    scheduler.record_scheduled_slot(now, "vid_today", "today.mp4", "Today Clip")

    overview = scheduler.get_schedule_overview()
    assert overview["daily_limit"] == 4
    assert overview["today"]["count"] == 1
    assert overview["today"]["remaining_capacity"] == 3
    assert overview["total_scheduled"] == 1

    cleared = scheduler.clear_schedule()
    assert cleared == 1
    assert len(scheduler.scheduled_slots) == 0
