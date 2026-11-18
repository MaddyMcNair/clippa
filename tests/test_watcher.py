"""
tests/test_watcher.py - Unit tests for Autopilot Channel Watcher & Deduplication
"""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from clippa.watcher import (
    normalize_channel_url,
    get_watchlist,
    save_watchlist,
    add_channel_to_watchlist,
    remove_channel_from_watchlist,
    get_history,
    save_history,
    is_video_processed,
    record_video_history,
    get_autopilot_settings,
    save_autopilot_settings,
    AutopilotDaemon,
)


def test_normalize_channel_url():
    assert normalize_channel_url("@MrBeast") == "https://www.youtube.com/@MrBeast/videos"
    assert normalize_channel_url("IShowSpeed") == "https://www.youtube.com/@IShowSpeed/videos"
    assert normalize_channel_url("https://www.youtube.com/@MrBeast") == "https://www.youtube.com/@MrBeast/videos"
    assert normalize_channel_url("https://www.youtube.com/@MrBeast/videos") == "https://www.youtube.com/@MrBeast/videos"
    assert normalize_channel_url("https://www.youtube.com/channel/UC12345") == "https://www.youtube.com/channel/UC12345/videos"


def test_watchlist_operations(tmp_path, monkeypatch):
    watch_file = tmp_path / "watchlist.json"
    monkeypatch.setattr("clippa.watcher.WATCHLIST_FILE", watch_file)

    # Empty initially
    assert get_watchlist() == []

    # Mock resolve_channel_info
    fake_ch = {
        "id": "UC_BEAST",
        "name": "MrBeast",
        "handle": "@MrBeast",
        "url": "https://www.youtube.com/@MrBeast",
        "avatar": "https://example.com/avatar.jpg",
        "added_at": 1000.0,
    }
    with patch("clippa.watcher.resolve_channel_info", return_value=fake_ch):
        added = add_channel_to_watchlist("@MrBeast")
        assert added["name"] == "MrBeast"

        # Check saved
        items = get_watchlist()
        assert len(items) == 1
        assert items[0]["handle"] == "@MrBeast"

        # Deduplicate add
        added2 = add_channel_to_watchlist("@MrBeast")
        assert len(get_watchlist()) == 1

        # Remove channel
        removed = remove_channel_from_watchlist("@MrBeast")
        assert removed is True
        assert len(get_watchlist()) == 0


def test_history_deduplication(tmp_path, monkeypatch):
    hist_file = tmp_path / "history.json"
    monkeypatch.setattr("clippa.watcher.HISTORY_FILE", hist_file)

    assert not is_video_processed("vid123")

    record_video_history(
        video_id="vid123",
        status="processing",
        title="Epic Video",
        channel_name="MrBeast",
        channel_handle="@MrBeast",
        url="https://youtube.com/watch?v=vid123",
    )

    assert is_video_processed("vid123")
    h = get_history()
    assert h["processed"]["vid123"]["status"] == "processing"
    assert h["processed"]["vid123"]["title"] == "Epic Video"

    # Update to completed with YouTube Short url
    record_video_history(
        video_id="vid123",
        status="completed",
        clips=["clip_1.mp4"],
        shorts_urls=["https://youtube.com/shorts/short123"],
    )

    h2 = get_history()
    assert h2["processed"]["vid123"]["status"] == "completed"
    assert h2["processed"]["vid123"]["uploaded"] is True
    assert h2["processed"]["vid123"]["shorts_urls"] == ["https://youtube.com/shorts/short123"]


def test_autopilot_settings(tmp_path, monkeypatch):
    settings_file = tmp_path / "autopilot_settings.json"
    monkeypatch.setattr("clippa.watcher.SETTINGS_FILE", settings_file)

    s = get_autopilot_settings()
    assert s["enabled"] is False
    assert s["auto_upload"] is True

    save_autopilot_settings({"enabled": True, "poll_interval_seconds": 300})
    s2 = get_autopilot_settings()
    assert s2["enabled"] is True
    assert s2["poll_interval_seconds"] == 300


def test_autopilot_daemon_status(tmp_path, monkeypatch):
    watch_file = tmp_path / "watchlist.json"
    hist_file = tmp_path / "history.json"
    settings_file = tmp_path / "autopilot_settings.json"
    monkeypatch.setattr("clippa.watcher.WATCHLIST_FILE", watch_file)
    monkeypatch.setattr("clippa.watcher.HISTORY_FILE", hist_file)
    monkeypatch.setattr("clippa.watcher.SETTINGS_FILE", settings_file)

    daemon = AutopilotDaemon(output_dir=tmp_path / "outputs", temp_dir=tmp_path / "temp")
    status = daemon.get_status()
    assert status["enabled"] is False
    assert status["watched_channels_count"] == 0
    assert status["total_processed_videos"] == 0

    daemon.log_activity("Test event")
    feed = daemon.get_activity_feed()
    assert len(feed) == 1
    assert feed[0]["message"] == "Test event"


def test_autopilot_daemon_poll_once(tmp_path, monkeypatch):
    watch_file = tmp_path / "watchlist.json"
    hist_file = tmp_path / "history.json"
    settings_file = tmp_path / "autopilot_settings.json"
    monkeypatch.setattr("clippa.watcher.WATCHLIST_FILE", watch_file)
    monkeypatch.setattr("clippa.watcher.HISTORY_FILE", hist_file)
    monkeypatch.setattr("clippa.watcher.SETTINGS_FILE", settings_file)

    save_watchlist([{
        "id": "UC_BEAST",
        "name": "MrBeast",
        "handle": "@MrBeast",
        "url": "https://www.youtube.com/@MrBeast",
    }])

    daemon = AutopilotDaemon(output_dir=tmp_path / "outputs", temp_dir=tmp_path / "temp")
    (tmp_path / "outputs").mkdir()
    (tmp_path / "temp").mkdir()

    fake_videos = [
        {"id": "new_vid_1", "title": "New Video 1", "url": "https://youtube.com/watch?v=new_vid_1"},
    ]

    def fake_pipeline(*args, **kwargs):
        # Simulate generating a clip and meta file
        clip = (tmp_path / "outputs" / "clip_1.mp4")
        clip.write_bytes(b"dummy mp4")
        meta = (tmp_path / "outputs" / "clip_1.meta.json")
        meta.write_text(json.dumps({
            "title": "New Video 1",
            "youtube": {
                "uploaded": True,
                "url": "https://youtube.com/shorts/auto_short_123",
            }
        }), encoding="utf-8")
        return 0

    with patch("clippa.watcher.fetch_recent_channel_videos", return_value=fake_videos), \
         patch("clippa.cli.run_pipeline", side_effect=fake_pipeline):
        processed = daemon.poll_once()
        assert len(processed) == 1
        assert processed[0]["video_id"] == "new_vid_1"
        assert processed[0]["shorts"] == ["https://youtube.com/shorts/auto_short_123"]
        assert is_video_processed("new_vid_1")

        # Second poll should find it already processed and skip
        second_pass = daemon.poll_once()
        assert len(second_pass) == 0

