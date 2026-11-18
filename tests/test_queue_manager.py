"""
tests/test_queue_manager.py - Unit tests for ClippingQueueManager and QueueItem
"""

import json
from pathlib import Path
from unittest.mock import patch
import pytest

from clippa.queue_manager import ClippingQueueManager, QueueItem


def test_queue_item_serialization():
    item = QueueItem(
        video_id="vid123",
        title="Test Video Title",
        url="https://youtube.com/watch?v=vid123",
        channel="@MrBeast",
        duration=180,
    )
    d = item.to_dict()
    assert d["video_id"] == "vid123"
    assert d["title"] == "Test Video Title"
    assert d["auto_upload"] is True

    recovered = QueueItem.from_dict(d)
    assert recovered.video_id == "vid123"
    assert recovered.channel == "@MrBeast"
    assert recovered.duration == 180


def test_queue_operations(tmp_path, monkeypatch):
    queue_file = tmp_path / "queue.json"
    monkeypatch.setattr("clippa.queue_manager.QUEUE_FILE", queue_file)

    qm = ClippingQueueManager(output_dir=tmp_path / "outputs", temp_dir=tmp_path / "temp")

    # Initially empty
    state = qm.get_state()
    assert state["pending_count"] == 0
    assert state["current"] is None

    # Add item 1
    item1 = qm.add_to_queue(
        video_id="v1",
        title="Video 1",
        url="https://youtube.com/watch?v=v1",
    )
    assert qm.get_state()["pending_count"] == 1

    # Add item 2
    item2 = qm.add_to_queue(
        video_id="v2",
        title="Video 2",
        url="https://youtube.com/watch?v=v2",
    )
    assert qm.get_state()["pending_count"] == 2
    assert qm.get_state()["pending"][0]["video_id"] == "v1"
    assert qm.get_state()["pending"][1]["video_id"] == "v2"

    # Move item 2 up
    moved = qm.move_item(item2.id, "up")
    assert moved is True
    assert qm.get_state()["pending"][0]["video_id"] == "v2"
    assert qm.get_state()["pending"][1]["video_id"] == "v1"

    # Add item 3 with prioritize_first=True
    item3 = qm.add_to_queue(
        video_id="v3",
        title="Video 3",
        url="https://youtube.com/watch?v=v3",
        prioritize_first=True,
    )
    assert qm.get_state()["pending_count"] == 3
    assert qm.get_state()["pending"][0]["video_id"] == "v3"

    # Remove item 2
    removed = qm.remove_from_queue(item2.id)
    assert removed is True
    assert qm.get_state()["pending_count"] == 2

    # Clear pending
    cleared = qm.clear_pending()
    assert cleared == 2
    assert qm.get_state()["pending_count"] == 0


def test_queue_worker_execution(tmp_path, monkeypatch):
    queue_file = tmp_path / "queue.json"
    monkeypatch.setattr("clippa.queue_manager.QUEUE_FILE", queue_file)

    out_dir = tmp_path / "outputs"
    tmp_dir = tmp_path / "temp"
    out_dir.mkdir()
    tmp_dir.mkdir()

    qm = ClippingQueueManager(output_dir=out_dir, temp_dir=tmp_dir)

    def fake_pipeline(*args, **kwargs):
        # Create clip and meta
        c = out_dir / "clip_q.mp4"
        c.write_bytes(b"dummy")
        m = out_dir / "clip_q.meta.json"
        m.write_text(json.dumps({
            "title": "Clip Q",
            "youtube": {"uploaded": True, "url": "https://youtube.com/shorts/q_short"}
        }), encoding="utf-8")
        return 0

    item = qm.add_to_queue(
        video_id="vq1",
        title="Queued Video 1",
        url="https://youtube.com/watch?v=vq1",
    )

    with patch("clippa.cli.run_pipeline", side_effect=fake_pipeline):
        qm._process_item(item)
        assert item.status == "completed"
        assert item.shorts_urls == ["https://youtube.com/shorts/q_short"]
        assert len(item.result_clips) == 1
