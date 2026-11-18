"""
tests/test_web_queue.py - Integration tests for YouTuber Video Browser & Queue Web Endpoints
"""

import json
from pathlib import Path
from unittest.mock import patch
import pytest
from fastapi.testclient import TestClient

from clippa.web import create_web_app


@pytest.fixture
def client(tmp_path: Path, monkeypatch):
    out_dir = tmp_path / "outputs"
    tmp_dir = tmp_path / "temp"
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    queue_file = tmp_path / "queue.json"
    watch_file = tmp_path / "watchlist.json"
    hist_file = tmp_path / "history.json"
    settings_file = tmp_path / "autopilot_settings.json"

    monkeypatch.setattr("clippa.queue_manager.QUEUE_FILE", queue_file)
    monkeypatch.setattr("clippa.watcher.WATCHLIST_FILE", watch_file)
    monkeypatch.setattr("clippa.watcher.HISTORY_FILE", hist_file)
    monkeypatch.setattr("clippa.watcher.SETTINGS_FILE", settings_file)

    app = create_web_app(output_dir=out_dir, temp_dir=tmp_dir)
    return TestClient(app)


def test_get_channel_videos_endpoint(client):
    fake_videos = [
        {"id": "vid1", "title": "MrBeast Video 1", "url": "https://youtube.com/watch?v=vid1", "duration": 600, "thumbnail": ""},
        {"id": "vid2", "title": "MrBeast Video 2", "url": "https://youtube.com/watch?v=vid2", "duration": 800, "thumbnail": ""},
    ]
    with patch("clippa.web.fetch_recent_channel_videos", return_value=fake_videos):
        resp = client.get("/api/channels/@MrBeast/videos")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["videos"]) == 2
        assert data["videos"][0]["id"] == "vid1"


def test_queue_endpoints_lifecycle(client):
    # 1. Initially empty queue
    resp = client.get("/api/queue")
    assert resp.status_code == 200
    assert resp.json()["pending_count"] == 0

    # 2. Add video 1 to queue
    add_resp1 = client.post("/api/queue", json={
        "video_id": "v1",
        "title": "Episode 1",
        "url": "https://youtube.com/watch?v=v1",
        "channel": "@MrBeast",
        "auto_upload": True,
    })
    assert add_resp1.status_code == 200
    assert add_resp1.json()["status"] == "queued"
    item1_id = add_resp1.json()["item"]["id"]

    # 3. Add video 2 to queue
    add_resp2 = client.post("/api/queue", json={
        "video_id": "v2",
        "title": "Episode 2",
        "url": "https://youtube.com/watch?v=v2",
        "channel": "@MrBeast",
        "auto_upload": True,
    })
    assert add_resp2.status_code == 200
    item2_id = add_resp2.json()["item"]["id"]

    # 4. Verify count
    q_state = client.get("/api/queue").json()
    assert q_state["pending_count"] == 2
    assert q_state["pending"][0]["id"] == item1_id
    assert q_state["pending"][1]["id"] == item2_id

    # 5. Reorder: move item 2 up
    reorder_resp = client.post("/api/queue/reorder", json={
        "item_id": item2_id,
        "direction": "up",
    })
    assert reorder_resp.status_code == 200
    q_state2 = client.get("/api/queue").json()
    assert q_state2["pending"][0]["id"] == item2_id
    assert q_state2["pending"][1]["id"] == item1_id

    # 6. Delete item from queue
    del_resp = client.delete(f"/api/queue/{item2_id}")
    assert del_resp.status_code == 200
    q_state3 = client.get("/api/queue").json()
    assert q_state3["pending_count"] == 1

    # 7. Clear queue
    clear_resp = client.post("/api/queue/clear")
    assert clear_resp.status_code == 200
    assert client.get("/api/queue").json()["pending_count"] == 0


def test_retry_queue_endpoint(client, monkeypatch):
    from clippa.queue_manager import ClippingQueueManager, QueueItem
    qm = ClippingQueueManager.get_instance()
    failed_item = QueueItem(
        video_id="v_fail",
        title="Failed Episode",
        url="https://youtube.com/watch?v=v_fail",
        item_id="q_failed_1",
        status="failed",
        error="Network error",
    )
    with qm._lock:
        qm._completed_history.append(failed_item)

    # Retry the failed item
    resp = client.post("/api/queue/retry/q_failed_1")
    assert resp.status_code == 200
    assert resp.json()["status"] == "retrying"

    # Verify it is now in pending queue
    q_state = client.get("/api/queue").json()
    assert any(x["id"] == "q_failed_1" for x in q_state["pending"])
