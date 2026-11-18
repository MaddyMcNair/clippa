"""
tests/test_web_autopilot.py - Integration tests for Web Autopilot & Channel Watcher endpoints
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

    watch_file = tmp_path / "watchlist.json"
    hist_file = tmp_path / "history.json"
    settings_file = tmp_path / "autopilot_settings.json"
    monkeypatch.setattr("clippa.watcher.WATCHLIST_FILE", watch_file)
    monkeypatch.setattr("clippa.watcher.HISTORY_FILE", hist_file)
    monkeypatch.setattr("clippa.watcher.SETTINGS_FILE", settings_file)

    app = create_web_app(output_dir=out_dir, temp_dir=tmp_dir)
    return TestClient(app)


def test_channels_api_crud(client):
    # Initial list is empty
    resp = client.get("/api/channels")
    assert resp.status_code == 200
    assert resp.json()["channels"] == []

    # Add channel with mocked resolve
    fake_ch = {
        "id": "UC_TEST",
        "name": "TestChannel",
        "handle": "@TestChannel",
        "url": "https://www.youtube.com/@TestChannel",
        "avatar": "",
        "added_at": 1000.0,
    }
    with patch("clippa.watcher.resolve_channel_info", return_value=fake_ch):
        add_resp = client.post("/api/channels", json={"channel": "@TestChannel"})
        assert add_resp.status_code == 200
        assert add_resp.json()["status"] == "added"
        assert add_resp.json()["channel"]["name"] == "TestChannel"

    # Verify present in list
    list_resp = client.get("/api/channels")
    assert len(list_resp.json()["channels"]) == 1

    # Delete channel
    del_resp = client.delete("/api/channels/UC_TEST")
    assert del_resp.status_code == 200
    assert del_resp.json()["status"] == "removed"

    # Verify list is empty again
    list_resp2 = client.get("/api/channels")
    assert len(list_resp2.json()["channels"]) == 0


def test_autopilot_status_and_toggle(client):
    # Check status
    st_resp = client.get("/api/autopilot/status")
    assert st_resp.status_code == 200
    data = st_resp.json()
    assert "daemon_alive" in data
    assert data["enabled"] is False
    assert "activity" in data

    # Toggle autopilot ON
    tog_resp = client.post("/api/autopilot/toggle", json={"enabled": True})
    assert tog_resp.status_code == 200
    assert tog_resp.json()["settings"]["enabled"] is True

    # Check status reflects ON
    st_resp2 = client.get("/api/autopilot/status")
    assert st_resp2.json()["enabled"] is True


def test_autopilot_check_now_and_history(client):
    # Trigger check now
    with patch("clippa.watcher.AutopilotDaemon.poll_once", return_value=[]):
        chk_resp = client.post("/api/autopilot/check-now")
        assert chk_resp.status_code == 200
        assert chk_resp.json()["status"] == "started"

    # Check history
    hist_resp = client.get("/api/autopilot/history")
    assert hist_resp.status_code == 200
    assert "processed" in hist_resp.json()
