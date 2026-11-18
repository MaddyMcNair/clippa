"""
tests/test_uploader.py - Unit Tests for YouTube Data API v3 OAuth & Uploading
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from clippa.uploader import (
    CONFIG_DIR,
    CLIENT_SECRETS_FILE,
    TOKEN_FILE,
    YouTubeAuthError,
    disconnect_account,
    get_channel_info,
    is_authenticated,
    save_client_secrets,
    upload_video_to_youtube,
)


def test_is_authenticated_false_when_no_token(tmp_path, monkeypatch):
    monkeypatch.setattr("clippa.uploader.TOKEN_FILE", tmp_path / "nonexistent.json")
    assert not is_authenticated()


def test_save_client_secrets_valid_json(tmp_path, monkeypatch):
    dummy_file = tmp_path / "client_secrets.json"
    monkeypatch.setattr("clippa.uploader.CONFIG_DIR", tmp_path)
    monkeypatch.setattr("clippa.uploader.CLIENT_SECRETS_FILE", dummy_file)

    valid_dict = {
        "installed": {
            "client_id": "test-client-id",
            "client_secret": "test-secret",
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
        }
    }
    saved_path = save_client_secrets(valid_dict)
    assert saved_path.exists()
    content = json.loads(saved_path.read_text(encoding="utf-8"))
    assert content["installed"]["client_id"] == "test-client-id"


def test_save_client_secrets_invalid_format(tmp_path, monkeypatch):
    monkeypatch.setattr("clippa.uploader.CONFIG_DIR", tmp_path)
    monkeypatch.setattr("clippa.uploader.CLIENT_SECRETS_FILE", tmp_path / "secrets.json")

    with pytest.raises(ValueError, match="Invalid client_secrets.json format"):
        save_client_secrets({"invalid_key": "data"})


def test_disconnect_account(tmp_path, monkeypatch):
    token = tmp_path / "youtube_credentials.json"
    token.write_text("{}", encoding="utf-8")
    monkeypatch.setattr("clippa.uploader.TOKEN_FILE", token)

    assert token.exists()
    assert disconnect_account() is True
    assert not token.exists()


def test_get_channel_info_unauthenticated(tmp_path, monkeypatch):
    monkeypatch.setattr("clippa.uploader.TOKEN_FILE", tmp_path / "none.json")
    info = get_channel_info()
    assert info["connected"] is False


@patch("clippa.uploader.get_stored_credentials")
@patch("googleapiclient.discovery.build")
def test_upload_video_mocked(mock_build, mock_get_creds, tmp_path):
    mock_get_creds.return_value = MagicMock()

    dummy_video = tmp_path / "clip.mp4"
    dummy_video.write_bytes(b"\x00" * 1024)

    mock_service = MagicMock()
    mock_request = MagicMock()
    mock_request.next_chunk.return_value = (None, {
        "id": "abc123xyz",
        "snippet": {"publishedAt": "2026-11-14T12:00:00Z"},
    })
    mock_service.videos().insert.return_value = mock_request
    mock_build.return_value = mock_service

    progress_records = []
    result = upload_video_to_youtube(
        video_path=dummy_video,
        title="Epic Tennis Shot #Shorts",
        description="Check this out!",
        tags=["#Shorts", "#Tennis"],
        privacy_status="unlisted",
        progress_callback=lambda p: progress_records.append(p),
    )

    assert result["video_id"] == "abc123xyz"
    assert result["url"] == "https://youtube.com/shorts/abc123xyz"
    assert result["privacy"] == "unlisted"


@patch("clippa.uploader.get_stored_credentials")
@patch("googleapiclient.discovery.build")
def test_upload_video_scheduled_forces_private_and_publish_at(mock_build, mock_get_creds, tmp_path):
    mock_get_creds.return_value = MagicMock()

    dummy_video = tmp_path / "clip_sched.mp4"
    dummy_video.write_bytes(b"\x00" * 1024)

    mock_service = MagicMock()
    mock_request = MagicMock()
    mock_request.next_chunk.return_value = (None, {
        "id": "sched123",
        "snippet": {"publishedAt": "2026-11-14T12:00:00Z"},
        "status": {"publishAt": "2026-11-15T04:30:00Z", "privacyStatus": "private"}
    })
    mock_service.videos().insert.return_value = mock_request
    mock_build.return_value = mock_service

    result = upload_video_to_youtube(
        video_path=dummy_video,
        title="Scheduled Short #Shorts",
        description="Auto scheduled",
        privacy_status="public",  # should be overridden to private because publish_at is set
        publish_at="2026-11-15T04:30:00Z",
    )

    assert result["video_id"] == "sched123"
    assert result["scheduled_for"] == "2026-11-15T04:30:00Z"
    assert result["privacy"] == "private"

    # Verify the API payload sent to YouTube videos().insert
    call_kwargs = mock_service.videos().insert.call_args[1]
    body = call_kwargs["body"]
    assert body["status"]["privacyStatus"] == "private"
    assert body["status"]["publishAt"] == "2026-11-15T04:30:00Z"

