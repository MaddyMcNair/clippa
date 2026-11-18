"""
tests/test_web_upload_all.py - Verification for Upload All button in Generated Shorts Gallery
"""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
from fastapi.testclient import TestClient

from clippa.web import create_web_app


@pytest.fixture
def web_env(tmp_path: Path):
    out_dir = tmp_path / "outputs"
    tmp_dir = tmp_path / "temp"
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    app = create_web_app(output_dir=out_dir, temp_dir=tmp_dir)
    client = TestClient(app)
    return client, out_dir, tmp_dir


def test_dashboard_html_contains_upload_all_button():
    """Verifies that dashboard.html contains the Upload All button and uploadAllClips handler."""
    dashboard_file = Path("clippa/dashboard.html")
    assert dashboard_file.exists()
    content = dashboard_file.read_text(encoding="utf-8")

    assert 'id="btnUploadAll"' in content
    assert "uploadAllClips()" in content
    assert "Upload All" in content


def test_upload_all_endpoint_unauthenticated(web_env):
    """Verifies that /api/youtube/upload-all returns 401 when account is not connected."""
    client, out_dir, _ = web_env
    with patch("clippa.web.is_authenticated", return_value=False):
        resp = client.post("/api/youtube/upload-all")
        assert resp.status_code == 401
        assert "not connected" in resp.json()["detail"].lower()


def test_upload_all_endpoint_empty_gallery(web_env):
    """Verifies that upload-all handles an empty gallery cleanly."""
    client, out_dir, _ = web_env
    with patch("clippa.web.is_authenticated", return_value=True):
        resp = client.post("/api/youtube/upload-all")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "completed"
        assert data["total_processed"] == 0
        assert data["uploaded"] == []


def test_upload_all_endpoint_success_and_skips_already_uploaded(web_env):
    """Verifies that upload-all uploads unuploaded clips and skips already uploaded clips."""
    client, out_dir, _ = web_env

    # 1. Create a dummy clip that is already uploaded
    clip1 = out_dir / "clip1.mp4"
    clip1.write_bytes(b"dummy mp4 1")
    meta1 = out_dir / "clip1.meta.json"
    meta1.write_text(json.dumps({
        "title": "Clip One",
        "youtube_title": "Clip One #Shorts",
        "youtube": {"uploaded": True, "url": "https://youtube.com/shorts/abc"}
    }), encoding="utf-8")

    # 2. Create two clips that are pending upload
    clip2 = out_dir / "clip2.mp4"
    clip2.write_bytes(b"dummy mp4 2")
    meta2 = out_dir / "clip2.meta.json"
    meta2.write_text(json.dumps({
        "title": "Clip Two",
        "youtube_title": "Clip Two #Shorts",
        "youtube": {"uploaded": False}
    }), encoding="utf-8")

    clip3 = out_dir / "clip3.mp4"
    clip3.write_bytes(b"dummy mp4 3")
    meta3 = out_dir / "clip3.meta.json"
    meta3.write_text(json.dumps({
        "title": "Clip Three",
        "youtube_title": "Clip Three #Shorts",
    }), encoding="utf-8")

    fake_result = {
        "video_id": "xyz123",
        "url": "https://youtube.com/shorts/xyz123",
        "published_at": "2026-11-14T12:00:00Z"
    }

    with patch("clippa.web.is_authenticated", return_value=True), \
         patch("clippa.web.upload_video_to_youtube", return_value=fake_result) as mock_upload:

        resp = client.post("/api/youtube/upload-all")
        assert resp.status_code == 200
        data = resp.json()

        assert data["status"] == "completed"
        assert len(data["uploaded"]) == 2
        assert mock_upload.call_count == 2

        # Verify companion meta files were updated
        d2 = json.loads(meta2.read_text(encoding="utf-8"))
        assert d2["youtube"]["uploaded"] is True
        assert d2["youtube"]["video_id"] == "xyz123"

        d3 = json.loads(meta3.read_text(encoding="utf-8"))
        assert d3["youtube"]["uploaded"] is True
        assert d3["youtube"]["video_id"] == "xyz123"

        # Clip1 was already uploaded and should not have been re-uploaded
        d1 = json.loads(meta1.read_text(encoding="utf-8"))
        assert d1["youtube"]["url"] == "https://youtube.com/shorts/abc"
