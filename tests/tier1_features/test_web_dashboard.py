"""
tests/tier1_features/test_web_dashboard.py
Test suite for CLIPPA Web Dashboard API and Frontend
"""

from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from clippa.web import create_web_app


@pytest.fixture
def test_client(tmp_path: Path):
    out_dir = tmp_path / "outputs"
    tmp_dir = tmp_path / "temp"
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Create a dummy clip
    dummy_clip = out_dir / "Sample_Clip_Title.mp4"
    dummy_clip.write_bytes(b"\x00" * 1024)

    app = create_web_app(output_dir=out_dir, temp_dir=tmp_dir)
    return TestClient(app)


def test_web_index_loads_html(test_client):
    """GET / returns responsive dark-themed dashboard HTML."""
    resp = test_client.get("/")
    assert resp.status_code == 200
    assert "CLIPPA" in resp.text
    assert "<video" in resp.text
    assert "Generate 9:16 Clips" in resp.text


def test_api_list_clips(test_client):
    """GET /api/clips lists rendered videos with metadata."""
    resp = test_client.get("/api/clips")
    assert resp.status_code == 200
    data = resp.json()
    assert "clips" in data
    assert len(data["clips"]) >= 1
    assert data["clips"][0]["filename"] == "Sample_Clip_Title.mp4"
    assert data["clips"][0]["title"] == "Sample Clip Title"
    assert "/videos/Sample_Clip_Title.mp4" in data["clips"][0]["url"]


def test_api_generate_empty_url_400(test_client):
    """POST /api/generate with empty URL rejects with HTTP 400."""
    resp = test_client.post("/api/generate", json={"url": ""})
    assert resp.status_code == 400


def test_api_generate_and_track_job(test_client):
    """POST /api/generate queues job and GET /api/jobs/{id} tracks progress."""
    resp = test_client.post(
        "/api/generate",
        json={
            "url": "https://www.youtube.com/watch?v=sample",
            "template": "mrbeast",
            "max_clips": 1,
        },
    )
    assert resp.status_code == 200
    job_id = resp.json().get("job_id")
    assert job_id is not None

    status_resp = test_client.get(f"/api/jobs/{job_id}")
    assert status_resp.status_code == 200
    assert status_resp.json()["id"] == job_id
    assert status_resp.json()["template"] == "mrbeast"


def test_api_job_not_found_404(test_client):
    """GET /api/jobs/nonexistent returns 404."""
    resp = test_client.get("/api/jobs/unknown-id")
    assert resp.status_code == 404


def test_api_delete_clip(test_client):
    """DELETE /api/clips/{filename} removes clip and returns 200."""
    resp = test_client.delete("/api/clips/Sample_Clip_Title.mp4")
    assert resp.status_code == 200
    assert resp.json()["status"] == "deleted"

    # Verify clip is gone
    list_resp = test_client.get("/api/clips")
    assert len(list_resp.json()["clips"]) == 0


def test_api_delete_clip_not_found_404(test_client):
    """DELETE /api/clips/nonexistent returns 404."""
    resp = test_client.delete("/api/clips/missing_file.mp4")
    assert resp.status_code == 404

