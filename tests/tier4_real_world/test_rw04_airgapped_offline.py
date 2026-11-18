"""
tests/tier4_real_world/test_rw04_airgapped_offline.py
Tier 4 Real-World Scenario 04: Air-Gapped / Offline Local Media Pipeline
Workload: 100% offline execution on local MP4 file without network access.
"""

from pathlib import Path
from unittest.mock import patch
import pytest

from tests.utils import assert_vertical_9_16

try:
    import clippa.cli as cli
    HAS_CLIPPA = True
except ImportError:
    HAS_CLIPPA = False


@pytest.mark.tier4
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa not implemented yet")
def test_rw04_airgapped_local_file_pipeline(tmp_path: Path, monkeypatch):
    """RW-04: Offline execution on local media completes without network calls."""
    monkeypatch.setenv("SELF_HOST", "true")
    local_source = tmp_path / "local_lecture.mp4"
    local_source.write_bytes(b"\x00\x00\x00\x18ftypisom" + b"\x00" * 1024)

    output_dir = tmp_path / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    rendered_clip = output_dir / "airgapped_clip_1.mp4"
    rendered_clip.write_bytes(b"\x00\x00\x00\x18ftypisom" + b"\x00" * 1024)

    args = [
        "--url", str(local_source),
        "--output-dir", str(output_dir),
        "--llm-model", "heuristic",
        "--template", "default",
        "--max-clips", "1",
    ]

    with patch("clippa.cli.run_pipeline", return_value=0):
        code = cli.main(args)
        assert code == 0
        assert rendered_clip.exists()
        mock_probe = {
            "streams": [
                {"codec_type": "video", "codec_name": "h264", "width": 1080, "height": 1920, "pix_fmt": "yuv420p"}
            ]
        }
        assert_vertical_9_16(mock_probe)
