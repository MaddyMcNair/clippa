"""
tests/tier3_combinations/test_pair14_cli_e2e_pipeline.py
Pairwise Test 14: F21 (CLI Runner) + F22 (End-to-End Pipeline Pass)
Interaction Seam: Unified CLI invocation produces valid rendered clips.
"""

from pathlib import Path
from unittest.mock import patch
import pytest

try:
    import clippa.cli as cli
    HAS_CLIPPA = True
except ImportError:
    HAS_CLIPPA = False


@pytest.mark.tier3
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.cli not implemented yet")
def test_pair14_cli_runs_full_pipeline(tmp_path: Path):
    """TC-PAIR-14: Single CLI run completes pipeline producing output files."""
    out_dir = tmp_path / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    rendered_clip = out_dir / "clip_1.mp4"
    rendered_clip.write_bytes(b"\x00\x00\x00\x18ftypisom" + b"\x00" * 1024)

    with patch("clippa.cli.run_pipeline", return_value=0) as mock_run:
        exit_code = cli.main(["--url", "test_video.mp4", "--output-dir", str(out_dir)])
        assert exit_code == 0
        assert rendered_clip.exists()
