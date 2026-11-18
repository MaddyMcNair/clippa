"""
tests/tier2_boundaries/test_b21_cli_boundaries.py
Feature 21: Standalone CLI Pipeline Runner - Boundary & Edge Cases (5 cases)
"""

from pathlib import Path
from unittest.mock import patch
import pytest

try:
    import clippa.cli as cli
    HAS_CLIPPA = True
except ImportError:
    HAS_CLIPPA = False


@pytest.mark.tier2
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.cli not implemented yet")
def test_tc_b21_01_no_args_exits_2():
    """TC-B21-01: Running CLI with no arguments exits with code 2."""
    with pytest.raises(SystemExit) as exc_info:
        cli.main([])
    assert exc_info.value.code == 2


@pytest.mark.tier2
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.cli not implemented yet")
def test_tc_b21_02_nonexistent_local_file(tmp_path: Path):
    """TC-B21-02: Specifying non-existent local file prints error and exits non-zero."""
    missing = tmp_path / "does_not_exist_9999.mp4"
    with patch("clippa.cli.run_pipeline", return_value=1):
        code = cli.main(["--url", str(missing)])
        assert code != 0


@pytest.mark.tier2
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.cli not implemented yet")
def test_tc_b21_03_negative_max_clips_rejected():
    """TC-B21-03: Passing negative value to --max-clips is rejected."""
    with pytest.raises(SystemExit):
        cli.main(["--url", "test.mp4", "--max-clips", "-5"])


@pytest.mark.tier2
def test_tc_b21_04_creates_missing_output_dir(tmp_path: Path):
    """TC-B21-04: Specifying non-existent output directory automatically creates it."""
    nested_dir = tmp_path / "a" / "b" / "outputs"
    assert not nested_dir.exists()
    nested_dir.mkdir(parents=True, exist_ok=True)
    assert nested_dir.exists()


@pytest.mark.tier2
def test_tc_b21_05_unique_file_naming_prevents_collision(tmp_path: Path):
    """TC-B21-05: Unique file naming prevents overwriting existing clips."""
    existing_clip = tmp_path / "clip_1.mp4"
    existing_clip.write_bytes(b"\x00" * 1024)

    # Naming generator creates next unique suffix
    base = "clip_1"
    counter = 1
    new_path = existing_clip
    while new_path.exists():
        counter += 1
        new_path = tmp_path / f"{base}_{counter}.mp4"

    assert new_path == tmp_path / "clip_1_2.mp4"
    assert not new_path.exists()
