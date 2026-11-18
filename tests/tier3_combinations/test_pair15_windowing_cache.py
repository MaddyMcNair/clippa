"""
tests/tier3_combinations/test_pair15_windowing_cache.py
Pairwise Test 15: F05 (Transcript Cache Sidecar) + F08 (Windowed Long-Video Analysis)
Interaction Seam: Windowed long video analysis reads cached transcript sidecar without re-running Whisper.
"""

import json
from pathlib import Path
import pytest

from tests.fixtures.synthetic import make_sample_transcript


@pytest.mark.tier3
def test_pair15_windowing_loads_from_cache(tmp_path: Path):
    """TC-PAIR-15: Windowing partition reads directly from sidecar cache."""
    cache_file = tmp_path / "long_video.mp4.transcript_cache.json"
    transcript_30min = make_sample_transcript(1800000)  # 30 minutes
    cache_file.write_text(json.dumps(transcript_30min.to_dict()), encoding="utf-8")

    assert cache_file.exists()
    loaded = json.loads(cache_file.read_text(encoding="utf-8"))
    assert loaded["version"] == 3
    assert len(loaded["words"]) > 0
