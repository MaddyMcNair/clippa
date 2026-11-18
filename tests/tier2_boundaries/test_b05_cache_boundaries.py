"""
tests/tier2_boundaries/test_b05_cache_boundaries.py
Feature 5: Transcript Cache Sidecar - Boundary & Edge Cases (5 cases)
"""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import pytest

try:
    import clippa.transcription as transcription
    HAS_CLIPPA = True
except ImportError:
    HAS_CLIPPA = False


def _create_mock_whisper():
    mock_model = MagicMock()
    words = [SimpleNamespace(word="Regenerated", start=0.0, end=1.0, probability=0.95)]
    seg = SimpleNamespace(start=0.0, end=1.0, text="Regenerated", words=words)
    mock_model.transcribe.return_value = (iter([seg]), SimpleNamespace(language="en", duration=1.0))
    return mock_model


@pytest.mark.tier2
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.transcription not implemented yet")
def test_tc_b05_01_empty_cache_file(tmp_path: Path):
    """TC-B05-01: Empty (0-byte) cache file is detected as corrupt and triggers regeneration."""
    video = tmp_path / "video.mp4"
    video.write_bytes(b"\x00" * 1024)
    cache = tmp_path / "video.mp4.transcript_cache.json"
    cache.write_bytes(b"")

    mock_model = _create_mock_whisper()
    with patch("faster_whisper.WhisperModel", return_value=mock_model):
        res = transcription.transcribe_audio(video)
        assert res is not None
        assert mock_model.transcribe.call_count >= 1


@pytest.mark.tier2
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.transcription not implemented yet")
def test_tc_b05_02_truncated_json_cache(tmp_path: Path):
    """TC-B05-02: Malformed JSON cache file triggers regeneration without crashing."""
    video = tmp_path / "video.mp4"
    video.write_bytes(b"\x00" * 1024)
    cache = tmp_path / "video.mp4.transcript_cache.json"
    cache.write_text('{"version": 3, "words": [', encoding="utf-8")

    mock_model = _create_mock_whisper()
    with patch("faster_whisper.WhisperModel", return_value=mock_model):
        res = transcription.transcribe_audio(video)
        assert res is not None
        assert mock_model.transcribe.call_count >= 1


@pytest.mark.tier2
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.transcription not implemented yet")
def test_tc_b05_03_mismatched_schema_version(tmp_path: Path):
    """TC-B05-03: Cache file with outdated schema version triggers version upgrade or retranscription."""
    video = tmp_path / "video.mp4"
    video.write_bytes(b"\x00" * 1024)
    cache = tmp_path / "video.mp4.transcript_cache.json"
    cache.write_text(json.dumps({"version": 1, "text": "Old schema text"}), encoding="utf-8")

    mock_model = _create_mock_whisper()
    with patch("faster_whisper.WhisperModel", return_value=mock_model):
        res = transcription.transcribe_audio(video)
        assert res is not None


@pytest.mark.tier2
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.transcription not implemented yet")
def test_tc_b05_04_read_only_cache_directory(tmp_path: Path):
    """TC-B05-04: Read-only directory where cache cannot be written continues in-memory."""
    video = tmp_path / "video.mp4"
    video.write_bytes(b"\x00" * 1024)
    mock_model = _create_mock_whisper()

    with patch("faster_whisper.WhisperModel", return_value=mock_model):
        with patch("pathlib.Path.write_text", side_effect=PermissionError("Permission denied")):
            res = transcription.transcribe_audio(video)
            assert res is not None


@pytest.mark.tier2
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.transcription not implemented yet")
def test_tc_b05_05_cache_with_zero_words(tmp_path: Path):
    """TC-B05-05: Cache containing zero words loads cleanly without IndexError."""
    video = tmp_path / "empty_words.mp4"
    video.write_bytes(b"\x00" * 1024)
    cache = tmp_path / "empty_words.mp4.transcript_cache.json"
    cache.write_text(json.dumps({"version": 3, "words": [], "utterances": [], "text": "", "language": "en"}), encoding="utf-8")

    res = transcription.transcribe_audio(video)
    assert len(res.words) == 0
    assert res.text == ""
