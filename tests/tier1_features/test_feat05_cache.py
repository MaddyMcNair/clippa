"""
tests/tier1_features/test_feat05_cache.py
Feature 5: Transcript Cache Sidecar
Category-Partition Tests (5 cases)
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


def _create_mock_whisper_model(text: str = "Cached sample text"):
    mock_model = MagicMock()
    words = [
        SimpleNamespace(word="Cached", start=0.0, end=0.5, probability=0.95),
        SimpleNamespace(word="sample", start=0.6, end=1.0, probability=0.96),
        SimpleNamespace(word="text", start=1.1, end=1.5, probability=0.97),
    ]
    seg = SimpleNamespace(start=0.0, end=1.5, text=text, words=words)
    info = SimpleNamespace(language="en", language_probability=0.99, duration=1.5)
    mock_model.transcribe.return_value = (iter([seg]), info)
    return mock_model


@pytest.mark.tier1
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.transcription not implemented yet")
def test_tc_f05_01_cache_sidecar_generated(tmp_path: Path):
    """TC-F05-01: Transcription generates .transcript_cache.json matching version 3 schema."""
    video = tmp_path / "sample.mp4"
    video.write_bytes(b"\x00" * 1024)
    mock_model = _create_mock_whisper_model()

    with patch("faster_whisper.WhisperModel", return_value=mock_model):
        transcription.transcribe_audio(video)

    cache_file = tmp_path / "sample.mp4.transcript_cache.json"
    assert cache_file.exists()
    data = json.loads(cache_file.read_text(encoding="utf-8"))
    assert data.get("version") == 3
    assert "words" in data
    assert "text" in data


@pytest.mark.tier1
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.transcription not implemented yet")
def test_tc_f05_02_subsequent_request_hits_cache(tmp_path: Path):
    """TC-F05-02: Subsequent request loads from sidecar cache without re-running Whisper."""
    video = tmp_path / "cached_sample.mp4"
    video.write_bytes(b"\x00" * 1024)
    mock_model = _create_mock_whisper_model()

    with patch("faster_whisper.WhisperModel", return_value=mock_model) as mock_cls:
        res1 = transcription.transcribe_audio(video)
        call_count_after_first = mock_model.transcribe.call_count

        res2 = transcription.transcribe_audio(video)
        # Second call must hit cache; Whisper transcribe not called again
        assert mock_model.transcribe.call_count == call_count_after_first
        assert res1.text == res2.text


@pytest.mark.tier1
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.transcription not implemented yet")
def test_tc_f05_03_cache_validates_required_fields(tmp_path: Path):
    """TC-F05-03: Cache file validates fields: words, utterances, text, language, version."""
    video = tmp_path / "schema_sample.mp4"
    video.write_bytes(b"\x00" * 1024)
    mock_model = _create_mock_whisper_model()

    with patch("faster_whisper.WhisperModel", return_value=mock_model):
        transcription.transcribe_audio(video)

    cache_file = tmp_path / "schema_sample.mp4.transcript_cache.json"
    data = json.loads(cache_file.read_text(encoding="utf-8"))
    assert "words" in data
    assert "utterances" in data
    assert "text" in data
    assert "language" in data or "language_code" in data
    assert data.get("version") == 3


@pytest.mark.tier1
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.transcription not implemented yet")
def test_tc_f05_04_corrupted_cache_regenerates(tmp_path: Path):
    """TC-F05-04: Corrupted cache file is detected and safely regenerated."""
    video = tmp_path / "corrupt_cache.mp4"
    video.write_bytes(b"\x00" * 1024)
    cache_file = tmp_path / "corrupt_cache.mp4.transcript_cache.json"
    cache_file.write_text("{invalid json", encoding="utf-8")

    mock_model = _create_mock_whisper_model()
    with patch("faster_whisper.WhisperModel", return_value=mock_model):
        res = transcription.transcribe_audio(video)
        assert res is not None
        assert mock_model.transcribe.call_count >= 1


@pytest.mark.tier1
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.transcription not implemented yet")
def test_tc_f05_05_force_retranscribe_bypasses_cache(tmp_path: Path):
    """TC-F05-05: force_retranscribe=True / --no-cache bypasses existing cache file."""
    video = tmp_path / "retranscribe.mp4"
    video.write_bytes(b"\x00" * 1024)

    mock_model = _create_mock_whisper_model("Fresh text")
    with patch("faster_whisper.WhisperModel", return_value=mock_model):
        transcription.transcribe_audio(video)
        first_calls = mock_model.transcribe.call_count

        res = transcription.transcribe_audio(video, force_retranscribe=True)
        assert mock_model.transcribe.call_count > first_calls
        assert res.text == "Fresh text"
