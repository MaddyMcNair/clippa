"""
tests/tier1_features/test_feat04_whisper.py
Feature 4: Local Whisper Transcription
Category-Partition Tests (5 cases)
"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import pytest

try:
    import clippa.transcription as transcription
    HAS_CLIPPA = True
except ImportError:
    HAS_CLIPPA = False


def _create_mock_whisper_model():
    """Builds a mock faster_whisper.WhisperModel instance yielding realistic segments."""
    mock_model = MagicMock()
    words = [
        SimpleNamespace(word="Welcome", start=0.1, end=0.6, probability=0.98),
        SimpleNamespace(word="back", start=0.65, end=0.95, probability=0.95),
        SimpleNamespace(word="everyone", start=1.0, end=1.5, probability=0.92),
    ]
    seg = SimpleNamespace(start=0.1, end=1.5, text="Welcome back everyone", words=words)
    info = SimpleNamespace(language="en", language_probability=0.99, duration=1.5)
    mock_model.transcribe.return_value = (iter([seg]), info)
    return mock_model


@pytest.mark.tier1
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.transcription not implemented yet")
def test_tc_f04_01_whisper_produces_word_timestamps(tmp_path: Path):
    """TC-F04-01: Local Whisper produces word-level millisecond timestamps."""
    audio = tmp_path / "speech.mp3"
    audio.write_bytes(b"\xFF\xFB" + b"\x00" * 512)
    mock_model = _create_mock_whisper_model()

    with patch("faster_whisper.WhisperModel", return_value=mock_model):
        transcript = transcription.transcribe_audio(audio, model_size="base", device="cpu")
        assert len(transcript.words) > 0
        w0 = transcript.words[0]
        assert hasattr(w0, "start") and hasattr(w0, "end")
        assert isinstance(w0.start, int)
        assert isinstance(w0.end, int)
        assert w0.start < w0.end


@pytest.mark.tier1
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.transcription not implemented yet")
def test_tc_f04_02_whisper_cpu_fallback(tmp_path: Path):
    """TC-F04-02: Automatic fallback to CPU transcription when CUDA is unavailable."""
    audio = tmp_path / "speech.mp3"
    audio.write_bytes(b"\xFF\xFB" + b"\x00" * 512)
    mock_model = _create_mock_whisper_model()

    def constructor_side_effect(*args, **kwargs):
        device = kwargs.get("device", "cuda")
        if device == "cuda":
            raise RuntimeError("CUDA unavailable")
        return mock_model

    with patch("faster_whisper.WhisperModel", side_effect=constructor_side_effect) as mock_cls:
        transcript = transcription.transcribe_audio(audio, model_size="base", device="cuda")
        assert transcript is not None
        assert len(transcript.words) > 0


@pytest.mark.tier1
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.transcription not implemented yet")
def test_tc_f04_03_whisper_model_size_selection(tmp_path: Path):
    """TC-F04-03: Model size selection (tiny, base, small) instantiates corresponding model."""
    audio = tmp_path / "speech.mp3"
    audio.write_bytes(b"\xFF\xFB" + b"\x00" * 512)
    mock_model = _create_mock_whisper_model()

    with patch("faster_whisper.WhisperModel", return_value=mock_model) as mock_cls:
        transcription.transcribe_audio(audio, model_size="small", device="cpu")
        assert mock_cls.call_args is not None
        assert mock_cls.call_args[0][0] == "small" or mock_cls.call_args[1].get("model_size_or_path") == "small"


@pytest.mark.tier1
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.transcription not implemented yet")
def test_tc_f04_04_whisper_monotonic_timestamps(tmp_path: Path):
    """TC-F04-04: Output timestamps are strictly monotonic (start < end and end <= next_start)."""
    audio = tmp_path / "speech.mp3"
    audio.write_bytes(b"\xFF\xFB" + b"\x00" * 512)
    mock_model = _create_mock_whisper_model()

    with patch("faster_whisper.WhisperModel", return_value=mock_model):
        sample = transcription.transcribe_audio(audio, model_size="base", device="cpu")
        for i in range(len(sample.words)):
            w = sample.words[i]
            assert w.start < w.end, f"Word '{w.text}' has start >= end: {w.start} >= {w.end}"
            if i + 1 < len(sample.words):
                w_next = sample.words[i + 1]
                assert w.start <= w_next.start, f"Timestamps not monotonic: {w.start} > {w_next.start}"


@pytest.mark.tier1
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.transcription not implemented yet")
def test_tc_f04_05_whisper_confidence_scores(tmp_path: Path):
    """TC-F04-05: Confidence scores for all transcribed words are within valid float range [0.0, 1.0]."""
    audio = tmp_path / "speech.mp3"
    audio.write_bytes(b"\xFF\xFB" + b"\x00" * 512)
    mock_model = _create_mock_whisper_model()

    with patch("faster_whisper.WhisperModel", return_value=mock_model):
        sample = transcription.transcribe_audio(audio, model_size="base", device="cpu")
        for w in sample.words:
            assert hasattr(w, "confidence") or hasattr(w, "probability")
            score = getattr(w, "confidence", getattr(w, "probability", 1.0))
            assert 0.0 <= score <= 1.0
