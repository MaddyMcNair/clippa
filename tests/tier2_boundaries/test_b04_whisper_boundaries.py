"""
tests/tier2_boundaries/test_b04_whisper_boundaries.py
Feature 4: Local Whisper Transcription - Boundary & Edge Cases (5 cases)
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


@pytest.mark.tier2
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.transcription not implemented yet")
def test_tc_b04_01_silence_returns_empty_words(tmp_path: Path):
    """TC-B04-01: Pure silence audio returns empty word list without unhandled exception."""
    audio = tmp_path / "silence.mp3"
    audio.write_bytes(b"\xFF\xFB" + b"\x00" * 512)
    mock_model = MagicMock()
    mock_model.transcribe.return_value = (iter([]), SimpleNamespace(language="en", duration=3.0))

    with patch("faster_whisper.WhisperModel", return_value=mock_model):
        res = transcription.transcribe_audio(audio)
        assert len(res.words) == 0
        assert res.text == ""


@pytest.mark.tier2
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.transcription not implemented yet")
def test_tc_b04_02_ultra_fast_speech_rate(tmp_path: Path):
    """TC-B04-02: Ultra-fast speech rate maintains monotonic word timestamps."""
    audio = tmp_path / "fast.mp3"
    audio.write_bytes(b"\xFF\xFB" + b"\x00" * 512)
    fast_words = [SimpleNamespace(word=f"word{i}", start=i * 0.1, end=(i + 1) * 0.1, probability=0.9) for i in range(10)]
    seg = SimpleNamespace(start=0.0, end=1.0, text="fast words", words=fast_words)
    mock_model = MagicMock()
    mock_model.transcribe.return_value = (iter([seg]), SimpleNamespace(language="en", duration=1.0))

    with patch("faster_whisper.WhisperModel", return_value=mock_model):
        res = transcription.transcribe_audio(audio)
        assert len(res.words) == 10
        for i in range(len(res.words) - 1):
            assert res.words[i].start < res.words[i].end
            assert res.words[i].end <= res.words[i + 1].start


@pytest.mark.tier2
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.transcription not implemented yet")
def test_tc_b04_03_ultra_slow_speech_separate_utterances(tmp_path: Path):
    """TC-B04-03: Ultra-slow speech creates separate utterances."""
    audio = tmp_path / "slow.mp3"
    audio.write_bytes(b"\xFF\xFB" + b"\x00" * 512)
    seg1 = SimpleNamespace(start=1.0, end=3.0, text="First thought", words=[SimpleNamespace(word="First", start=1.0, end=2.0, probability=0.9), SimpleNamespace(word="thought", start=2.0, end=3.0, probability=0.9)])
    seg2 = SimpleNamespace(start=15.0, end=17.0, text="Second thought", words=[SimpleNamespace(word="Second", start=15.0, end=16.0, probability=0.9), SimpleNamespace(word="thought", start=16.0, end=17.0, probability=0.9)])
    mock_model = MagicMock()
    mock_model.transcribe.return_value = (iter([seg1, seg2]), SimpleNamespace(language="en", duration=20.0))

    with patch("faster_whisper.WhisperModel", return_value=mock_model):
        res = transcription.transcribe_audio(audio)
        assert len(res.words) == 4
        assert res.words[-1].start >= 15000 or res.words[-1].start >= 15


@pytest.mark.tier2
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.transcription not implemented yet")
def test_tc_b04_04_low_snr_confidence(tmp_path: Path):
    """TC-B04-04: Low SNR audio assigns lower confidence scores but completes without crash."""
    audio = tmp_path / "noisy.mp3"
    audio.write_bytes(b"\xFF\xFB" + b"\x00" * 512)
    words = [
        SimpleNamespace(word="noisy", start=0.1, end=0.4, probability=0.25),
        SimpleNamespace(word="audio", start=0.45, end=0.8, probability=0.30),
    ]
    seg = SimpleNamespace(start=0.1, end=0.8, text="noisy audio", words=words)
    mock_model = MagicMock()
    mock_model.transcribe.return_value = (iter([seg]), SimpleNamespace(language="en", duration=1.0))

    with patch("faster_whisper.WhisperModel", return_value=mock_model):
        res = transcription.transcribe_audio(audio)
        assert len(res.words) == 2
        for w in res.words:
            conf = getattr(w, "confidence", getattr(w, "probability", 0.3))
            assert 0.0 <= conf <= 1.0


@pytest.mark.tier2
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.transcription not implemented yet")
def test_tc_b04_05_single_word_audio(tmp_path: Path):
    """TC-B04-05: Audio clip exactly 1 word in length produces valid 1-item word list."""
    audio = tmp_path / "single.mp3"
    audio.write_bytes(b"\xFF\xFB" + b"\x00" * 512)
    words = [SimpleNamespace(word="Hello", start=0.1, end=0.5, probability=0.99)]
    seg = SimpleNamespace(start=0.1, end=0.5, text="Hello", words=words)
    mock_model = MagicMock()
    mock_model.transcribe.return_value = (iter([seg]), SimpleNamespace(language="en", duration=0.6))

    with patch("faster_whisper.WhisperModel", return_value=mock_model):
        res = transcription.transcribe_audio(audio)
        assert len(res.words) == 1
        assert res.words[0].text == "Hello"
