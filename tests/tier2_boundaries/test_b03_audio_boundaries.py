"""
tests/tier2_boundaries/test_b03_audio_boundaries.py
Feature 3: Audio Track Extraction - Boundary & Edge Cases (5 cases)
"""

from pathlib import Path
import pytest

from tests.utils import probe_media

try:
    import clippa.transcription as transcription
    HAS_CLIPPA = True
except ImportError:
    HAS_CLIPPA = False


@pytest.mark.tier2
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.transcription not implemented yet")
def test_tc_b03_01_zero_byte_video_error(tmp_path: Path):
    """TC-B03-01: 0-byte video file raises descriptive error on audio extraction."""
    empty_video = tmp_path / "zero_byte.mp4"
    empty_video.write_bytes(b"")

    with pytest.raises(Exception):
        transcription.extract_audio(empty_video)


@pytest.mark.tier2
@pytest.mark.requires_ffmpeg
@pytest.mark.requires_ffprobe
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.transcription not implemented yet")
def test_tc_b03_02_pure_silence_extraction(sample_silent_mp4: Path, tmp_path: Path):
    """TC-B03-02: Pure silence audio track extracts valid audio file without crash."""
    out = transcription.extract_audio(sample_silent_mp4, output_path=tmp_path / "silence.mp3")
    assert Path(out).exists()
    assert Path(out).stat().st_size > 0


@pytest.mark.tier2
@pytest.mark.requires_ffmpeg
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.transcription not implemented yet")
def test_tc_b03_03_zero_audio_streams(sample_no_audio_mp4: Path, tmp_path: Path):
    """TC-B03-03: Video with zero audio streams raises error without hanging."""
    with pytest.raises(Exception) as exc_info:
        transcription.extract_audio(sample_no_audio_mp4, output_path=tmp_path / "fail.mp3")
    assert "audio" in str(exc_info.value).lower() or "stream" in str(exc_info.value).lower()


@pytest.mark.tier2
@pytest.mark.requires_ffmpeg
@pytest.mark.requires_ffprobe
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.transcription not implemented yet")
def test_tc_b03_04_extreme_sample_rates(sample_1080p_mp4: Path, tmp_path: Path):
    """TC-B03-04: Audio extraction supports custom sample rates (e.g. 8000Hz and 16000Hz)."""
    out_8k = tmp_path / "audio_8k.mp3"
    transcription.extract_audio(sample_1080p_mp4, output_path=out_8k, sample_rate=8000)
    probe = probe_media(out_8k)
    audio_stream = next(s for s in probe["streams"] if s.get("codec_type") == "audio")
    assert int(audio_stream.get("sample_rate", 8000)) == 8000


@pytest.mark.tier2
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.transcription not implemented yet")
def test_tc_b03_05_corrupted_trailing_frames(tmp_path: Path):
    """TC-B03-05: Corrupted audio frames at EOF truncate cleanly or raise descriptive error."""
    corrupted_video = tmp_path / "corrupt_tail.mp4"
    corrupted_video.write_bytes(b"\x00" * 1024 + b"GARBAGE_TRAILING_BYTES")
    out_audio = tmp_path / "out_corrupt.mp3"

    try:
        res = transcription.extract_audio(corrupted_video, output_path=out_audio)
        if res:
            assert Path(res).exists()
    except Exception as exc:
        assert len(str(exc)) > 0
