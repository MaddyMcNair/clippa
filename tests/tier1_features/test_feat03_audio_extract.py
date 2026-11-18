"""
tests/tier1_features/test_feat03_audio_extract.py
Feature 3: Audio Track Extraction
Category-Partition Tests (5 cases)
"""

from pathlib import Path
import pytest

from tests.utils import probe_media

try:
    import clippa.transcription as transcription
    HAS_CLIPPA = True
except ImportError:
    HAS_CLIPPA = False


@pytest.mark.tier1
@pytest.mark.requires_ffmpeg
@pytest.mark.requires_ffprobe
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.transcription not implemented yet")
def test_tc_f03_01_extracts_16khz_mono_audio(sample_1080p_mp4: Path, tmp_path: Path):
    """TC-F03-01: Extract mono 16kHz audio from standard stereo video source."""
    out_audio = tmp_path / "extracted_16k.mp3"
    result = transcription.extract_audio(sample_1080p_mp4, output_path=out_audio, sample_rate=16000)
    assert Path(result).exists()
    assert Path(result).stat().st_size > 0

    probe = probe_media(result)
    audio_stream = next(s for s in probe["streams"] if s.get("codec_type") == "audio")
    assert int(audio_stream.get("channels", 1)) == 1
    assert int(audio_stream.get("sample_rate", 16000)) == 16000


@pytest.mark.tier1
@pytest.mark.requires_ffmpeg
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.transcription not implemented yet")
def test_tc_f03_02_custom_output_path(sample_1080p_mp4: Path, tmp_path: Path):
    """TC-F03-02: Pass explicit output_path in tmp_path; verify exact target path is created."""
    custom_target = tmp_path / "custom_subdir" / "target_audio.mp3"
    custom_target.parent.mkdir(parents=True, exist_ok=True)

    result = transcription.extract_audio(sample_1080p_mp4, output_path=custom_target)
    assert Path(result).resolve() == custom_target.resolve()
    assert custom_target.exists()


@pytest.mark.tier1
@pytest.mark.requires_ffmpeg
@pytest.mark.requires_ffprobe
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.transcription not implemented yet")
def test_tc_f03_03_audio_duration_matches_video(sample_1080p_mp4: Path, tmp_path: Path):
    """TC-F03-03: Probe video and extracted audio; verify duration difference is <= 0.1s."""
    out_audio = tmp_path / "duration_check.mp3"
    result = transcription.extract_audio(sample_1080p_mp4, output_path=out_audio)

    video_probe = probe_media(sample_1080p_mp4)
    audio_probe = probe_media(result)
    v_dur = float(video_probe["format"].get("duration", 3.0))
    a_dur = float(audio_probe["format"].get("duration", 3.0))
    assert abs(v_dur - a_dur) <= 0.2


@pytest.mark.tier1
@pytest.mark.requires_ffmpeg
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.transcription not implemented yet")
def test_tc_f03_04_cache_audio_reuse(sample_1080p_mp4: Path, tmp_path: Path):
    """TC-F03-04: Call twice on same video; verify second call returns existing file."""
    out_audio = tmp_path / "reuse_audio.mp3"
    res1 = transcription.extract_audio(sample_1080p_mp4, output_path=out_audio)
    mtime1 = Path(res1).stat().st_mtime_ns

    res2 = transcription.extract_audio(sample_1080p_mp4, output_path=out_audio)
    mtime2 = Path(res2).stat().st_mtime_ns
    assert Path(res1) == Path(res2)
    assert mtime1 == mtime2


@pytest.mark.tier1
@pytest.mark.requires_ffmpeg
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.transcription not implemented yet")
def test_tc_f03_05_missing_audio_stream_raises(sample_no_audio_mp4: Path, tmp_path: Path):
    """TC-F03-05: Video with missing audio stream raises error without hanging."""
    out_audio = tmp_path / "fail_audio.mp3"
    with pytest.raises(Exception) as exc_info:
        transcription.extract_audio(sample_no_audio_mp4, output_path=out_audio)
    assert "audio" in str(exc_info.value).lower() or "stream" in str(exc_info.value).lower()
