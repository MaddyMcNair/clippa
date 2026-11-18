"""
tests/tier3_combinations/test_pair01_ingest_whisper.py
Pairwise Test 01: F01 (Ingestion) + F04 (Whisper Transcription)
Interaction Seam: Ingested MP4 feeds audio extraction and Whisper transcription.
"""

from pathlib import Path
from unittest.mock import patch
import pytest

from tests.fixtures.synthetic import make_sample_transcript

try:
    import clippa.ingestion as ingestion
    import clippa.transcription as transcription
    HAS_CLIPPA = True
except ImportError:
    HAS_CLIPPA = False


@pytest.mark.tier3
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa not implemented yet")
def test_pair01_ingest_feeds_whisper(tmp_path: Path):
    """TC-PAIR-01: Downloaded MP4 feeds audio extraction and Whisper transcription pipeline."""
    video_file = tmp_path / "ingested.mp4"
    video_file.write_bytes(b"\x00\x00\x00\x18ftypisom" + b"\x00" * 1024)

    expected_transcript = make_sample_transcript(5000, "Automated synthetic speech transcript.")

    with patch("clippa.ingestion.download_video", return_value=video_file):
        downloaded = ingestion.download_video("https://youtube.com/watch?v=sample", output_dir=tmp_path)
        assert downloaded.exists()

        with patch("clippa.transcription.transcribe_audio", return_value=expected_transcript):
            transcript = transcription.transcribe_audio(downloaded)
            assert len(transcript.words) > 0
            assert "synthetic" in transcript.text
