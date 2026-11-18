"""
tests/test_adversarial_m4_coverage.py
Milestone 4 - Tier 5 Adversarial Coverage Hardening Test Suite for CLIPPA.

Exercises edge cases, boundary failures, unexpected data types, zero-length sequences,
unusual YouTube URLs, corrupt caches, extreme aspect ratios, ASS control character escaping,
FFmpeg filter escaping, and error handling across all clippa modules:
- clippa.config (Zero-paid-API mode, environment poisoning, quotas)
- clippa.ingestion (URL parsing, ID extraction, duration boundaries, preflight metadata cache)
- clippa.transcription (Confidence clamping, monotonicity enforcement, cache schema v3, silence)
- clippa.analysis (ASS title sanitization, windowing, deduplication, teasers, heuristic fallback)
- clippa.reframing (round_to_even, extreme aspect ratios, face centroid tracking, Ken Burns zoom)
- clippa.subtitles (ASS formatting, karaoke tags, emoji injection boundaries, template fallbacks)
- clippa.rendering (FFmpeg filtergraph escaping, atomic temp cleanup, error handling)
- clippa.cli (Argument validation, non-colliding paths, word timestamp relative shifting)

Author: Challenger 1 (Milestone 4)
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import clippa.analysis as analysis
import clippa.cli as cli
import clippa.config as config
import clippa.ingestion as ingestion
import clippa.reframing as reframing
import clippa.rendering as rendering
import clippa.subtitles as subtitles
import clippa.transcription as transcription
from clippa.analysis import TeaserCandidate, ViralSegment
from clippa.ingestion import (
    InvalidURLError,
    PreflightTimeoutError,
    StreamDurationUnavailableError,
    VideoAgeRestrictedError,
    VideoDownloadError,
    VideoDurationExceededError,
    VideoMetadata,
    VideoUnavailableError,
)
from clippa.rendering import (
    FFmpegExecutionError,
    MediaNotFoundError,
    RenderingError,
    SubtitleBurnError,
)
from clippa.transcription import (
    AudioExtractionError,
    TranscriptData,
    TranscriptionError,
    Utterance,
    Word,
)


# ==============================================================================
# 1. Configuration & Zero-Paid-API Boundary Stress
# ==============================================================================

class TestConfigAdversarial:
    """Adversarial boundary tests for clippa.config."""

    @pytest.mark.parametrize("val,expected", [
        ("true", True),
        ("TRUE", True),
        ("1", True),
        ("yes", True),
        ("on", True),
        ("false", False),
        ("FALSE", False),
        ("0", False),
        ("no", False),
        ("off", False),
        ("", False),
        ("   ", False),
        ("invalid_str", False),
    ])
    def test_config_self_host_variations_falsy_and_truthy(self, monkeypatch, val, expected):
        """Verifies parsing of truthy and falsy representations of SELF_HOST."""
        monkeypatch.setenv("SELF_HOST", val)
        cfg = config.Config()
        assert cfg.self_hosted is expected
        assert cfg.self_host is expected

    def test_config_poisons_all_cloud_keys_even_when_injected(self, monkeypatch):
        """In self-hosted mode, cloud API keys must be completely suppressed/None."""
        monkeypatch.setenv("SELF_HOST", "true")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-adversarial-fake-key-12345")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "claude-fake-token-67890")
        monkeypatch.setenv("ASSEMBLY_AI_API_KEY", "assembly-fake-token")
        monkeypatch.setenv("GOOGLE_API_KEY", "google-fake-token")
        monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-fake-token")
        monkeypatch.setenv("APIFY_API_TOKEN", "apify-fake-token")

        cfg = config.Config()
        assert cfg.openai_api_key is None
        assert cfg.anthropic_api_key is None
        assert cfg.assembly_ai_api_key is None
        assert cfg.google_api_key is None
        assert cfg.openrouter_api_key is None
        assert cfg.apify_api_token is None

    def test_config_quotas_and_durations_unrestricted_in_self_hosted(self, monkeypatch):
        """Self-hosted mode unlocks duration and disables quota enforcement."""
        monkeypatch.setenv("SELF_HOST", "true")
        monkeypatch.setenv("QUOTAS_ENABLED", "true")

        cfg = config.Config()
        assert cfg.is_quota_enforced() is False
        assert cfg.max_duration_seconds == 86400
        assert cfg.get_allowed_resolutions() == ["1080p", "720p"]

    def test_config_singleton_override_and_reset(self):
        """Verifies set_config_override and restoration of singleton."""
        original = config.get_config()
        custom = config.Config()
        custom.default_resolution = "720p"

        try:
            config.set_config_override(custom)
            assert config.get_config() is custom
            assert config.get_config().default_resolution == "720p"
        finally:
            config.set_config_override(None)

        restored = config.get_config()
        assert restored is not custom


# ==============================================================================
# 2. Ingestion, URL Parsing & Metadata Preflight Adversarial
# ==============================================================================

class TestIngestionAdversarial:
    """Stress tests for clippa.ingestion module."""

    @pytest.mark.parametrize("url,expected_id", [
        ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/watch?v=dQw4w9WgXcQ&feature=share&t=10s", "dQw4w9WgXcQ"),
        ("https://youtu.be/dQw4w9WgXcQ?si=abcdef123", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/embed/dQw4w9WgXcQ?autoplay=1", "dQw4w9WgXcQ"),
        ("https://www.youtube-nocookie.com/embed/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/shorts/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/live/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/v/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("http://youtube.com:8080/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("  https://youtu.be/dQw4w9WgXcQ  \n", "dQw4w9WgXcQ"),
        ("https://m.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
    ])
    def test_extract_video_id_unusual_formats(self, url, expected_id):
        """Extracts 11-char video IDs across diverse real-world URL formats."""
        assert ingestion.extract_video_id(url) == expected_id
        assert ingestion.is_valid_youtube_url(url) is True
        assert ingestion.canonical_youtube_url(url) == f"https://www.youtube.com/watch?v={expected_id}"

    @pytest.mark.parametrize("url", [
        "",
        "   ",
        None,
        "https://vimeo.com/123456789",
        "https://tiktok.com/@user/video/123",
        "https://www.youtube.com/watch?v=",
        "https://youtu.be/",
        "https://www.youtube.com/embed/",
        "https://www.youtube.com/watch?other=123",
        "http://not_youtube.com/watch?v=dQw4w9WgXcQ",
        "<script>alert(1)</script>",
        "SELECT * FROM videos WHERE id = '1';",
        "https://www.youtube.com/watch?v=ab",  # Less than 3 chars
    ])
    def test_extract_video_id_adversarial_inputs(self, url):
        """Adversarial/malformed inputs safely evaluate to None."""
        assert ingestion.extract_video_id(url) is None
        assert ingestion.is_valid_youtube_url(url) is False
        assert ingestion.canonical_youtube_url(url) is None

    def test_validate_video_duration_boundary_values(self):
        """Tests duration validation with boundary, negative, infinite, and NaN values."""
        # 0 seconds
        assert ingestion.validate_video_duration(0.0) is False
        with pytest.raises(StreamDurationUnavailableError):
            ingestion.validate_video_duration(0.0, raise_on_error=True)

        # Negative seconds
        assert ingestion.validate_video_duration(-10.0) is False
        with pytest.raises(StreamDurationUnavailableError):
            ingestion.validate_video_duration(-10.0, raise_on_error=True)

        # Exact upper limit
        assert ingestion.validate_video_duration(5400.0, max_duration=5400.0) is True

        # Exceeding upper limit
        assert ingestion.validate_video_duration(5400.1, max_duration=5400.0) is False
        with pytest.raises(VideoDurationExceededError):
            ingestion.validate_video_duration(5400.1, max_duration=5400.0, raise_on_error=True)

        # Infinite duration
        with pytest.raises(VideoDurationExceededError):
            ingestion.validate_video_duration(float("inf"), raise_on_error=True)

        # Dictionary input
        assert ingestion.validate_video_duration({"duration": 180.0}) is True
        assert ingestion.validate_video_duration({"duration": 0.0}) is False

        # VideoMetadata input
        meta_valid = VideoMetadata(id="1", title="T", duration=120.0)
        assert ingestion.validate_video_duration(meta_valid) is True
        meta_invalid = VideoMetadata(id="2", title="T", duration=6000.0)
        with pytest.raises(VideoDurationExceededError):
            ingestion.validate_video_duration(meta_invalid, max_duration=5400.0, raise_on_error=True)

    def test_resolve_local_video_path_file_uri_and_raw_path(self, tmp_path):
        """Validates local file path resolution for Path, string, and file:// URIs."""
        sample_file = tmp_path / "sample_video.mp4"
        sample_file.write_bytes(b"dummy mp4 payload")

        # Direct Path
        assert ingestion.resolve_local_video_path(sample_file) == sample_file.resolve()
        assert ingestion.is_local_video(sample_file) is True

        # String path
        assert ingestion.resolve_local_video_path(str(sample_file)) == sample_file.resolve()
        assert ingestion.is_local_video(str(sample_file)) is True

        # file:// URI
        file_uri = sample_file.as_uri()
        assert ingestion.resolve_local_video_path(file_uri) == sample_file.resolve()
        assert ingestion.is_local_video(file_uri) is True

        # Non-existent file
        non_existent = tmp_path / "ghost.mp4"
        assert ingestion.resolve_local_video_path(non_existent) is None
        assert ingestion.is_local_video(non_existent) is False
        assert ingestion.resolve_local_video_path("") is None
        assert ingestion.resolve_local_video_path(None) is None

    def test_video_metadata_dict_interface_and_serialization(self):
        """Exercises VideoMetadata dict access, get(), to_dict(), and from_dict()."""
        meta = VideoMetadata(
            id="vid123",
            title="Adversarial Title",
            duration=45.5,
            uploader="Tester",
            is_live=False,
            view_count=1000,
        )
        assert meta["id"] == "vid123"
        assert meta["duration"] == 45.5
        assert meta.get("uploader") == "Tester"
        assert meta.get("nonexistent_prop", "default_val") == "default_val"
        assert "view_count" in meta

        with pytest.raises(KeyError):
            _ = meta["missing_key"]

        data = meta.to_dict()
        assert data["id"] == "vid123"
        assert data["duration"] == 45.5

        # Roundtrip from dict with missing optional keys
        restored = VideoMetadata.from_dict({
            "id": "vid123",
            "title": "Adversarial Title",
            "duration": "45.5",  # string representation should convert cleanly
        })
        assert restored.id == "vid123"
        assert restored.duration == 45.5
        assert restored.uploader == "Unknown"

    def test_download_video_invalid_inputs_fast_reject(self, tmp_path):
        """download_video must reject empty or invalid inputs immediately without calling yt-dlp."""
        with pytest.raises(InvalidURLError):
            ingestion.download_video(None, output_dir=tmp_path)

        with pytest.raises(InvalidURLError):
            ingestion.download_video("", output_dir=tmp_path)

        with pytest.raises(InvalidURLError):
            ingestion.download_video("   ", output_dir=tmp_path)

        with pytest.raises(InvalidURLError):
            ingestion.download_video("https://vimeo.com/987654321", output_dir=tmp_path)

        with pytest.raises(InvalidURLError):
            ingestion.download_video("C:/definitely/not/a/real/video.mp4", output_dir=tmp_path)

    def test_metadata_cache_lifecycle(self):
        """Verifies in-memory metadata cache hit and clear_metadata_cache."""
        ingestion.clear_metadata_cache()
        test_url = "https://www.youtube.com/watch?v=cached12345"

        meta = VideoMetadata(id="cached12345", title="Cached Video", duration=60.0)
        ingestion._METADATA_CACHE[test_url] = meta

        # Cache hit returns immediately
        fetched = ingestion.get_video_metadata(test_url, use_cache=True)
        assert fetched is meta

        # Clearing cache purges it
        ingestion.clear_metadata_cache()
        assert test_url not in ingestion._METADATA_CACHE


# ==============================================================================
# 3. Transcription, Monotonicity & Cache Schema v3 Adversarial
# ==============================================================================

class TestTranscriptionAdversarial:
    """Stress tests for clippa.transcription module."""

    @pytest.mark.parametrize("val,alt,default,expected", [
        (None, 0.85, 1.0, 0.85),
        ("0.75", None, 1.0, 0.75),
        ("invalid", None, 0.5, 0.5),
        (-0.5, None, 1.0, 0.0),      # Clamped to 0.0
        (1.5, None, 1.0, 1.0),       # Clamped to 1.0
        (0.0, None, 1.0, 0.0),       # 0.0 preserved, not treated as falsy
        (float("nan"), None, 1.0, 1.0),
    ])
    def test_safe_confidence_extreme_types_and_bounds(self, val, alt, default, expected):
        """_safe_confidence handles out-of-bounds, strings, NaNs, and falsy 0.0."""
        assert transcription._safe_confidence(val, alt_val=alt, default=default) == expected

    def test_transcript_data_roundtrip_v3_schema(self):
        """Verifies schema version 3 compliance and roundtrip serialization."""
        w1 = Word(text="Hello", start=100, end=400, confidence=0.98)
        w2 = Word(text="world", start=450, end=800, confidence=0.95)
        u1 = Utterance(text="Hello world", start=100, end=800, words=[w1, w2])
        td = TranscriptData(text="Hello world", language="en", words=[w1, w2], utterances=[u1])

        d = td.to_dict()
        assert d["version"] == 3
        assert d["language_code"] == "en"
        assert len(d["words"]) == 2
        assert len(d["utterances"]) == 1

        restored = TranscriptData.from_dict(d)
        assert restored.text == "Hello world"
        assert restored.language == "en"
        assert len(restored.words) == 2
        assert restored.words[0].start == 100
        assert restored.words[1].end == 800

    def test_transcript_cache_corruption_resilience(self, tmp_path):
        """Corrupt, truncated, or outdated cache sidecars must return None safely."""
        video_p = tmp_path / "clip.mp4"
        video_p.write_bytes(b"dummy video")

        # 1. Non-existent cache
        assert transcription.load_cached_transcript(video_p) is None

        # 2. 0-byte cache
        cache_p = transcription.get_cache_path(video_p)
        cache_p.write_bytes(b"")
        assert transcription.load_cached_transcript(video_p) is None

        # 3. Truncated / invalid JSON
        cache_p.write_text('{"version": 3, "text": "hello", ', encoding="utf-8")
        assert transcription.load_cached_transcript(video_p) is None

        # 4. Outdated schema version (e.g. version 2)
        cache_p.write_text('{"version": 2, "text": "hello", "words": []}', encoding="utf-8")
        assert transcription.load_cached_transcript(video_p) is None

    def test_transcribe_audio_strict_monotonicity_enforcement(self, tmp_path):
        """Enforces strictly monotonic start and end timestamps even on erratic model output."""
        dummy_audio = tmp_path / "speech.mp3"
        dummy_audio.write_bytes(b"dummy audio")

        # Mock segment with non-monotonic and zero-duration timestamps
        # Word 1: 1.0s to 1.5s (1000ms to 1500ms)
        # Word 2: 0.8s to 0.7s (Backwards start and negative duration!)
        # Word 3: 1.5s to 1.5s (Zero duration!)
        raw_words = [
            MagicMock(word="First", start=1.0, end=1.5, probability=0.9),
            MagicMock(word="Second", start=0.8, end=0.7, probability=0.8),
            MagicMock(word="Third", start=1.5, end=1.5, probability=0.85),
        ]
        mock_seg = MagicMock(words=raw_words, start=0.8, end=1.5, text="First Second Third")
        mock_info = MagicMock(language="en")

        mock_whisper = MagicMock()
        mock_whisper.transcribe.return_value = ([mock_seg], mock_info)

        with patch("clippa.transcription.faster_whisper.WhisperModel", return_value=mock_whisper):
            transcript = transcription.transcribe_audio(dummy_audio, force_retranscribe=True)

        assert len(transcript.words) == 3
        w0, w1, w2 = transcript.words

        # Assert w_start < w_end for each word
        assert w0.start < w0.end
        assert w1.start < w1.end
        assert w2.start < w2.end

        # Assert non-decreasing sequence across words
        assert w0.start <= w1.start
        assert w1.start <= w2.start

    def test_transcribe_audio_zero_speech_silence(self, tmp_path):
        """Pure silence returning 0 Whisper segments completes with empty word list."""
        dummy_audio = tmp_path / "silent.mp3"
        dummy_audio.write_bytes(b"dummy audio")

        mock_whisper = MagicMock()
        mock_whisper.transcribe.return_value = ([], MagicMock(language="en"))

        with patch("clippa.transcription.faster_whisper.WhisperModel", return_value=mock_whisper):
            transcript = transcription.transcribe_audio(dummy_audio, force_retranscribe=True)

        assert transcript.text == ""
        assert len(transcript.words) == 0
        assert len(transcript.utterances) == 0

    def test_extract_audio_zero_byte_source_fast_fail(self, tmp_path):
        """0-byte source video raises AudioExtractionError immediately."""
        zero_file = tmp_path / "zero.mp4"
        zero_file.write_bytes(b"")

        with pytest.raises(AudioExtractionError):
            transcription.extract_audio(zero_file)

        non_existent = tmp_path / "missing.mp4"
        with pytest.raises(AudioExtractionError):
            transcription.extract_audio(non_existent)

    def test_save_cached_transcript_read_only_directory(self, tmp_path):
        """PermissionError on saving cache logs warning and returns None gracefully."""
        dummy_video = tmp_path / "readonly_video.mp4"
        td = TranscriptData(text="Test", language="en", words=[], utterances=[])

        with patch("pathlib.Path.write_text", side_effect=PermissionError("Access Denied")):
            saved = transcription.save_cached_transcript(dummy_video, td)
            assert saved is None


# ==============================================================================
# 4. Virality Analysis, Heuristic Scorer & Sanitization Adversarial
# ==============================================================================

class TestAnalysisAdversarial:
    """Stress tests for clippa.analysis module."""

    def test_sanitize_hook_title_ass_injection_and_quotes(self):
        """ASS control characters and hashtags are sanitized from hook titles."""
        raw = "  \"{\\b1}Why You Should Never 'Eat' #viral #fyp{\\b0}\"  "
        clean = analysis.sanitize_hook_title(raw)
        assert "{" not in clean
        assert "}" not in clean
        assert "\\" not in clean
        assert "#" not in clean
        assert not clean.startswith('"')
        assert not clean.endswith('"')

    def test_sanitize_hook_title_length_and_word_constraints(self):
        """Hook titles strictly enforce 3 to 9 words and max 64 characters."""
        # 1-word title expanded
        one_word = analysis.sanitize_hook_title("Secrets")
        assert len(one_word.split()) >= 3

        # >9 words title truncated
        ten_words = "One two three four five six seven eight nine ten eleven twelve"
        sanitized_10 = analysis.sanitize_hook_title(ten_words)
        assert len(sanitized_10.split()) <= 9

        # >64 characters trimmed cleanly
        long_title = "The Most Incredible Unbelievable Secret That Nobody Dared To Tell You In 2026"
        sanitized_long = analysis.sanitize_hook_title(long_title)
        assert len(sanitized_long) <= 64

    def test_sanitize_hook_title_punctuation_and_emojis_only(self):
        """Punctuation-only or emoji-only titles trigger safe fallback title."""
        punct = analysis.sanitize_hook_title("🔥🔥🔥 !!! ??? ...")
        assert len(punct.split()) >= 3
        assert any(c.isalnum() for c in punct)

        empty = analysis.sanitize_hook_title("")
        assert len(empty.split()) >= 3

    def test_create_analysis_windows_exact_boundaries(self):
        """Videos <= 15 minutes return 1 window; > 15 minutes return overlapping windows."""
        # Exact boundary 900.0s (15m) -> 1 window
        w_900 = analysis.create_analysis_windows(900.0)
        assert len(w_900) == 1
        assert w_900[0] == (0.0, 900.0)

        # 900.1s -> Multiple windows
        w_901 = analysis.create_analysis_windows(901.0, window_size_s=720.0, overlap_s=120.0)
        assert len(w_901) == 2
        assert w_901[0] == (0.0, 720.0)
        assert w_901[1][0] == 600.0  # 720 - 120 = 600 step
        assert w_901[1][1] == 901.0

    def test_deduplicate_segments_overlap_and_tie_breaking(self):
        """Overlapping segments (>50%) keep higher-scoring candidate with deterministic tie-breaking."""
        # Seg A: 10s -> 40s (30s dur), score 85
        # Seg B: 15s -> 45s (30s dur), score 75 (overlaps A by 25s / 30s = 83% > 50%)
        # Seg C: 60s -> 90s (30s dur), score 70 (no overlap with A)
        seg_a = ViralSegment(10000, 40000, "Hook A", 20, 20, 20, 25, 85, "A")
        seg_b = ViralSegment(15000, 45000, "Hook B", 18, 18, 19, 20, 75, "B")
        seg_c = ViralSegment(60000, 90000, "Hook C", 15, 15, 20, 20, 70, "C")

        deduped = analysis.deduplicate_segments([seg_b, seg_a, seg_c], overlap_threshold=0.5)
        assert len(deduped) == 2
        assert deduped[0] is seg_a
        assert deduped[1] is seg_c

    def test_detect_cold_open_teaser_boundary_suppression(self):
        """Clips <20s or with opening hook_score >= 25 cleanly suppress cold-open teaser."""
        words = [
            Word("Hello", 0, 500),
            Word("world", 600, 1000),
            Word("this", 1100, 1500),
            Word("is", 1600, 2000),
            Word("a", 2100, 2500),
            Word("short", 2600, 3000),
            Word("clip", 3100, 3500),
            Word("now", 3600, 4000),
        ]
        # 1. Total segment duration < 20s
        short_seg = ViralSegment(0, 15000, "Title", 15, 15, 15, 15, 60, "R")
        assert analysis.detect_cold_open_teaser(short_seg, words) is None

        # 2. Strong opening hook >= 25
        long_seg = ViralSegment(0, 30000, "Title", 25, 20, 20, 20, 85, "R")
        assert analysis.detect_cold_open_teaser(long_seg, words) is None

    def test_heuristic_scorer_zero_words_and_short_clips(self):
        """Heuristic scorer handles empty words and short videos (<15s) gracefully."""
        # Empty transcript
        empty_t = TranscriptData(text="", language="en", words=[])
        assert analysis.heuristic_score_segments(empty_t) == []

        # Short clip under 15s clamps to available duration
        w1 = Word("Secret", 1000, 2000)
        w2 = Word("revealed", 2500, 5000)
        short_t = TranscriptData(text="Secret revealed", language="en", words=[w1, w2])
        segs = analysis.heuristic_score_segments(short_t)
        assert len(segs) == 1
        assert segs[0].start_time_ms == 0
        assert segs[0].end_time_ms == 5000

    def test_ollama_parser_resilience_markdown_and_corrupt_data(self):
        """Ollama JSON parser extracts from markdown codeblocks, clamps bounds, and filters corrupted items."""
        transcript = TranscriptData(
            text="Test transcript",
            language="en",
            words=[Word("Word", 0, 40000)],
        )
        raw_markdown = """
        Here is the virality analysis:
        ```json
        {
          "segments": [
            {
              "start_time": "00:10",
              "end_time": "00:40",
              "hook_score": "22.4",
              "engagement_score": 30,
              "value_score": -5,
              "shareability_score": 20,
              "hook_title": "Huge Secret Unveiled"
            },
            {
              "start_time_ms": 30000,
              "end_time_ms": 10000,
              "hook_title": "Invalid End Before Start"
            }
          ]
        }
        ```
        Hope this helps!
        """
        parsed = analysis._parse_and_validate_ollama_response(raw_markdown, transcript)
        assert len(parsed) == 1
        seg = parsed[0]
        assert seg.start_time_ms == 10000
        assert seg.end_time_ms == 40000
        assert seg.hook_score == 22
        assert seg.engagement_score == 25  # Clamped to 25
        assert seg.value_score == 0         # Clamped to 0
        assert seg.hook_title == "Huge Secret Unveiled"


# ==============================================================================
# 5. Reframing, Aspect Ratios & Face Centroid Adversarial
# ==============================================================================

class TestReframingAdversarial:
    """Stress tests for clippa.reframing module."""

    @pytest.mark.parametrize("val,expected", [
        (0, 0),
        (1, 0),
        (2, 2),
        (1079, 1078),
        (1080, 1080),
        (1081, 1080),
        (-3, 0),
        (1920.0, 1920),
    ])
    def test_round_to_even_comprehensive(self, val, expected):
        """Ensures all values convert to even integers for H.264 macroblock constraints."""
        res = reframing.round_to_even(val)
        assert res == expected
        assert res % 2 == 0

    def test_calculate_crop_coordinates_extreme_aspect_ratios(self):
        """Calculates 9:16 crop window across 16:9, 21:9, 4:3, 1:1, and native 9:16."""
        # 16:9 Landscape (1920x1080)
        cw, ch, cx, cy = reframing.calculate_crop_coordinates(1920, 1080, 1080, 1920)
        assert (cw, ch) == (608, 1080)
        assert cx == 656
        assert cy == 0

        # Native 9:16 Vertical (1080x1920)
        cw, ch, cx, cy = reframing.calculate_crop_coordinates(1080, 1920, 1080, 1920)
        assert (cw, ch) == (1080, 1920)
        assert cx == 0
        assert cy == 0

        # Ultra-wide 21:9 (2560x1080)
        cw, ch, cx, cy = reframing.calculate_crop_coordinates(2560, 1080, 1080, 1920)
        assert (cw, ch) == (608, 1080)
        assert 0 <= cx <= 2560 - 608

        # Extreme face coordinates outside frame boundaries are safely clamped
        cw, ch, cx, cy = reframing.calculate_crop_coordinates(
            1920, 1080, 1080, 1920, face_box=(-500, -500, 100, 100)
        )
        assert cx >= 0
        assert cy >= 0

        cw, ch, cx, cy = reframing.calculate_crop_coordinates(
            1920, 1080, 1080, 1920, face_box=(3000, 3000, 100, 100)
        )
        assert cx + cw <= 1920
        assert cy + ch <= 1080

    def test_smooth_face_trajectory_shot_cut_reset(self):
        """Exponential smoothing smoothly tracks gradual pan but resets instantly on shot cuts."""
        # Empty input
        assert reframing.smooth_face_trajectory([]) == []

        # All None values default to frame center
        all_none = reframing.smooth_face_trajectory([None, None, None], src_w=1920)
        assert all_none == [960.0, 960.0, 960.0]

        # Sudden jump > 800px immediately snaps rather than drifting slowly
        positions = [100.0, 105.0, 1500.0, 1502.0]
        smoothed = reframing.smooth_face_trajectory(positions, shot_cut_threshold=800.0)
        assert smoothed[0] == 100.0
        assert smoothed[1] < 105.0  # EMA damping
        assert smoothed[2] == 1500.0  # Shot cut reset!

        # Clamping to crop bounds with crop_w
        with_crop = reframing.smooth_face_trajectory([10.0, 1900.0], src_w=1920, crop_w=608)
        assert all(0.0 <= x <= (1920 - 608) for x in with_crop)
        assert all(int(x) % 2 == 0 for x in with_crop)

    def test_ken_burns_crops_velocity_suppression(self):
        """Dynamic talking head zoom suppresses when pan velocity exceeds threshold."""
        # Static face -> Zoom progress applied
        static_traj = [600.0, 600.0, 600.0, 600.0]
        static_crops = reframing.compute_ken_burns_crops(static_traj, duration_s=2.0, fps=10.0)
        assert len(static_crops) == 20
        assert static_crops[0]["zoom"] == 1.0
        assert static_crops[-1]["zoom"] == 1.05

        # Rapidly panning face -> Zoom suppressed to 1.0
        pan_traj = [100.0, 300.0, 500.0, 700.0]
        pan_crops = reframing.compute_ken_burns_crops(pan_traj, duration_s=2.0, fps=10.0)
        assert all(c["zoom"] == 1.0 for c in pan_crops)

    def test_build_crop_filter_native_vertical_passthrough(self, tmp_path):
        """Native 9:16 vertical input passes through without unnecessary crop filter."""
        dummy_vid = tmp_path / "vert.mp4"
        dummy_vid.write_bytes(b"dummy")

        with patch("clippa.reframing.probe_video_geometry", return_value=(1080, 1920, 30.0)):
            filter_expr = reframing.build_crop_filter(dummy_vid, 0, 5000, 1080, 1920)
            assert filter_expr == "scale=1080:1920"

    def test_select_primary_face_empty_and_crowded(self):
        """Primary face selector picks largest and most central speaker from crowd."""
        assert reframing.select_primary_face([]) is None

        # Face 1: tiny background face near center (x=900, y=500, w=50, h=50, score=0.9)
        # Face 2: big speaker face near center (x=800, y=300, w=300, h=300, score=0.95)
        # Face 3: medium face off-screen edge (x=10, y=100, w=200, h=200, score=0.9)
        faces = [
            (900.0, 500.0, 50.0, 50.0, 0.9),
            (800.0, 300.0, 300.0, 300.0, 0.95),
            (10.0, 100.0, 200.0, 200.0, 0.9),
        ]
        primary = reframing.select_primary_face(faces, frame_w=1920)
        assert primary == faces[1]


# ==============================================================================
# 6. Subtitles & Rendering Adversarial
# ==============================================================================

class TestSubtitlesAndRenderingAdversarial:
    """Stress tests for clippa.subtitles and clippa.rendering."""

    @pytest.mark.parametrize("ms,expected", [
        (0, "0:00:00.00"),
        (-500, "0:00:00.00"),
        (990, "0:00:00.99"),
        (65430, "0:01:05.43"),
        (3665430, "1:01:05.43"),
    ])
    def test_ms_to_ass_time_boundaries(self, ms, expected):
        """Converts millisecond timestamps into ASS centisecond representation."""
        assert subtitles.ms_to_ass_time(ms) == expected

    def test_emoji_injection_word_boundary_precision(self):
        """Ensures trigger word boundaries are respected and max 1 emoji per line is enforced."""
        # "fisherman" contains "fish" but should NOT trigger
        w_fish = Word("fisherman", 0, 500)
        res_fish = subtitles.inject_emojis([w_fish])
        assert res_fish[0].text == "fisherman"

        # Consecutive triggers "fire money cash" -> only 1 emoji added
        words = [Word("fire", 0, 500), Word("money", 500, 1000), Word("cash", 1000, 1500)]
        injected = subtitles.inject_emojis(words)
        emojis_count = sum(1 for w in injected if any(e in w.text for e in ["🔥", "💰", "💵"]))
        assert emojis_count == 1

    def test_generate_ass_subtitles_missing_hook_and_empty_words(self, tmp_path):
        """Empty words and missing hook title generate a valid skeleton ASS file."""
        out_ass = tmp_path / "empty.ass"
        subtitles.generate_ass_subtitles(words=[], hook_title=None, output_path=out_ass)

        content = out_ass.read_text(encoding="utf-8")
        assert "[Script Info]" in content
        assert "[V4+ Styles]" in content
        assert "Style: Hook" not in content  # Hook style omitted
        assert "Style: Default" in content

    def test_generate_ass_subtitles_template_fallback(self, tmp_path):
        """Unrecognized template preset falls back safely to 'default' theme."""
        out_ass = tmp_path / "fallback.ass"
        words = [Word("Testing", 100, 500), Word("fallback", 550, 900)]
        subtitles.generate_ass_subtitles(words=words, template="unknown_custom_theme", output_path=out_ass)

        content = out_ass.read_text(encoding="utf-8")
        assert "Style: Default" in content
        assert "Dialogue: 0," in content

    def test_escape_filter_path_windows_special_chars(self):
        """FFmpeg filter path escaping converts backslashes, colons, quotes, and brackets."""
        win_path = r"C:\My Videos\Shorts [2026]\clip '1'.ass"
        escaped = rendering.escape_filter_path(win_path)

        assert "\\" not in escaped.replace(r"\\", "").replace(r"\:", "").replace(r"\[", "").replace(r"\]", "").replace(r"\'", "")
        assert r"C\:" in escaped
        assert r"\[2026\]" in escaped
        assert r"\'1\'" in escaped

    def test_build_ffmpeg_render_args_filtergraph(self):
        """Verifies full FFmpeg render argument structure and libass filter invocation."""
        args = rendering.build_ffmpeg_render_args(
            source_video="input.mp4",
            output_file="output.mp4",
            start_ms=1000,
            end_ms=5000,
            crop_filter="crop=608:1080:656:0,scale=1080:1920",
            ass_path=r"C:\subs.ass",
        )
        assert "-ss" in args
        assert args[args.index("-ss") + 1] == "1.000"
        assert "-t" in args
        assert args[args.index("-t") + 1] == "4.000"
        assert "-vf" in args
        vf_idx = args.index("-vf")
        vf_expr = args[vf_idx + 1]
        assert "setpts=PTS-STARTPTS" in vf_expr
        assert "crop=608:1080:656:0" in vf_expr
        assert "ass=filename=" in vf_expr
        assert "-movflags" in args
        assert "+faststart" in args

    def test_render_clip_missing_inputs_media_not_found(self, tmp_path):
        """Missing input video or subtitle file raises MediaNotFoundError immediately."""
        with pytest.raises(MediaNotFoundError):
            rendering.render_clip("non_existent_source.mp4", 0, 5000)

        existing_vid = tmp_path / "vid.mp4"
        existing_vid.write_bytes(b"dummy")
        with pytest.raises(MediaNotFoundError):
            rendering.render_clip(existing_vid, 0, 5000, ass_path="non_existent.ass")

    def test_render_clip_atomic_temp_file_cleanup_on_error(self, tmp_path):
        """FFmpeg render errors cleanly unlink atomic temporary files."""
        src_vid = tmp_path / "source.mp4"
        src_vid.write_bytes(b"dummy")
        out_vid = tmp_path / "final_clip.mp4"

        # Simulate FFmpeg failure
        mock_proc = MagicMock(returncode=1, stderr="libass error: Could not create a libass track")
        with patch("subprocess.run", return_value=mock_proc):
            with pytest.raises(SubtitleBurnError):
                rendering.render_clip(src_vid, 0, 5000, output_file=out_vid)

        # Temporary files cleaned up
        tmp_files = list(tmp_path.glob(".*.tmp.mp4"))
        assert len(tmp_files) == 0


# ==============================================================================
# 7. CLI Runner, Argument Validation & Seam Safety Adversarial
# ==============================================================================

class TestCLIOptimizationAndSafetyAdversarial:
    """Stress tests for clippa.cli module."""

    @pytest.mark.parametrize("invalid_arg", ["0", "-1", "-10"])
    def test_cli_max_clips_validation(self, invalid_arg):
        """--max-clips <= 0 must fail argument validation with exit code 2."""
        with pytest.raises(SystemExit) as exc_info:
            cli.parse_cli_args(["--url", "test.mp4", "--max-clips", invalid_arg])
        assert exc_info.value.code == 2

    def test_get_unique_clip_path_multi_collision(self, tmp_path):
        """Sequentially increments filename when multiple collision files exist."""
        (tmp_path / "clip_1.mp4").write_text("1")
        (tmp_path / "clip_1_2.mp4").write_text("2")
        (tmp_path / "clip_1_3.mp4").write_text("3")

        unique_path = cli.get_unique_clip_path(tmp_path, 1)
        assert unique_path.name == "clip_1_4.mp4"
        assert not unique_path.exists()

    def test_run_pipeline_relative_word_timestamp_clamping(self):
        """Words straddling segment boundary clamp start to 0 and enforce minimum 50ms duration."""
        start_ms = 10000
        end_ms = 25000

        # Word A: begins before segment start
        # Word B: zero-duration inside segment
        words = [
            Word("Before", 9500, 11000),
            Word("ZeroDur", 12000, 12000),
        ]
        clip_words = []
        for w in words:
            if w.end >= start_ms and w.start <= end_ms:
                rel_start = max(0, w.start - start_ms)
                rel_end = max(rel_start + 50, w.end - start_ms)
                clip_words.append(Word(text=w.text, start=rel_start, end=rel_end))

        assert len(clip_words) == 2
        # Word A start clamped to 0
        assert clip_words[0].start == 0
        assert clip_words[0].end == 1000

        # Word B duration padded to at least 50ms
        assert clip_words[1].start == 2000
        assert clip_words[1].end == 2050
