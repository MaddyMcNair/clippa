"""
tests/test_adversarial_m2.py - Adversarial Stress & Edge Case Test Suite for Milestone 2
Target Modules: clippa.transcription and clippa.analysis

Adversarial Dimensions:
1. Pure silence audio stream (graceful handling, no unhandled exceptions, empty transcript).
2. Corrupted, truncated, and malformed .transcript_cache.json (automatic invalidation & regeneration).
3. Offline Ollama simulation (connection refused, timeouts, 500 error, malformed JSON, score >= 10).
4. Extreme speech rate (>300 WPM) and zero-punctuation transcripts (>500ms word pause segmentation).
5. Video windowing boundaries (exactly 900s vs 901s) and deduplication.
"""

from __future__ import annotations

import io
import json
import socket
import urllib.error
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import pytest

import clippa.transcription as transcription
from clippa.transcription import (
    Word,
    Utterance,
    TranscriptData,
    AudioExtractionError,
    TranscriptionError,
    extract_audio,
    transcribe_audio,
    get_cache_path,
    load_cached_transcript,
    save_cached_transcript,
)
import clippa.analysis as analysis
from clippa.analysis import (
    ViralSegment,
    TeaserCandidate,
    POWER_KEYWORDS,
    sanitize_hook_title,
    generate_hook_title,
    detect_cold_open_teaser,
    create_analysis_windows,
    deduplicate_segments,
    heuristic_score_segments,
    extract_viral_segments,
    analyze_long_video_windowed,
    _chunk_words,
    _parse_and_validate_ollama_response,
)
from tests.fixtures.synthetic import (
    generate_silent_mp4,
    generate_video_no_audio,
    generate_audio_only,
    generate_corrupt_media,
    require_ffmpeg,
)


# ==============================================================================
# Category 1: Pure Silence Audio Stream & Extraction Hazards
# ==============================================================================

class TestPureSilenceAndAudioHazards:

    def test_probe_s01_pure_silence_audio_transcription(self, tmp_path: Path):
        """Probe S1: Pure silence audio returns empty transcript without unhandled exception."""
        silent_audio = tmp_path / "pure_silence.mp3"
        silent_audio.write_bytes(b"\xFF\xFB" + b"\x00" * 1024)

        mock_model = MagicMock()
        mock_model.transcribe.return_value = (iter([]), SimpleNamespace(language="en", duration=5.0))

        with patch("faster_whisper.WhisperModel", return_value=mock_model):
            transcript = transcribe_audio(silent_audio, model_size="base", device="cpu")

            assert isinstance(transcript, TranscriptData)
            assert len(transcript.words) == 0
            assert transcript.text == ""
            assert transcript.language == "en"

            # Cache sidecar should still be generated for silence
            cache_file = get_cache_path(silent_audio)
            assert cache_file.exists()
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            assert data["words"] == []
            assert data["version"] == 3

    def test_probe_s02_real_ffmpeg_silent_mp4_extraction_and_transcription(self, tmp_path: Path):
        """Probe S2: Real FFmpeg-generated silent video extracts audio and transcribes cleanly."""
        video_path = tmp_path / "silent_video.mp4"
        generate_silent_mp4(video_path, duration_s=2.0)

        # 1. Verify audio extraction succeeds on pure silence
        extracted_audio = extract_audio(video_path)
        assert extracted_audio.exists()
        assert extracted_audio.stat().st_size > 0

        # 2. Verify transcription on this audio handles empty segments cleanly
        mock_model = MagicMock()
        mock_model.transcribe.return_value = (iter([]), SimpleNamespace(language="en", duration=2.0))

        with patch("faster_whisper.WhisperModel", return_value=mock_model):
            transcript = transcribe_audio(video_path, model_size="base", device="cpu")
            assert len(transcript.words) == 0
            assert transcript.text == ""

    def test_probe_s03_video_without_audio_stream_raises_audio_extraction_error(self, tmp_path: Path):
        """Probe S3: Video stripped of audio stream raises AudioExtractionError."""
        no_audio_video = tmp_path / "no_audio.mp4"
        generate_video_no_audio(no_audio_video, duration_s=2.0)

        with pytest.raises(AudioExtractionError):
            extract_audio(no_audio_video)

    def test_probe_s04_empty_0byte_video_raises_audio_extraction_error(self, tmp_path: Path):
        """Probe S4: 0-byte video file raises AudioExtractionError immediately."""
        zero_video = tmp_path / "zero_byte.mp4"
        zero_video.write_bytes(b"")

        with pytest.raises(AudioExtractionError):
            extract_audio(zero_video)

    def test_probe_s05_ultra_short_silent_audio_safe(self, tmp_path: Path):
        """Probe S5: Ultra-short audio stream (10ms) does not crash on empty/short timeline."""
        short_audio = tmp_path / "micro.wav"
        short_audio.write_bytes(b"RIFF" + b"\x00" * 40)

        mock_model = MagicMock()
        mock_model.transcribe.return_value = (iter([]), SimpleNamespace(language="en", duration=0.01))

        with patch("faster_whisper.WhisperModel", return_value=mock_model):
            res = transcribe_audio(short_audio)
            assert res.words == []

    def test_probe_s06_whisper_word_none_probability_does_not_crash(self, tmp_path: Path):
        """Probe S6: Whisper word with probability=None does not raise TypeError in float() conversion."""
        audio = tmp_path / "speech.mp3"
        audio.write_bytes(b"\xFF\xFB" + b"\x00" * 512)

        rw = SimpleNamespace(word="test", start=0.1, end=0.5, probability=None)
        seg = SimpleNamespace(start=0.1, end=0.5, text="test", words=[rw])
        mock_model = MagicMock()
        mock_model.transcribe.return_value = (iter([seg]), SimpleNamespace(language="en", duration=1.0))

        with patch("faster_whisper.WhisperModel", return_value=mock_model):
            transcript = transcribe_audio(audio)
            assert len(transcript.words) == 1
            assert transcript.words[0].confidence == 1.0


# ==============================================================================
# Category 2: Corrupted, Truncated & Malformed Sidecar Cache
# ==============================================================================

class TestTranscriptCacheHazards:

    def test_probe_c01_zero_byte_cache_auto_invalidates_and_regenerates(self, tmp_path: Path):
        """Probe C1: 0-byte cache file is rejected and re-transcribed with new cache."""
        audio = tmp_path / "speech.mp3"
        audio.write_bytes(b"\xFF\xFB" + b"\x00" * 512)

        # Pre-seed empty cache file
        cache_file = get_cache_path(audio)
        cache_file.write_bytes(b"")
        assert cache_file.stat().st_size == 0

        # load_cached_transcript returns None on 0-byte file
        assert load_cached_transcript(audio) is None

        # transcribe_audio detects empty cache, transcribes, and writes valid cache
        mock_model = MagicMock()
        seg = SimpleNamespace(
            start=0.0, end=1.0, text="regenerated words",
            words=[SimpleNamespace(word="regenerated", start=0.0, end=0.5, probability=0.9),
                   SimpleNamespace(word="words", start=0.5, end=1.0, probability=0.9)]
        )
        mock_model.transcribe.return_value = (iter([seg]), SimpleNamespace(language="en", duration=1.0))

        with patch("faster_whisper.WhisperModel", return_value=mock_model):
            res = transcribe_audio(audio)
            assert len(res.words) == 2
            assert res.words[0].text == "regenerated"

            # Cache file should now be populated and valid schema v3
            assert cache_file.stat().st_size > 0
            payload = json.loads(cache_file.read_text(encoding="utf-8"))
            assert payload["version"] == 3
            assert len(payload["words"]) == 2

    def test_probe_c02_truncated_json_cache_invalidates(self, tmp_path: Path):
        """Probe C2: Mid-flight truncated JSON cache is detected as corrupted and bypassed."""
        audio = tmp_path / "speech.mp3"
        audio.write_bytes(b"\xFF\xFB" + b"\x00" * 512)

        cache_file = get_cache_path(audio)
        cache_file.write_text('{"version": 3, "text": "Truncated content without closing brace...', encoding="utf-8")

        assert load_cached_transcript(audio) is None

        mock_model = MagicMock()
        seg = SimpleNamespace(start=0.0, end=0.8, text="repaired", words=[SimpleNamespace(word="repaired", start=0.0, end=0.8, probability=0.95)])
        mock_model.transcribe.return_value = (iter([seg]), SimpleNamespace(language="en", duration=0.8))

        with patch("faster_whisper.WhisperModel", return_value=mock_model):
            res = transcribe_audio(audio)
            assert res.words[0].text == "repaired"
            # Cache successfully overwritten with valid JSON
            new_data = json.loads(cache_file.read_text(encoding="utf-8"))
            assert new_data["version"] == 3

    def test_probe_c03_random_binary_garbage_cache_invalidates(self, tmp_path: Path):
        """Probe C3: Binary garbage bytes in cache file do not crash loader and regenerate cleanly."""
        audio = tmp_path / "speech.mp3"
        audio.write_bytes(b"\xFF\xFB" + b"\x00" * 512)

        cache_file = get_cache_path(audio)
        cache_file.write_bytes(b"\x00\xFF\xFE\xFD\x80\x90\xAA\xBB\xCC\xDD" * 20)

        assert load_cached_transcript(audio) is None

        mock_model = MagicMock()
        mock_model.transcribe.return_value = (iter([]), SimpleNamespace(language="en", duration=0.5))

        with patch("faster_whisper.WhisperModel", return_value=mock_model):
            res = transcribe_audio(audio)
            assert res is not None

    def test_probe_c04_outdated_schema_version_invalidates(self, tmp_path: Path):
        """Probe C4: Schema version 1 or 2 is rejected and upgraded to version 3."""
        audio = tmp_path / "speech.mp3"
        audio.write_bytes(b"\xFF\xFB" + b"\x00" * 512)

        cache_file = get_cache_path(audio)
        v1_data = {
            "version": 1,
            "text": "old version",
            "words": [{"text": "old", "start": 0, "end": 500}],
        }
        cache_file.write_text(json.dumps(v1_data), encoding="utf-8")

        # Must return None so pipeline re-transcribes for v3 compliance
        assert load_cached_transcript(audio) is None

    def test_probe_c05_read_only_filesystem_cache_write_fallback(self, tmp_path: Path, monkeypatch):
        """Probe C5: Write permission error to cache sidecar logs warning and continues in-memory."""
        audio = tmp_path / "speech.mp3"
        audio.write_bytes(b"\xFF\xFB" + b"\x00" * 512)

        t_data = TranscriptData(text="test in-memory", language="en", words=[Word("test", 0, 500)])

        def mock_write_text(*args, **kwargs):
            raise PermissionError("Access denied (read-only filesystem)")

        monkeypatch.setattr(Path, "write_text", mock_write_text)

        # save_cached_transcript returns None without unhandled exception
        result = save_cached_transcript(audio, t_data)
        assert result is None

    def test_probe_c06_force_retranscribe_overwrites_valid_cache(self, tmp_path: Path):
        """Probe C6: force_retranscribe=True bypasses and overwrites an existing valid cache."""
        audio = tmp_path / "speech.mp3"
        audio.write_bytes(b"\xFF\xFB" + b"\x00" * 512)

        old_transcript = TranscriptData(text="original text", language="en", words=[Word("original", 0, 500)])
        save_cached_transcript(audio, old_transcript)

        # When force_retranscribe=False, reads cached
        cached = transcribe_audio(audio, force_retranscribe=False)
        assert cached.text == "original text"

        # When force_retranscribe=True, runs whisper and overwrites
        mock_model = MagicMock()
        seg = SimpleNamespace(start=0.0, end=0.6, text="new fresh text", words=[SimpleNamespace(word="fresh", start=0.0, end=0.6, probability=0.99)])
        mock_model.transcribe.return_value = (iter([seg]), SimpleNamespace(language="en", duration=0.6))

        with patch("faster_whisper.WhisperModel", return_value=mock_model):
            fresh = transcribe_audio(audio, force_retranscribe=True)
            assert fresh.text == "new fresh text"

            # Cache on disk must now reflect fresh text
            on_disk = load_cached_transcript(audio)
            assert on_disk.text == "new fresh text"


# ==============================================================================
# Category 3: Offline Ollama Simulation & Fallback Scorer Integrity
# ==============================================================================

class TestOfflineOllamaSimulation:

    def _make_sample_transcript(self) -> TranscriptData:
        """Helper to create a 35-second transcript with power words."""
        words: list[Word] = []
        cur_t = 0
        tokens = (
            "Why do ninety nine percent of people make this catastrophic mistake ? "
            "The secret is that billionaires never follow traditional advice . "
            "Here is the proven rule that changes your financial future forever . "
            "First step is understanding the math behind compound interest and cash flow ."
        ).split()

        for token in tokens:
            dur = 350
            words.append(Word(text=token, start=cur_t, end=cur_t + dur, confidence=0.95))
            cur_t += dur + 50

        return TranscriptData(text=" ".join(tokens), language="en", words=words)

    def test_probe_o01_ollama_port_closed_connection_refused_fallback(self):
        """Probe O1: Completely closed Ollama port triggers seamless fallback to heuristic scoring."""
        transcript = self._make_sample_transcript()

        # Point to closed high port
        unreachable_url = "http://127.0.0.1:59999/api/generate"

        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError(ConnectionRefusedError("Offline"))):
            segments = extract_viral_segments(
                transcript=transcript,
                max_clips=2,
                llm_model="ollama:llama3.2:3b",
            )

            assert len(segments) > 0
            assert len(segments) <= 2

            for seg in segments:
                assert isinstance(seg, ViralSegment)
                # Verify total score >= 10 baseline guarantee
                assert seg.total_score >= 10
                # Verify subscore ranges [0, 25]
                assert 0 <= seg.hook_score <= 25
                assert 0 <= seg.engagement_score <= 25
                assert 0 <= seg.value_score <= 25
                assert 0 <= seg.shareability_score <= 25
                # Verify duration bounds [15s, 60s]
                dur = seg.end_time_ms - seg.start_time_ms
                assert 15_000 <= dur <= 60_000
                # Verify reasoning is non-empty
                assert seg.reasoning

    def test_probe_o02_ollama_http_500_internal_error_fallback(self):
        """Probe O2: HTTP 500 error from Ollama triggers graceful heuristic fallback."""
        transcript = self._make_sample_transcript()

        http_err = urllib.error.HTTPError(
            url="http://localhost:11434/api/generate",
            code=500,
            msg="Internal Server Error: Model out of memory",
            hdrs={},
            fp=io.BytesIO(b"Internal Server Error: Model out of memory"),
        )

        with patch("urllib.request.urlopen", side_effect=http_err):
            segments = extract_viral_segments(transcript, max_clips=1)
            assert len(segments) == 1
            assert segments[0].total_score >= 10

    def test_probe_o03_ollama_socket_timeout_fallback(self):
        """Probe O3: Socket timeout triggers fallback without hanging."""
        transcript = self._make_sample_transcript()

        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError(socket.timeout("Request timed out"))):
            segments = extract_viral_segments(transcript, max_clips=2)
            assert len(segments) > 0

    def test_probe_o04_ollama_returns_malformed_non_json_garbage(self):
        """Probe O4: Ollama returning conversational text without JSON triggers fallback."""
        transcript = self._make_sample_transcript()

        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({
            "response": "I am an AI assistant and I cannot extract viral segments for this video."
        }).encode("utf-8")

        with patch("urllib.request.urlopen", return_value=mock_resp):
            segments = extract_viral_segments(transcript, max_clips=2)
            # Must fall back to heuristic segments
            assert len(segments) > 0
            assert segments[0].total_score >= 10

    def test_probe_o05_ollama_markdown_codeblock_json_parsing(self):
        """Probe O5: Ollama returning markdown code block ```json ... ``` parses correctly."""
        transcript = self._make_sample_transcript()

        ollama_inner = {
            "segments": [
                {
                    "start_time_ms": 1000,
                    "end_time_ms": 25000,
                    "hook_title": "The Secret Billionaires Hide",
                    "hook_score": 22,
                    "engagement_score": 21,
                    "value_score": 23,
                    "shareability_score": 20,
                    "total_score": 86,
                    "reasoning": "Strong opening premise with question."
                }
            ]
        }
        markdown_response = f"Here are the viral highlights:\n```json\n{json.dumps(ollama_inner)}\n```\nEnjoy!"

        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({"response": markdown_response}).encode("utf-8")

        with patch("urllib.request.urlopen", return_value=mock_resp):
            segments = extract_viral_segments(transcript, max_clips=2)
            assert len(segments) == 1
            assert segments[0].hook_title == "The Secret Billionaires Hide"
            assert segments[0].total_score == 86

    def test_probe_o06_ollama_out_of_range_scores_clamped(self):
        """Probe O6: Out-of-bounds scores (-10, 500) from Ollama are clamped to [0, 25] and [0, 100]."""
        transcript = self._make_sample_transcript()

        raw_json = json.dumps({
            "segments": [
                {
                    "start_time_ms": 0,
                    "end_time_ms": 20000,
                    "hook_title": "Wild Exaggerated Claim",
                    "hook_score": -10,
                    "engagement_score": 100,
                    "value_score": 250,
                    "shareability_score": -5,
                    "reasoning": "Clamping test"
                }
            ]
        })

        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({"response": raw_json}).encode("utf-8")

        with patch("urllib.request.urlopen", return_value=mock_resp):
            segments = extract_viral_segments(transcript, max_clips=1)
            assert len(segments) == 1
            seg = segments[0]
            assert seg.hook_score == 0
            assert seg.engagement_score == 25
            assert seg.value_score == 25
            assert seg.shareability_score == 0
            assert seg.total_score == 50  # 0 + 25 + 25 + 0

    def test_probe_o07_heuristic_explicit_mode_never_calls_ollama(self):
        """Probe O7: Setting llm_model='heuristic' executes heuristic scorer directly without network."""
        transcript = self._make_sample_transcript()

        with patch("urllib.request.urlopen") as mock_url:
            segments = extract_viral_segments(transcript, max_clips=2, llm_model="heuristic")
            assert mock_url.call_count == 0
            assert len(segments) > 0

    def test_probe_o08_ollama_teaser_outside_segment_bounds_discarded(self):
        """Probe O8: Ollama returning hook_start_time_ms outside segment duration must be clamped or discarded."""
        transcript = self._make_sample_transcript()
        raw_json = json.dumps({
            "segments": [
                {
                    "start_time_ms": 1000,
                    "end_time_ms": 25000,
                    "hook_title": "Valid Hook Title",
                    "hook_start_time_ms": 50000,  # Far outside [1000, 25000]!
                    "hook_end_time_ms": 55000,
                }
            ]
        })
        segs = _parse_and_validate_ollama_response(raw_json, transcript)
        assert len(segs) == 1
        seg = segs[0]
        if seg.hook_start_time_ms is not None:
            assert seg.start_time_ms <= seg.hook_start_time_ms < seg.hook_end_time_ms <= seg.end_time_ms, (
                f"Teaser [{seg.hook_start_time_ms}, {seg.hook_end_time_ms}] is outside "
                f"segment [{seg.start_time_ms}, {seg.end_time_ms}]!"
            )

    def test_probe_o09_ollama_float_string_scores_and_timestamps(self):
        """Probe O9: Ollama returning string floats for timestamps and scores should not raise unhandled ValueError."""
        transcript = self._make_sample_transcript()
        raw_json = json.dumps({
            "segments": [
                {
                    "start_time_ms": "1000.0",
                    "end_time_ms": "20000.0",
                    "hook_score": "22.5",
                    "engagement_score": "21.0",
                    "value_score": "23.5",
                    "shareability_score": "20.0",
                    "hook_title": "String Float Scores Test",
                }
            ]
        })
        segs = _parse_and_validate_ollama_response(raw_json, transcript)
        assert len(segs) == 1
        assert segs[0].start_time_ms == 1000
        assert segs[0].hook_score == 22


# ==============================================================================
# Category 4: Extreme Speech Rate (>300 WPM) & Zero-Punctuation Transcripts
# ==============================================================================

class TestExtremeSpeechRateAndPunctuation:

    def test_probe_r01_extreme_speech_rate_360_wpm(self):
        """Probe R1: Extreme speech rate (>350 WPM) computes scores without divide-by-zero or crash."""
        words: list[Word] = []
        cur_t = 0
        # 120 words in 20,000ms = 360 WPM!
        for i in range(120):
            token = f"word{i}"
            words.append(Word(text=token, start=cur_t, end=cur_t + 140, confidence=0.88))
            cur_t += 166

        transcript = TranscriptData(
            text=" ".join(w.text for w in words),
            language="en",
            words=words,
        )

        segments = heuristic_score_segments(transcript, max_clips=2)
        assert len(segments) > 0
        for seg in segments:
            assert seg.total_score >= 10
            assert 0 <= seg.engagement_score <= 25

    def test_probe_r02_zero_punctuation_segmented_by_500ms_pauses(self):
        """Probe R2: Zero-punctuation transcript correctly segments words on >500ms pauses."""
        # 3 phrases without any punctuation, separated by 700ms silence
        phrase1 = ["why", "do", "people", "never", "listen", "to", "the", "secret"]  # 8 words
        phrase2 = ["this", "crucial", "lesson", "explains", "everything", "about", "money"]  # 7 words
        phrase3 = ["now", "you", "know", "the", "shocking", "truth", "behind", "success"]  # 8 words

        words: list[Word] = []
        cur_t = 1000

        for phrase in [phrase1, phrase2, phrase3]:
            for w in phrase:
                words.append(Word(text=w, start=cur_t, end=cur_t + 800, confidence=0.9))
                cur_t += 900  # 100ms gap within phrase
            cur_t += 700  # 700ms pause between phrases!

        # Verify _chunk_words splits into 3 chunks
        chunks = _chunk_words(words)
        assert len(chunks) == 3, f"Expected 3 chunks split by >500ms pauses, got {len(chunks)}"

        transcript = TranscriptData(text=" ".join(w.text for w in words), language="en", words=words)
        segments = heuristic_score_segments(transcript, max_clips=2)
        assert len(segments) > 0
        assert segments[0].total_score >= 10

    def test_probe_r03_zero_punctuation_and_zero_pauses_wall_of_text(self):
        """Probe R3: Continuous speech without punctuation or long pauses falls back to baseline safely."""
        words: list[Word] = []
        cur_t = 0
        # 100 words with 20ms gap each over 60 seconds
        for i in range(100):
            words.append(Word(text=f"tok{i}", start=cur_t, end=cur_t + 500, confidence=0.9))
            cur_t += 520

        transcript = TranscriptData(text=" ".join(w.text for w in words), language="en", words=words)
        segments = heuristic_score_segments(transcript, max_clips=1)
        assert len(segments) == 1
        assert segments[0].total_score >= 10
        assert segments[0].end_time_ms > segments[0].start_time_ms

    def test_probe_r04_repetitive_single_word_transcript_no_division_by_zero(self):
        """Probe R4: Repetitive word ('test test test') does not cause division by zero in TTR."""
        words: list[Word] = []
        cur_t = 0
        for _ in range(50):
            words.append(Word(text="test", start=cur_t, end=cur_t + 400, confidence=0.9))
            cur_t += 500

        transcript = TranscriptData(text=" ".join(w.text for w in words), language="en", words=words)
        segments = heuristic_score_segments(transcript, max_clips=1)
        assert len(segments) == 1
        assert segments[0].total_score >= 10

    def test_probe_r05_hook_title_on_extreme_inputs(self):
        """Probe R5: Hook title generation sanitizes extreme, punctuation-only, or overly long text."""
        # 1. Punctuation only
        t1 = sanitize_hook_title("?!?!?! ... ---")
        assert t1 != ""
        assert len(t1.split()) >= 3

        # 2. Overly long string (>100 characters)
        long_title = "This Is An Extremely Long Title That Keeps Going On And On Without Any Regard For Length Limits Or Brevity At All"
        t2 = sanitize_hook_title(long_title)
        assert len(t2) <= 64
        assert len(t2.split()) <= 9

        # 3. ASS control character injection attempt
        injection = "The {\\b1}Secret{\\b0} You \\N Never \\h Knew"
        t3 = sanitize_hook_title(injection)
        assert "{" not in t3
        assert "}" not in t3
        assert "\\" not in t3


# ==============================================================================
# Category 5: Video Windowing Boundaries & Deduplication
# ==============================================================================

class TestVideoWindowingAndDeduplication:

    def test_probe_w01_exact_window_boundary_900s_vs_901s(self):
        """Probe W1: Boundary test between single window (900.0s) and windowed analysis (901.0s)."""
        # Exactly 15 minutes (900.0s)
        w_900 = create_analysis_windows(total_duration_s=900.0)
        assert len(w_900) == 1
        assert w_900[0] == (0.0, 900.0)

        # 15 minutes + 1 second (901.0s)
        w_901 = create_analysis_windows(total_duration_s=901.0, window_size_s=720.0, overlap_s=120.0)
        assert len(w_901) == 2
        assert w_901[0] == (0.0, 720.0)
        assert w_901[1] == (600.0, 901.0)

        # Verify exact 120s overlap
        overlap = w_901[0][1] - w_901[1][0]
        assert overlap == 120.0

    def test_probe_w02_deduplication_keeps_higher_scoring_overlapping_segment(self):
        """Probe W2: Deduplication with >50% overlap keeps higher-scoring segment and drops lower."""
        # Candidate A: [600s, 640s], score 75 (lower)
        seg_a = ViralSegment(
            start_time_ms=600_000,
            end_time_ms=640_000,
            hook_title="Candidate A Lower",
            hook_score=18, engagement_score=19, value_score=19, shareability_score=19,
            total_score=75, reasoning="Lower score segment"
        )
        # Candidate B: [610s, 650s], score 92 (higher)
        # Overlap is [610s, 640s] = 30s. Shorter is 40s. Ratio = 30/40 = 75% > 50%
        seg_b = ViralSegment(
            start_time_ms=610_000,
            end_time_ms=650_000,
            hook_title="Candidate B Higher",
            hook_score=23, engagement_score=23, value_score=23, shareability_score=23,
            total_score=92, reasoning="Higher score segment"
        )

        # Order 1: seg_a then seg_b
        deduped1 = deduplicate_segments([seg_a, seg_b], overlap_threshold=0.5)
        assert len(deduped1) == 1
        assert deduped1[0].hook_title == "Candidate B Higher"
        assert deduped1[0].total_score == 92

        # Order 2: seg_b then seg_a
        deduped2 = deduplicate_segments([seg_b, seg_a], overlap_threshold=0.5)
        assert len(deduped2) == 1
        assert deduped2[0].hook_title == "Candidate B Higher"

    def test_probe_w03_deterministic_tie_breaking_on_identical_scores(self):
        """Probe W3: Two overlapping segments with identical scores resolve deterministically."""
        seg1 = ViralSegment(
            start_time_ms=100_000,
            end_time_ms=140_000,
            hook_title="Earlier Segment",
            hook_score=20, engagement_score=20, value_score=20, shareability_score=20,
            total_score=80, reasoning="Tie 1"
        )
        seg2 = ViralSegment(
            start_time_ms=110_000,
            end_time_ms=150_000,
            hook_title="Later Segment",
            hook_score=20, engagement_score=20, value_score=20, shareability_score=20,
            total_score=80, reasoning="Tie 2"
        )

        res_a = deduplicate_segments([seg1, seg2], overlap_threshold=0.5)
        res_b = deduplicate_segments([seg2, seg1], overlap_threshold=0.5)

        assert len(res_a) == 1
        assert len(res_b) == 1
        # Deterministic: earlier start time is prioritized
        assert res_a[0].hook_title == "Earlier Segment"
        assert res_b[0].hook_title == "Earlier Segment"

    def test_probe_w04_non_overlapping_segments_all_preserved(self):
        """Probe W4: Non-overlapping segments are cleanly retained without false deduplication."""
        segs = [
            ViralSegment(0, 30_000, "Clip 1", 20, 20, 20, 20, 80, "R1"),
            ViralSegment(60_000, 90_000, "Clip 2", 21, 21, 21, 21, 84, "R2"),
            ViralSegment(120_000, 150_000, "Clip 3", 22, 22, 22, 22, 88, "R3"),
        ]
        deduped = deduplicate_segments(segs, overlap_threshold=0.5)
        assert len(deduped) == 3

    def test_probe_w05_long_video_windowed_analysis_30_minutes(self):
        """Probe W5: 30-minute video partitions into windows and returns top candidate clips."""
        # Create a synthetic transcript spanning 1800 seconds (30 minutes)
        words: list[Word] = []
        cur_t = 0
        # Sparse speech every 10 seconds across 30 mins
        for sec in range(0, 1800, 10):
            token = "secret" if sec % 60 == 0 else "word"
            words.append(Word(text=token, start=sec * 1000, end=(sec + 4) * 1000, confidence=0.9))

        transcript = TranscriptData(text=" ".join(w.text for w in words), language="en", words=words)

        segments = analyze_long_video_windowed(
            transcript=transcript,
            window_size_s=720,
            overlap_s=120,
            max_clips=3,
            llm_model="heuristic",
        )

        assert len(segments) <= 3
        for seg in segments:
            assert isinstance(seg, ViralSegment)
            assert seg.total_score >= 10
            assert seg.start_time_ms >= 0
            assert seg.end_time_ms <= 1800 * 1000
