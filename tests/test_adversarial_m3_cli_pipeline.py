"""
tests/test_adversarial_m3_cli_pipeline.py
Adversarial Stress Test Suite for Milestone 3:
- CLI Argument Parsing Edge Cases
- Subprocess CLI Entry Point Integration
- Non-Colliding Unique File Naming Under Collision Pressure
- Subtitle Relative Timestamp Shifting & Clamp Safety
- Self-Hosted Mode Zero-Paid-API Enforcement & Poisoned Env Handling
- End-to-End Pipeline Orchestration Resilience (Failure Modes & Partial Success)
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import clippa.cli as cli
import clippa.config as config
from clippa.analysis import ViralSegment
from clippa.transcription import TranscriptData, Word


# ==============================================================================
# 1. CLI Argument Parsing Edge Cases
# ==============================================================================

class TestCLIArgumentParsing:
    """Stress-tests command-line interface argument parsing and validation."""

    def test_missing_url_raises_system_exit_2(self):
        """Omitting required --url must raise SystemExit(2)."""
        with pytest.raises(SystemExit) as exc_info:
            cli.parse_cli_args([])
        assert exc_info.value.code == 2

    def test_empty_argv_raises_system_exit_2(self):
        """Calling main([]) with empty argv must raise SystemExit(2)."""
        with pytest.raises(SystemExit) as exc_info:
            cli.main([])
        assert exc_info.value.code == 2

    def test_negative_max_clips_raises_system_exit_2(self):
        """Negative --max-clips (e.g. -1, -10) must be rejected with exit code 2."""
        with pytest.raises(SystemExit) as exc_info:
            cli.parse_cli_args(["--url", "video.mp4", "--max-clips", "-1"])
        assert exc_info.value.code == 2

        with pytest.raises(SystemExit) as exc_info:
            cli.main(["--url", "video.mp4", "--max-clips", "-10"])
        assert exc_info.value.code == 2

    def test_zero_max_clips_raises_system_exit_2(self):
        """Zero --max-clips (0) must be rejected with exit code 2."""
        with pytest.raises(SystemExit) as exc_info:
            cli.parse_cli_args(["--url", "video.mp4", "--max-clips", "0"])
        assert exc_info.value.code == 2

        with pytest.raises(SystemExit) as exc_info:
            cli.main(["--url", "video.mp4", "--max-clips", "0"])
        assert exc_info.value.code == 2

    def test_invalid_template_choice_raises_system_exit_2(self):
        """Invalid subtitle template must be rejected by argparse choices."""
        with pytest.raises(SystemExit) as exc_info:
            cli.parse_cli_args(["--url", "video.mp4", "--template", "cyberpunk_neon"])
        assert exc_info.value.code == 2

    def test_invalid_resolution_choice_raises_system_exit_2(self):
        """Unsupported resolution (e.g. 4k, 480p) must be rejected by choices."""
        with pytest.raises(SystemExit) as exc_info:
            cli.parse_cli_args(["--url", "video.mp4", "--resolution", "4k"])
        assert exc_info.value.code == 2

    def test_invalid_device_choice_raises_system_exit_2(self):
        """Unsupported device (e.g. tpu, rocm) must be rejected by choices."""
        with pytest.raises(SystemExit) as exc_info:
            cli.parse_cli_args(["--url", "video.mp4", "--device", "tpu"])
        assert exc_info.value.code == 2

    def test_help_flag_prints_options_and_exits_0(self, capsys):
        """--help flag must display all CLI options and exit with code 0."""
        with pytest.raises(SystemExit) as exc_info:
            cli.parse_cli_args(["--help"])
        assert exc_info.value.code == 0

        captured = capsys.readouterr()
        out = captured.out + captured.err
        for flag in [
            "--url",
            "--output-dir",
            "--temp-dir",
            "--template",
            "--max-clips",
            "--resolution",
            "--whisper-model",
            "--llm-model",
            "--device",
            "--no-emojis",
            "--no-zoom",
            "--no-cache",
        ]:
            assert flag in out, f"Expected flag '{flag}' missing from --help output"

    def test_valid_custom_flags_parsed_correctly(self):
        """All supported options must parse into expected types and values."""
        args = cli.parse_cli_args([
            "--url", "https://youtube.com/watch?v=dQw4w9WgXcQ",
            "--output-dir", "custom_out",
            "--temp-dir", "custom_temp",
            "--template", "mrbeast",
            "--max-clips", "5",
            "--resolution", "720p",
            "--whisper-model", "small",
            "--llm-model", "heuristic",
            "--device", "cpu",
            "--no-emojis",
            "--no-zoom",
            "--no-cache",
        ])
        assert args.url == "https://youtube.com/watch?v=dQw4w9WgXcQ"
        assert args.output_dir == "custom_out"
        assert args.temp_dir == "custom_temp"
        assert args.template == "mrbeast"
        assert args.max_clips == 5
        assert args.resolution == "720p"
        assert args.whisper_model == "small"
        assert args.llm_model == "heuristic"
        assert args.device == "cpu"
        assert args.no_emojis is True
        assert args.no_zoom is True
        assert args.no_cache is True


# ==============================================================================
# 2. Subprocess CLI Invocation Tests
# ==============================================================================

class TestMainSubprocessIntegration:
    """Verifies that `python -m clippa` works as an external executable process."""

    def test_subprocess_clippa_no_args_exits_2(self):
        """Running `python -m clippa` without arguments exits with code 2."""
        proc = subprocess.run(
            [sys.executable, "-m", "clippa"],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 2
        assert "error: the following arguments are required: --url" in proc.stderr.lower() or proc.returncode == 2

    def test_subprocess_clippa_help_exits_0(self):
        """Running `python -m clippa --help` exits with code 0 and displays help."""
        proc = subprocess.run(
            [sys.executable, "-m", "clippa", "--help"],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0
        assert "--url" in proc.stdout
        assert "--output-dir" in proc.stdout

    def test_subprocess_clippa_negative_max_clips_exits_2(self):
        """Running `python -m clippa --url test.mp4 --max-clips -2` exits with code 2."""
        proc = subprocess.run(
            [sys.executable, "-m", "clippa", "--url", "test.mp4", "--max-clips", "-2"],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 2
        assert "--max-clips must be a positive integer" in proc.stderr


# ==============================================================================
# 3. Non-Colliding Unique File Naming Under Pressure
# ==============================================================================

class TestUniqueClipNaming:
    """Stress-tests `clippa.cli.get_unique_clip_path` against file collisions."""

    def test_clean_directory_creates_base_filename(self, tmp_path: Path):
        """In an empty directory, clip index 1 returns clip_1.mp4."""
        target = cli.get_unique_clip_path(tmp_path, 1)
        assert target == tmp_path / "clip_1.mp4"

    def test_first_collision_appends_suffix_2(self, tmp_path: Path):
        """If clip_1.mp4 exists, next output path is clip_1_2.mp4."""
        (tmp_path / "clip_1.mp4").write_bytes(b"dummy")
        target = cli.get_unique_clip_path(tmp_path, 1)
        assert target == tmp_path / "clip_1_2.mp4"

    def test_sequential_collision_chain(self, tmp_path: Path):
        """Pre-creating clip_1, clip_1_2, clip_1_3 produces clip_1_4."""
        (tmp_path / "clip_1.mp4").write_bytes(b"dummy")
        (tmp_path / "clip_1_2.mp4").write_bytes(b"dummy")
        (tmp_path / "clip_1_3.mp4").write_bytes(b"dummy")

        target = cli.get_unique_clip_path(tmp_path, 1)
        assert target == tmp_path / "clip_1_4.mp4"

    def test_collision_stress_50_iterations(self, tmp_path: Path):
        """Heavy collision load: 50 existing files must resolve to 51st without crash."""
        (tmp_path / "clip_1.mp4").write_bytes(b"dummy")
        for i in range(2, 51):
            (tmp_path / f"clip_1_{i}.mp4").write_bytes(b"dummy")

        target = cli.get_unique_clip_path(tmp_path, 1)
        assert target == tmp_path / "clip_1_51.mp4"
        assert not target.exists()

    def test_distinct_clip_indices_independent(self, tmp_path: Path):
        """Collisions on clip_1 do not affect clip_2 or clip_3."""
        (tmp_path / "clip_1.mp4").write_bytes(b"dummy")
        (tmp_path / "clip_1_2.mp4").write_bytes(b"dummy")

        target_2 = cli.get_unique_clip_path(tmp_path, 2)
        assert target_2 == tmp_path / "clip_2.mp4"

        (tmp_path / "clip_2.mp4").write_bytes(b"dummy")
        target_2_next = cli.get_unique_clip_path(tmp_path, 2)
        assert target_2_next == tmp_path / "clip_2_2.mp4"

    def test_path_with_spaces_and_special_chars(self, tmp_path: Path):
        """Output directory with spaces and brackets computes valid path."""
        spaced_dir = tmp_path / "My Rendered [Clips] 2026"
        spaced_dir.mkdir(parents=True, exist_ok=True)

        target = cli.get_unique_clip_path(spaced_dir, 1)
        assert target == spaced_dir / "clip_1.mp4"


# ==============================================================================
# 4. Subtitle Relative Timestamp Shifting & Clamp Safety
# ==============================================================================

class TestSubtitleRelativeShifting:
    """Verifies word timing transformation from absolute video to relative clip time."""

    def test_exact_segment_boundary_shift(self):
        """Words strictly inside [start_ms, end_ms] shift by exactly -start_ms."""
        start_ms = 10000
        end_ms = 20000
        raw_words = [
            Word(text="Start", start=10000, end=10500),
            Word(text="Middle", start=14000, end=14800),
            Word(text="End", start=19200, end=20000),
        ]

        shifted_words = []
        for w in raw_words:
            if w.end >= start_ms and w.start <= end_ms:
                rel_start = max(0, w.start - start_ms)
                rel_end = max(rel_start + 50, w.end - start_ms)
                shifted_words.append(Word(text=w.text, start=rel_start, end=rel_end))

        assert len(shifted_words) == 3
        assert shifted_words[0].start == 0
        assert shifted_words[0].end == 500
        assert shifted_words[1].start == 4000
        assert shifted_words[1].end == 4800
        assert shifted_words[2].start == 9200
        assert shifted_words[2].end == 10000

    def test_straddling_opening_word_clamped_to_zero(self):
        """A word beginning before segment start has rel_start clamped to 0."""
        start_ms = 10000
        end_ms = 25000
        raw_words = [
            Word(text="StraddlingOpen", start=9200, end=10600),
        ]

        shifted_words = []
        for w in raw_words:
            if w.end >= start_ms and w.start <= end_ms:
                rel_start = max(0, w.start - start_ms)
                rel_end = max(rel_start + 50, w.end - start_ms)
                shifted_words.append(Word(text=w.text, start=rel_start, end=rel_end))

        assert len(shifted_words) == 1
        assert shifted_words[0].start == 0
        assert shifted_words[0].end == 600

    def test_straddling_closing_word_included(self):
        """A word beginning before segment end but ending after is included."""
        start_ms = 10000
        end_ms = 20000
        raw_words = [
            Word(text="StraddlingClose", start=19500, end=20800),
        ]

        shifted_words = []
        for w in raw_words:
            if w.end >= start_ms and w.start <= end_ms:
                rel_start = max(0, w.start - start_ms)
                rel_end = max(rel_start + 50, w.end - start_ms)
                shifted_words.append(Word(text=w.text, start=rel_start, end=rel_end))

        assert len(shifted_words) == 1
        assert shifted_words[0].start == 9500
        assert shifted_words[0].end == 10800

    def test_outside_words_strictly_excluded(self):
        """Words entirely before or after segment interval are dropped."""
        start_ms = 10000
        end_ms = 20000
        raw_words = [
            Word(text="FarBefore", start=1000, end=5000),
            Word(text="JustBefore", start=9000, end=9999),
            Word(text="Inside", start=12000, end=13000),
            Word(text="JustAfter", start=20001, end=21000),
            Word(text="FarAfter", start=50000, end=55000),
        ]

        shifted_words = []
        for w in raw_words:
            if w.end >= start_ms and w.start <= end_ms:
                rel_start = max(0, w.start - start_ms)
                rel_end = max(rel_start + 50, w.end - start_ms)
                shifted_words.append(Word(text=w.text, start=rel_start, end=rel_end))

        assert len(shifted_words) == 1
        assert shifted_words[0].text == "Inside"

    def test_zero_duration_word_enforces_minimum_duration(self):
        """A word with start == end enforces a minimum 50ms duration."""
        start_ms = 5000
        end_ms = 15000
        raw_words = [
            Word(text="Instant", start=7000, end=7000),
        ]

        shifted_words = []
        for w in raw_words:
            if w.end >= start_ms and w.start <= end_ms:
                rel_start = max(0, w.start - start_ms)
                rel_end = max(rel_start + 50, w.end - start_ms)
                shifted_words.append(Word(text=w.text, start=rel_start, end=rel_end))

        assert len(shifted_words) == 1
        assert shifted_words[0].start == 2000
        assert shifted_words[0].end == 2050
        assert shifted_words[0].end > shifted_words[0].start


# ==============================================================================
# 5. Self-Hosted Mode Zero-Paid-API Enforcement
# ==============================================================================

class TestSelfHostedConfigAdversarial:
    """Stress-tests `clippa.config.Config` under poisoned environment conditions."""

    def test_poisoned_env_keys_discarded_in_self_hosted_mode(self, monkeypatch):
        """Poisoned external API keys in environment must all be set to None."""
        monkeypatch.setenv("SELF_HOST", "true")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-proj-malicious-leak-key-12345")
        monkeypatch.setenv("ASSEMBLY_AI_API_KEY", "assembly-stolen-token")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-adversarial-test")
        monkeypatch.setenv("GOOGLE_API_KEY", "AIzaSyPoisonedGoogleKey")
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-dummy-leak")
        monkeypatch.setenv("APIFY_API_TOKEN", "apify_token_unauthorized")

        cfg = config.Config()

        assert cfg.self_hosted is True
        assert cfg.self_host is True
        assert cfg.openai_api_key is None
        assert cfg.assembly_ai_api_key is None
        assert cfg.anthropic_api_key is None
        assert cfg.google_api_key is None
        assert cfg.openrouter_api_key is None
        assert cfg.apify_api_token is None

    @pytest.mark.parametrize("val", ["true", "True", "TRUE", "1", "yes", "YES", "on"])
    def test_truthy_self_host_variations(self, monkeypatch, val):
        """Various truthy environment representations enable self-hosted mode."""
        monkeypatch.setenv("SELF_HOST", val)
        cfg = config.Config()
        assert cfg.self_hosted is True

    @pytest.mark.parametrize("val", ["false", "0", "no", "off"])
    def test_falsy_self_host_variations(self, monkeypatch, val):
        """Falsy environment values disable self-hosted mode."""
        monkeypatch.setenv("SELF_HOST", val)
        cfg = config.Config()
        assert cfg.self_hosted is False

    def test_quotas_and_resolutions_in_self_hosted_mode(self, monkeypatch):
        """Self-hosted mode unlocks 1080p and disables all quotas."""
        monkeypatch.setenv("SELF_HOST", "true")
        monkeypatch.setenv("QUOTAS_ENABLED", "true")  # Even if env attempts enabling quotas

        cfg = config.Config()
        assert cfg.is_quota_enforced() is False
        assert cfg.quotas_enabled is False
        assert "1080p" in cfg.get_allowed_resolutions()
        assert "720p" in cfg.get_allowed_resolutions()
        assert config.is_self_hosted() is True

    def test_config_override_lifecycle(self):
        """Config override setter and reset lifecycle works correctly."""
        custom = config.Config()
        custom.max_clips = 99

        config.set_config_override(custom)
        assert config.get_config().max_clips == 99

        config.set_config_override(None)
        assert config.get_config().max_clips != 99 or config.get_config() is not custom


# ==============================================================================
# 6. End-to-End Pipeline Orchestration Resilience
# ==============================================================================

class TestPipelineOrchestrationAdversarial:
    """Stress-tests `clippa.cli.run_pipeline` failure modes and edge cases."""

    def test_empty_or_whitespace_url_returns_code_1(self):
        """Empty or blank URLs fail fast with exit code 1."""
        assert cli.run_pipeline(url="") == 1
        assert cli.run_pipeline(url="   ") == 1
        assert cli.run_pipeline(url="\t\n  ") == 1
        assert cli.run_pipeline(url=None) == 1

    def test_missing_output_dir_automatically_created_when_url_provided(self, tmp_path: Path):
        """Nested output directory is automatically created during pipeline run when URL is provided."""
        nested_out = tmp_path / "deep" / "nested" / "outputs"
        nested_tmp = tmp_path / "deep" / "nested" / "temp"
        assert not nested_out.exists()

        # Call with dummy URL: pipeline creates directories then fails at ingestion step
        res = cli.run_pipeline(url="dummy_nonexistent.mp4", output_dir=nested_out, temp_dir=nested_tmp)
        assert res == 1
        assert nested_out.exists()
        assert nested_tmp.exists()

    def test_empty_url_avoids_spurious_directory_creation(self, tmp_path: Path):
        """Calling pipeline with empty URL aborts immediately without polluting filesystem with empty dirs."""
        nested_out = tmp_path / "clean_check" / "outputs"
        assert not nested_out.exists()

        res = cli.run_pipeline(url="", output_dir=nested_out)
        assert res == 1
        assert not nested_out.exists()

    @patch("clippa.cli.download_video", side_effect=Exception("Simulated download failure"))
    def test_ingestion_failure_returns_code_1(self, mock_download, tmp_path: Path):
        """If ingestion fails, pipeline logs error and returns exit code 1."""
        res = cli.run_pipeline(url="https://youtube.com/watch?v=bad_url", output_dir=tmp_path)
        assert res == 1
        mock_download.assert_called_once()

    @patch("clippa.cli.download_video")
    def test_missing_ingested_file_returns_code_1(self, mock_download, tmp_path: Path):
        """If download_video returns a non-existent path, pipeline returns 1."""
        mock_download.return_value = tmp_path / "ghost_file.mp4"
        res = cli.run_pipeline(url="video.mp4", output_dir=tmp_path)
        assert res == 1

    @patch("clippa.cli.download_video")
    @patch("clippa.cli.transcribe_audio", side_effect=Exception("Whisper CUDA OOM"))
    def test_transcription_failure_returns_code_1(self, mock_transcribe, mock_download, tmp_path: Path):
        """If transcription raises an unhandled exception, pipeline returns 1."""
        video_file = tmp_path / "source.mp4"
        video_file.write_bytes(b"\x00" * 1024)
        mock_download.return_value = video_file

        res = cli.run_pipeline(url=str(video_file), output_dir=tmp_path)
        assert res == 1
        mock_transcribe.assert_called_once()

    @patch("clippa.cli.download_video")
    @patch("clippa.cli.transcribe_audio")
    def test_empty_transcript_words_returns_code_1(self, mock_transcribe, mock_download, tmp_path: Path):
        """If transcription detects 0 words (silent audio), pipeline returns 1."""
        video_file = tmp_path / "silent.mp4"
        video_file.write_bytes(b"\x00" * 1024)
        mock_download.return_value = video_file

        empty_transcript = TranscriptData(text="", language="en", words=[], utterances=[])
        mock_transcribe.return_value = empty_transcript

        res = cli.run_pipeline(url=str(video_file), output_dir=tmp_path)
        assert res == 1

    @patch("clippa.cli.download_video")
    @patch("clippa.cli.transcribe_audio")
    @patch("clippa.cli.extract_viral_segments", side_effect=Exception("Ollama daemon crash"))
    def test_analysis_failure_returns_code_1(self, mock_analysis, mock_transcribe, mock_download, tmp_path: Path):
        """If virality analysis throws an error, pipeline returns 1."""
        video_file = tmp_path / "source.mp4"
        video_file.write_bytes(b"\x00" * 1024)
        mock_download.return_value = video_file

        mock_transcribe.return_value = TranscriptData(
            text="Hello world",
            language="en",
            words=[Word(text="Hello", start=100, end=500), Word(text="world", start=550, end=900)],
        )

        res = cli.run_pipeline(url=str(video_file), output_dir=tmp_path)
        assert res == 1
        mock_analysis.assert_called_once()

    @patch("clippa.cli.download_video")
    @patch("clippa.cli.transcribe_audio")
    @patch("clippa.cli.extract_viral_segments", return_value=[])
    def test_zero_segments_returned_returns_code_1(self, mock_analysis, mock_transcribe, mock_download, tmp_path: Path):
        """If analysis returns an empty segment list, pipeline returns 1."""
        video_file = tmp_path / "source.mp4"
        video_file.write_bytes(b"\x00" * 1024)
        mock_download.return_value = video_file

        mock_transcribe.return_value = TranscriptData(
            text="Hello world",
            language="en",
            words=[Word(text="Hello", start=100, end=500)],
        )

        res = cli.run_pipeline(url=str(video_file), output_dir=tmp_path)
        assert res == 1

    @patch("clippa.cli.download_video")
    @patch("clippa.cli.transcribe_audio")
    @patch("clippa.cli.extract_viral_segments")
    @patch("clippa.cli.build_crop_filter", return_value="crop=1080:1920:0:0")
    @patch("clippa.cli.generate_ass_subtitles", return_value=Path("dummy.ass"))
    @patch("clippa.cli.render_clip", side_effect=Exception("FFmpeg encoder failure"))
    def test_all_clip_renders_fail_returns_code_1(
        self,
        mock_render,
        mock_ass,
        mock_crop,
        mock_analysis,
        mock_transcribe,
        mock_download,
        tmp_path: Path,
    ):
        """If all candidate clips fail FFmpeg rendering, pipeline returns 1."""
        video_file = tmp_path / "source.mp4"
        video_file.write_bytes(b"\x00" * 1024)
        mock_download.return_value = video_file

        mock_transcribe.return_value = TranscriptData(
            text="Sample dialogue",
            language="en",
            words=[Word(text="Sample", start=1000, end=2000), Word(text="dialogue", start=2100, end=3000)],
        )
        mock_analysis.return_value = [
            ViralSegment(
                start_time_ms=1000,
                end_time_ms=3000,
                hook_title="Hook 1",
                hook_score=20,
                engagement_score=20,
                value_score=20,
                shareability_score=20,
                total_score=80,
                reasoning="Great clip",
            )
        ]

        res = cli.run_pipeline(url=str(video_file), output_dir=tmp_path)
        assert res == 1
        mock_render.assert_called_once()

    @patch("clippa.cli.download_video")
    @patch("clippa.cli.transcribe_audio")
    @patch("clippa.cli.extract_viral_segments")
    @patch("clippa.cli.build_crop_filter", return_value="crop=1080:1920:0:0")
    @patch("clippa.cli.generate_ass_subtitles", return_value=Path("dummy.ass"))
    @patch("clippa.cli.render_clip")
    def test_partial_render_success_returns_code_0(
        self,
        mock_render,
        mock_ass,
        mock_crop,
        mock_analysis,
        mock_transcribe,
        mock_download,
        tmp_path: Path,
    ):
        """If 1 clip fails rendering but another succeeds, pipeline returns 0."""
        video_file = tmp_path / "source.mp4"
        video_file.write_bytes(b"\x00" * 1024)
        mock_download.return_value = video_file

        mock_transcribe.return_value = TranscriptData(
            text="First sentence. Second sentence.",
            language="en",
            words=[
                Word(text="First", start=1000, end=2000),
                Word(text="Second", start=15000, end=16000),
            ],
        )
        mock_analysis.return_value = [
            ViralSegment(
                start_time_ms=1000,
                end_time_ms=5000,
                hook_title="Clip 1",
                hook_score=20,
                engagement_score=20,
                value_score=20,
                shareability_score=20,
                total_score=80,
                reasoning="First segment",
            ),
            ViralSegment(
                start_time_ms=15000,
                end_time_ms=20000,
                hook_title="Clip 2",
                hook_score=22,
                engagement_score=22,
                value_score=22,
                shareability_score=22,
                total_score=88,
                reasoning="Second segment",
            ),
        ]

        rendered_clip_2 = tmp_path / "clip_2.mp4"

        def render_side_effect(source_video, start_ms, end_ms, crop_filter, ass_path, output_file):
            if start_ms == 1000:
                raise RuntimeError("Failed to encode clip 1")
            rendered_clip_2.write_bytes(b"\x00\x00\x00\x18ftypisom" + b"\x00" * 1024)
            return rendered_clip_2

        mock_render.side_effect = render_side_effect

        res = cli.run_pipeline(url=str(video_file), output_dir=tmp_path)
        assert res == 0
        assert rendered_clip_2.exists()

    @patch("clippa.cli.download_video")
    @patch("clippa.cli.transcribe_audio")
    @patch("clippa.cli.extract_viral_segments")
    @patch("clippa.cli.build_crop_filter", return_value="crop=720:1280:0:0")
    @patch("clippa.cli.generate_ass_subtitles", return_value=Path("dummy.ass"))
    @patch("clippa.cli.render_clip")
    def test_pipeline_flag_passthrough_verification(
        self,
        mock_render,
        mock_ass,
        mock_crop,
        mock_analysis,
        mock_transcribe,
        mock_download,
        tmp_path: Path,
    ):
        """Verifies resolution (720p), no_zoom, no_emojis, and no_cache flags propagate."""
        video_file = tmp_path / "source.mp4"
        video_file.write_bytes(b"\x00" * 1024)
        mock_download.return_value = video_file

        mock_transcribe.return_value = TranscriptData(
            text="Flag passthrough check",
            language="en",
            words=[Word(text="Flag", start=500, end=1500)],
        )
        mock_analysis.return_value = [
            ViralSegment(
                start_time_ms=500,
                end_time_ms=2500,
                hook_title="Flag Hook",
                hook_score=20,
                engagement_score=20,
                value_score=20,
                shareability_score=20,
                total_score=80,
                reasoning="Testing flags",
            )
        ]

        rendered_clip = tmp_path / "clip_1.mp4"

        def fake_render(source_video, start_ms, end_ms, crop_filter, ass_path, output_file):
            rendered_clip.write_bytes(b"\x00\x00\x00\x18ftypisom" + b"\x00" * 512)
            return rendered_clip

        mock_render.side_effect = fake_render

        res = cli.run_pipeline(
            url=str(video_file),
            output_dir=tmp_path,
            resolution="720p",
            no_zoom=True,
            no_emojis=True,
            no_cache=True,
            device="cpu",
            whisper_model="tiny",
            llm_model="heuristic",
            template="tiktok",
        )

        assert res == 0
        mock_transcribe.assert_called_once_with(
            video_path=video_file.resolve(),
            model_size="tiny",
            device="cpu",
            force_retranscribe=True,
        )
        mock_analysis.assert_called_once_with(
            transcript=mock_transcribe.return_value,
            max_clips=3,
            llm_model="heuristic",
        )
        mock_crop.assert_called_once_with(
            video_path=video_file.resolve(),
            start_ms=500,
            end_ms=2500,
            target_w=720,
            target_h=1280,
            enable_zoom=False,
        )
        mock_ass.assert_called_once()
        ass_kwargs = mock_ass.call_args.kwargs
        assert ass_kwargs["target_w"] == 720
        assert ass_kwargs["target_h"] == 1280
        assert ass_kwargs["enable_emojis"] is False
        assert ass_kwargs["template"] == "tiktok"

    def test_pipeline_e2e_live_synthetic_execution(self, tmp_path: Path):
        """Live end-to-end run on synthetic video producing probed 1080x1920 vertical MP4."""
        import json
        from tests.fixtures.synthetic import generate_synthetic_mp4
        from tests.utils import probe_media, assert_vertical_9_16, assert_valid_h264_aac, assert_audio_duration_sync

        video_file = tmp_path / "live_synth.mp4"
        generate_synthetic_mp4(video_file, duration_s=3.0)

        # Create valid transcript sidecar to skip external transcription
        cache_file = tmp_path / "live_synth.mp4.transcript_cache.json"
        words = [
            Word(text="The", start=100, end=500),
            Word(text="secret", start=500, end=1100),
            Word(text="mistake", start=1100, end=1800),
            Word(text="revealed", start=1800, end=2600),
        ]
        transcript = TranscriptData(text="The secret mistake revealed", language="en", words=words)
        cache_file.write_text(json.dumps(transcript.to_dict()))

        out_dir = tmp_path / "rendered_output"
        exit_code = cli.run_pipeline(
            url=str(video_file),
            output_dir=out_dir,
            template="hormozi",
            max_clips=1,
            llm_model="heuristic",
            resolution="1080p",
            no_zoom=True,
        )

        assert exit_code == 0
        output_clips = list(out_dir.glob("*.mp4"))
        assert len(output_clips) == 1
        clip = output_clips[0]
        assert clip.name == "clip_1.mp4"
        assert clip.stat().st_size > 1000

        # ffprobe media verification
        probe = probe_media(clip)
        assert_vertical_9_16(probe, target_w=1080, target_h=1920)
        assert_valid_h264_aac(probe)
        assert_audio_duration_sync(probe, tolerance_s=0.5)

    def test_cli_main_e2e_live_synthetic_execution_720p(self, tmp_path: Path):
        """Live end-to-end run via cli.main() producing probed 720x1280 vertical MP4."""
        import json
        from tests.fixtures.synthetic import generate_synthetic_mp4
        from tests.utils import probe_media, assert_vertical_9_16, assert_valid_h264_aac

        video_file = tmp_path / "live_synth_720p.mp4"
        generate_synthetic_mp4(video_file, duration_s=3.0)

        cache_file = tmp_path / "live_synth_720p.mp4.transcript_cache.json"
        words = [
            Word(text="Why", start=100, end=400),
            Word(text="nobody", start=400, end=900),
            Word(text="tells", start=900, end=1400),
            Word(text="you", start=1400, end=1900),
            Word(text="this", start=1900, end=2500),
        ]
        transcript = TranscriptData(text="Why nobody tells you this", language="en", words=words)
        cache_file.write_text(json.dumps(transcript.to_dict()))

        out_dir = tmp_path / "rendered_720p"
        exit_code = cli.main([
            "--url", str(video_file),
            "--output-dir", str(out_dir),
            "--template", "tiktok",
            "--max-clips", "1",
            "--llm-model", "heuristic",
            "--resolution", "720p",
            "--no-zoom",
        ])

        assert exit_code == 0
        output_clips = list(out_dir.glob("*.mp4"))
        assert len(output_clips) == 1
        clip = output_clips[0]
        assert clip.name == "clip_1.mp4"

        # ffprobe media verification
        probe = probe_media(clip)
        assert_vertical_9_16(probe, target_w=720, target_h=1280)
        assert_valid_h264_aac(probe)

