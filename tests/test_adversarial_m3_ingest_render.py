"""
tests/test_adversarial_m3_ingest_render.py
Empirical Adversarial Stress-Test Suite for Milestone 3:
- clippa/ingestion.py (Feature 1: Download & Local Bypass, Feature 2: Preflight & Cache)
- clippa/rendering.py (Feature 19: Subtitle Burning, Single-Pass 9:16 Crop, Windows Path Escaping)

Author: Challenger 1 (Empirical Challenger)
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from clippa.ingestion import (
    VideoMetadata,
    canonical_youtube_url,
    clear_metadata_cache,
    download_video,
    extract_video_id,
    get_video_metadata,
    is_local_video,
    is_valid_youtube_url,
    resolve_local_video_path,
    validate_video_duration,
    InvalidURLError,
    PreflightTimeoutError,
    StreamDurationUnavailableError,
    VideoAgeRestrictedError,
    VideoDownloadError,
    VideoDurationExceededError,
    VideoUnavailableError,
)
from clippa.rendering import (
    build_ffmpeg_render_args,
    escape_filter_path,
    escape_path_for_ffmpeg,
    find_ffmpeg,
    render_clip,
    FFmpegExecutionError,
    MediaNotFoundError,
    RenderingError,
    SubtitleBurnError,
)
from tests.fixtures.synthetic import generate_synthetic_mp4
from tests.utils import (
    assert_audio_duration_sync,
    assert_burned_subtitles_present,
    assert_media_integrity,
    assert_valid_h264_aac,
    assert_vertical_9_16,
    probe_media,
    require_ffmpeg,
    require_ffprobe,
)


# ==============================================================================
# Helper Fixtures
# ==============================================================================

@pytest.fixture(autouse=True)
def clean_cache():
    """Guarantees preflight metadata cache is clean before and after each test."""
    clear_metadata_cache()
    yield
    clear_metadata_cache()


@pytest.fixture
def sample_ass_content() -> str:
    """Returns valid ASS subtitle content with 1080x1920 play resolution."""
    return """[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,64,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,1,0,0,0,100,100,0,0,1,3,0,2,20,20,80,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:00.00,0:00:02.00,Default,,0,0,0,,{\\k20}Empirical {\\k20}Stress {\\k20}Test
"""


# ==============================================================================
# 1. Ingestion: Whitespace, Empty, and Malformed URLs
# ==============================================================================

class TestAdversarialIngestionURLs:
    """Adversarial stress-testing of URL validation and parsing in clippa.ingestion."""

    @pytest.mark.parametrize("bad_url", [
        None,
        "",
        "   ",
        "\t\r\n   ",
        "   \v\f   ",
    ])
    def test_adv_ingest_empty_and_whitespace_urls(self, bad_url: str | None, tmp_path: Path):
        """Empty, None, or whitespace-only URLs immediately raise InvalidURLError."""
        with pytest.raises(InvalidURLError):
            download_video(bad_url, output_dir=tmp_path)

        if bad_url is not None:
            with pytest.raises(InvalidURLError):
                get_video_metadata(bad_url)

    @pytest.mark.parametrize("malformed_url", [
        "https://",
        "https://www.youtube.com",
        "https://www.youtube.com/",
        "https://www.youtube.com/watch",
        "https://www.youtube.com/watch?v=",
        "https://www.youtube.com/watch?v=  ",
        "https://youtu.be/",
        "https://youtu.be",
        "https://www.youtube.com/embed/",
        "https://www.youtube.com/shorts/",
        "not_a_valid_url_or_file",
        "http:///invalid",
    ])
    def test_adv_ingest_malformed_urls_without_id(self, malformed_url: str, tmp_path: Path):
        """Malformed URLs lacking an extractable video ID fail immediately without network call."""
        assert extract_video_id(malformed_url) is None
        assert not is_valid_youtube_url(malformed_url)
        assert canonical_youtube_url(malformed_url) is None

        with pytest.raises(InvalidURLError):
            download_video(malformed_url, output_dir=tmp_path)

        with pytest.raises(InvalidURLError):
            get_video_metadata(malformed_url)

    @pytest.mark.parametrize("spoofed_url", [
        "https://vimeo.com/123456789",
        "https://evil-youtube.com/watch?v=dQw4w9WgXcQ",
        "https://youtube.com.attacker.site/watch?v=dQw4w9WgXcQ",
        "https://notyoutube.org/watch?v=dQw4w9WgXcQ",
        "https://dailymotion.com/video/x12345",
    ])
    def test_adv_ingest_non_youtube_domains_rejected(self, spoofed_url: str, tmp_path: Path):
        """Non-YouTube domains and spoofed subdomains must be rejected cleanly."""
        assert extract_video_id(spoofed_url) is None
        with pytest.raises(InvalidURLError):
            download_video(spoofed_url, output_dir=tmp_path)

    def test_adv_ingest_complex_query_parameters(self):
        """Extracts valid YouTube ID regardless of extraneous query parameters and order."""
        test_cases = [
            ("https://www.youtube.com/watch?feature=youtu.be&list=PL123&v=dQw4w9WgXcQ&index=3&t=42s", "dQw4w9WgXcQ"),
            ("https://youtu.be/dQw4w9WgXcQ?si=abcdef123456&t=10s", "dQw4w9WgXcQ"),
            ("https://m.youtube.com/watch?app=desktop&v=synthetic_id_01", "synthetic_id_01"),
            ("https://www.youtube-nocookie.com/embed/embed_id_999?autoplay=1&controls=0", "embed_id_999"),
            ("https://www.youtube.com/shorts/viral_short_123?feature=share", "viral_short_123"),
            ("https://music.youtube.com/watch?v=music_track_77", "music_track_77"),
        ]
        for url, expected_id in test_cases:
            vid_id = extract_video_id(url)
            assert vid_id == expected_id, f"Failed on URL '{url}': expected '{expected_id}', got '{vid_id}'"
            assert is_valid_youtube_url(url)
            assert canonical_youtube_url(url) == f"https://www.youtube.com/watch?v={expected_id}"

    def test_adv_ingest_case_insensitivity_and_synthetic_ids(self):
        """Extracts synthetic and alphanumeric IDs with case insensitivity in domain and scheme."""
        url = "HTTPS://WWW.YOUTUBE.COM/watch?v=CUSTOM_MOCK_VID_99"
        assert extract_video_id(url) == "CUSTOM_MOCK_VID_99"

        url_short = "HTTPS://YOUTU.BE/CUSTOM-ID-123"
        assert extract_video_id(url_short) == "CUSTOM-ID-123"

        url_scheme_less = "youtube.com/watch?v=scheme_less_id_42"
        assert extract_video_id(url_scheme_less) == "scheme_less_id_42"

    @pytest.mark.parametrize("invalid_id_url", [
        "https://www.youtube.com/watch?v=ab",           # too short (<3 chars)
        "https://www.youtube.com/watch?v=abc;rm -rf /",  # command injection attempt
        "https://www.youtube.com/watch?v=id with spaces",
        "https://www.youtube.com/watch?v=<script>alert(1)</script>",
    ])
    def test_adv_ingest_malformed_chars_in_video_id(self, invalid_id_url: str):
        """Video IDs with invalid characters or below minimum length are rejected."""
        assert extract_video_id(invalid_id_url) is None
        assert not is_valid_youtube_url(invalid_id_url)


# ==============================================================================
# 2. Ingestion: Local File Bypass & URI Resolution
# ==============================================================================

class TestAdversarialLocalFileBypass:
    """Stress-testing the local file bypass and file:// URI handling."""

    def test_adv_local_file_path_bypass_complete(self, tmp_path: Path):
        """Local file paths bypass yt-dlp completely without initializing YoutubeDL."""
        local_file = generate_synthetic_mp4(tmp_path / "existing_local.mp4", duration_s=1.0)
        assert local_file.is_file()

        with patch("yt_dlp.YoutubeDL") as mock_ydl_cls:
            # 1. Path object
            p1 = download_video(local_file, output_dir=tmp_path / "out")
            assert p1 == local_file.resolve()
            mock_ydl_cls.assert_not_called()

            # 2. String path
            p2 = download_video(str(local_file), output_dir=tmp_path / "out")
            assert p2 == local_file.resolve()
            mock_ydl_cls.assert_not_called()

            # 3. Relative path string
            cwd = Path.cwd()
            try:
                rel_path = os.path.relpath(local_file, cwd)
                p3 = download_video(rel_path, output_dir=tmp_path / "out")
                assert p3 == local_file.resolve()
                mock_ydl_cls.assert_not_called()
            except ValueError:
                pass

            # 4. get_video_metadata on local file
            meta = get_video_metadata(str(local_file))
            assert meta.id == local_file.stem
            assert meta.uploader == "Local"
            mock_ydl_cls.assert_not_called()

    def test_adv_file_uri_bypass(self, tmp_path: Path):
        """file:// URIs resolve to absolute local paths and bypass yt-dlp."""
        local_file = generate_synthetic_mp4(tmp_path / "uri_test_video.mp4", duration_s=1.0)
        file_uri = local_file.as_uri()  # e.g. file:///C:/...

        assert is_local_video(file_uri)
        resolved = resolve_local_video_path(file_uri)
        assert resolved == local_file.resolve()

        with patch("yt_dlp.YoutubeDL") as mock_ydl_cls:
            res = download_video(file_uri, output_dir=tmp_path)
            assert res == local_file.resolve()
            mock_ydl_cls.assert_not_called()

    def test_adv_nonexistent_local_file_rejected(self, tmp_path: Path):
        """Non-existent local file paths must not bypass and must raise InvalidURLError."""
        bogus_path = tmp_path / "non_existent_clip_12345.mp4"
        assert not bogus_path.exists()
        assert not is_local_video(bogus_path)
        assert resolve_local_video_path(bogus_path) is None

        with pytest.raises(InvalidURLError):
            download_video(bogus_path, output_dir=tmp_path)


# ==============================================================================
# 3. Ingestion: VideoMetadata Model Dual Access & Dict Interface
# ==============================================================================

class TestAdversarialVideoMetadataModel:
    """Stress-testing VideoMetadata dataclass with dual dict/attribute access."""

    def test_adv_metadata_dual_access_and_keyerror(self):
        """Asserts dict subscript access, attribute access, and KeyError behavior."""
        meta = VideoMetadata(
            id="vid_123",
            title="Adversarial Title",
            duration=120.5,
            uploader="Test Channel",
            is_live=False,
        )

        # Attribute access
        assert meta.id == "vid_123"
        assert meta.title == "Adversarial Title"
        assert meta.duration == 120.5

        # Dict subscript get
        assert meta["id"] == "vid_123"
        assert meta["title"] == "Adversarial Title"
        assert meta["duration"] == 120.5
        assert meta["uploader"] == "Test Channel"
        assert meta["is_live"] is False

        # Dict subscript mutation
        meta["title"] = "Updated Title"
        assert meta.title == "Updated Title"
        assert meta["title"] == "Updated Title"

        # Attribute mutation
        meta.duration = 180.0
        assert meta["duration"] == 180.0

        # Subscript KeyError on missing attribute
        with pytest.raises(KeyError):
            _ = meta["non_existent_key_xyz"]

    def test_adv_metadata_in_operator_and_get(self):
        """Asserts 'in' operator, .get() with defaults, and to_dict/from_dict roundtrip."""
        meta = VideoMetadata(
            id="vid_abc",
            title="Sample",
            duration=60.0,
        )

        # 'in' operator
        assert "title" in meta
        assert "duration" in meta
        assert "id" in meta
        assert "nonexistent_field" not in meta

        # .get()
        assert meta.get("title") == "Sample"
        assert meta.get("nonexistent_field") is None
        assert meta.get("nonexistent_field", "fallback_val") == "fallback_val"

        # to_dict roundtrip
        d = meta.to_dict()
        assert isinstance(d, dict)
        assert d["id"] == "vid_abc"
        assert d["duration"] == 60.0

        reconstructed = VideoMetadata.from_dict(d)
        assert reconstructed.id == meta.id
        assert reconstructed.title == meta.title
        assert reconstructed.duration == meta.duration

    def test_adv_metadata_from_dict_defaults_and_missing_keys(self):
        """VideoMetadata.from_dict provides safe defaults when input dict is sparse."""
        sparse = {"id": "sparse_1"}
        m = VideoMetadata.from_dict(sparse)
        assert m.id == "sparse_1"
        assert m.title == ""
        assert m.duration == 0.0
        assert m.uploader == "Unknown"
        assert m.is_live is False
        assert m.ext == "mp4"

    def test_adv_metadata_to_dict_roundtrip(self):
        """Asserts to_dict and from_dict roundtrip preserves all optional and required fields."""
        full_dict = {
            "id": "vid_full",
            "title": "Comprehensive Title",
            "duration": 250.5,
            "uploader": "Test Channel",
            "is_live": False,
            "description": "Full description with special characters: & < > ' \"",
            "view_count": 1000000,
            "like_count": 50000,
            "thumbnail": "https://i.ytimg.com/vi/vid_full/hqdefault.jpg",
            "width": 1920,
            "height": 1080,
            "fps": 60.0,
            "ext": "mp4",
        }
        meta = VideoMetadata.from_dict(full_dict)
        assert meta.view_count == 1000000
        assert meta.fps == 60.0
        assert meta.thumbnail.startswith("https://")

        roundtrip_dict = meta.to_dict()
        assert roundtrip_dict["id"] == "vid_full"
        assert roundtrip_dict["view_count"] == 1000000
        assert roundtrip_dict["like_count"] == 50000
        assert roundtrip_dict["description"] == full_dict["description"]


# ==============================================================================
# 4. Ingestion: Duration Limits & Livestream Rejection
# ==============================================================================

class TestAdversarialDurationAndLivestream:
    """Stress-testing duration boundaries and livestream detection."""

    def test_adv_validate_video_duration_boundaries(self):
        """Validates exact duration boundaries (0, negative, max, over max)."""
        # Exactly at max limit: passes
        assert validate_video_duration(5400.0, max_duration=5400.0, raise_on_error=True) is True

        # Just over limit: fails
        assert validate_video_duration(5400.1, max_duration=5400.0, raise_on_error=False) is False
        with pytest.raises(VideoDurationExceededError):
            validate_video_duration(5400.1, max_duration=5400.0, raise_on_error=True)

        # Zero duration: fails
        assert validate_video_duration(0.0, raise_on_error=False) is False
        with pytest.raises(StreamDurationUnavailableError):
            validate_video_duration(0.0, raise_on_error=True)

        # Negative duration: fails
        assert validate_video_duration(-15.0, raise_on_error=False) is False
        with pytest.raises(StreamDurationUnavailableError):
            validate_video_duration(-15.0, raise_on_error=True)

    def test_adv_validate_video_duration_type_polymorphism(self):
        """Accepts int, float, dict, and VideoMetadata instances."""
        meta = VideoMetadata(id="1", title="T", duration=300.0)
        assert validate_video_duration(meta) is True

        meta_too_long = VideoMetadata(id="2", title="T", duration=6000.0)
        assert validate_video_duration(meta_too_long, max_duration=5400.0) is False

        d = {"duration": 150}
        assert validate_video_duration(d) is True

    def test_adv_livestream_detection_rejection(self, tmp_path: Path):
        """Live streams are rejected with StreamDurationUnavailableError."""
        mock_ydl = MagicMock()
        mock_ydl.extract_info.return_value = {
            "id": "live_stream_id",
            "title": "Live 24/7 Stream",
            "is_live": True,
            "duration": None,
        }

        with patch("yt_dlp.YoutubeDL", return_value=mock_ydl):
            with pytest.raises(StreamDurationUnavailableError):
                download_video("https://www.youtube.com/watch?v=live_stream_id", output_dir=tmp_path)

    def test_adv_duration_exceeded_rejection(self, tmp_path: Path):
        """Videos exceeding max_duration are rejected with VideoDurationExceededError."""
        mock_ydl = MagicMock()
        mock_ydl.extract_info.return_value = {
            "id": "too_long_vid",
            "title": "10 Hour Ambient Noise",
            "is_live": False,
            "duration": 36000.0,
            "_filename": str(tmp_path / "sample.mp4"),
        }

        with patch("yt_dlp.YoutubeDL", return_value=mock_ydl):
            with pytest.raises(VideoDurationExceededError):
                download_video("https://www.youtube.com/watch?v=too_long_vid", output_dir=tmp_path, max_duration=5400.0)


# ==============================================================================
# 5. Ingestion: Preflight Metadata Cache Lifecycle
# ==============================================================================

class TestAdversarialMetadataPreflightCache:
    """Stress-testing the in-memory preflight cache and clearing."""

    def test_adv_metadata_cache_lifecycle_and_clear(self):
        """Cache stores first query result and avoids redundant yt-dlp calls until cleared."""
        url = "https://www.youtube.com/watch?v=cached_vid_01"
        mock_ydl = MagicMock()
        mock_ydl.extract_info.return_value = {
            "id": "cached_vid_01",
            "title": "Cached Video",
            "duration": 45.0,
        }

        with patch("yt_dlp.YoutubeDL", return_value=mock_ydl):
            # First fetch: calls yt-dlp
            meta1 = get_video_metadata(url, use_cache=True)
            assert meta1.id == "cached_vid_01"
            assert mock_ydl.extract_info.call_count == 1

            # Second fetch: served from cache, no new yt-dlp call
            meta2 = get_video_metadata(url, use_cache=True)
            assert meta2.id == "cached_vid_01"
            assert mock_ydl.extract_info.call_count == 1
            assert meta1 is meta2

            # Clear cache: subsequent fetch must call yt-dlp again
            clear_metadata_cache()
            meta3 = get_video_metadata(url, use_cache=True)
            assert meta3.id == "cached_vid_01"
            assert mock_ydl.extract_info.call_count == 2

    def test_adv_metadata_cache_bypass_flag(self):
        """Passing use_cache=False bypasses cache lookup and storage."""
        url = "https://www.youtube.com/watch?v=nocache_vid_02"
        mock_ydl = MagicMock()
        mock_ydl.extract_info.return_value = {
            "id": "nocache_vid_02",
            "title": "No Cache",
            "duration": 30.0,
        }

        with patch("yt_dlp.YoutubeDL", return_value=mock_ydl):
            get_video_metadata(url, use_cache=False)
            assert mock_ydl.extract_info.call_count == 1

            get_video_metadata(url, use_cache=False)
            assert mock_ydl.extract_info.call_count == 2


# ==============================================================================
# 6. Rendering: Windows Path Escaping & Complex Filesystems
# ==============================================================================

class TestAdversarialRenderingWindowsPaths:
    """Stress-testing Windows path escaping and real FFmpeg rendering with complex paths."""

    def test_adv_render_escape_filter_path_logic(self):
        """Verifies escaping of Windows colons, path separators, brackets, and single quotes."""
        raw_path = "C:\\Users\\User\\Videos [Clips]\\Clip '1'.ass"
        escaped = escape_filter_path(raw_path)

        assert r"C\:" in escaped
        assert "/Users/User/Videos" in escaped
        assert r"\[Clips\]" in escaped
        assert r"\'1\'" in escaped
        assert escape_path_for_ffmpeg(raw_path) == escaped

    def test_adv_render_real_ffmpeg_complex_paths_spaces_brackets(self, tmp_path: Path, sample_ass_content: str):
        """EMPIRICAL TEST: FFmpeg successfully renders with real paths containing spaces, brackets, and quotes."""
        require_ffmpeg()
        require_ffprobe()

        # Complex folder with brackets and spaces
        complex_folder = tmp_path / "Folder [1] with spaces"
        complex_folder.mkdir(parents=True, exist_ok=True)

        source_video = generate_synthetic_mp4(complex_folder / "source [raw].mp4", duration_s=2.0)
        ass_file = complex_folder / "subtitles [karaoke].ass"
        ass_file.write_text(sample_ass_content, encoding="utf-8")

        rendered_output = complex_folder / "output [9x16].mp4"

        # Execute real rendering
        result = render_clip(
            source_video=source_video,
            start_ms=0,
            end_ms=2000,
            crop_filter="crop=608:1080:656:0,scale=1080:1920",
            ass_path=ass_file,
            output_file=rendered_output,
        )

        assert result.exists()
        assert result.stat().st_size > 0

        # ffprobe assertion
        probe = probe_media(result)
        assert_vertical_9_16(probe, target_w=1080, target_h=1920)
        assert_valid_h264_aac(probe)


# ==============================================================================
# 7. Rendering: Timestamp Normalization (setpts=PTS-STARTPTS)
# ==============================================================================

class TestAdversarialRenderingTimestampNormalization:
    """Verifies presentation timestamp normalization prevents subtitle/video desync."""

    def test_adv_render_timestamp_normalization_probe(self, tmp_path: Path, sample_ass_content: str):
        """EMPIRICAL TEST: Trimming a clip resets presentation timestamp to 0.0s."""
        require_ffmpeg()
        require_ffprobe()

        # 5-second source video
        src = generate_synthetic_mp4(tmp_path / "long_source.mp4", duration_s=5.0)
        ass = tmp_path / "test.ass"
        ass.write_text(sample_ass_content, encoding="utf-8")
        out = tmp_path / "trimmed_clip.mp4"

        # Seek from 2000ms to 4500ms (duration 2.5s)
        render_clip(
            source_video=src,
            start_ms=2000,
            end_ms=4500,
            crop_filter="crop=608:1080:656:0,scale=1080:1920",
            ass_path=ass,
            output_file=out,
        )

        assert out.exists()
        probe = probe_media(out)

        # Check duration is ~2.5s
        v_stream = [s for s in probe["streams"] if s["codec_type"] == "video"][0]
        dur = float(v_stream.get("duration") or probe["format"]["duration"])
        assert abs(dur - 2.5) <= 0.2, f"Expected duration ~2.5s, got {dur}s"

        # Check start_pts or start_time is 0.0
        start_time = float(v_stream.get("start_time", 0.0))
        assert abs(start_time - 0.0) <= 0.05, f"Expected start_time 0.0s, got {start_time}s"

        # Audio and video durations remain synced
        assert_audio_duration_sync(probe, tolerance_s=0.3)


# ==============================================================================
# 8. Rendering: Atomic Writes and Failure Cleanup
# ==============================================================================

class TestAdversarialRenderingAtomicWritesAndCleanup:
    """Stress-testing atomic output file replacement and temporary file cleanup."""

    def test_adv_render_atomic_cleanup_on_ffmpeg_failure(self, tmp_path: Path):
        """Temporary .tmp.mp4 files are cleaned up on FFmpeg failure, leaving no corrupted files."""
        require_ffmpeg()

        src = generate_synthetic_mp4(tmp_path / "valid_src.mp4", duration_s=1.0)
        out = tmp_path / "failed_render.mp4"

        # Pass an invalid crop filter designed to cause FFmpeg filtergraph parse failure
        with pytest.raises((FFmpegExecutionError, RenderingError)):
            render_clip(
                source_video=src,
                start_ms=0,
                end_ms=1000,
                crop_filter="crop=99999:99999:0:0",
                output_file=out,
            )

        # Output target file must not exist
        assert not out.exists()

        # No .tmp.mp4 leftover in directory
        tmp_files = list(tmp_path.glob("*.tmp.mp4")) + list(tmp_path.glob(".*.tmp.mp4"))
        assert len(tmp_files) == 0, f"Found lingering temporary files: {tmp_files}"

    def test_adv_render_atomic_preserves_existing_target_on_failure(self, tmp_path: Path):
        """If rendering fails, an existing file at output_file is preserved intact."""
        require_ffmpeg()

        src = generate_synthetic_mp4(tmp_path / "valid_src2.mp4", duration_s=1.0)
        target_file = tmp_path / "existing_precious_file.mp4"
        sentinel_bytes = b"PREVIOUS_VALID_CONTENT_1234567890"
        target_file.write_bytes(sentinel_bytes)

        # Try to render with failing arguments targeting the existing file
        with pytest.raises((FFmpegExecutionError, RenderingError)):
            render_clip(
                source_video=src,
                start_ms=0,
                end_ms=1000,
                crop_filter="crop=99999:99999:0:0",
                output_file=target_file,
            )

        # Target file must still exist and contain the original bytes
        assert target_file.exists()
        assert target_file.read_bytes() == sentinel_bytes

    def test_adv_render_missing_input_files_raise_media_not_found(self, tmp_path: Path):
        """Missing source video or subtitle file raises MediaNotFoundError immediately."""
        bogus_video = tmp_path / "missing_video.mp4"
        bogus_ass = tmp_path / "missing_sub.ass"

        with pytest.raises(MediaNotFoundError):
            render_clip(bogus_video, start_ms=0, end_ms=1000, output_file=tmp_path / "out.mp4")

        valid_src = generate_synthetic_mp4(tmp_path / "valid_video.mp4", duration_s=1.0)
        with pytest.raises(MediaNotFoundError):
            render_clip(valid_src, start_ms=0, end_ms=1000, ass_path=bogus_ass, output_file=tmp_path / "out.mp4")


# ==============================================================================
# 9. Rendering: Single-Pass Combined Filter & Media Stream Integrity
# ==============================================================================

class TestAdversarialRenderingCombinedFilterAndIntegrity:
    """EMPIRICAL TEST: Comprehensive end-to-end verification of single-pass rendering."""

    def test_adv_render_single_pass_command_filter_order(self, tmp_path: Path):
        """Verifies filtergraph ordering: setpts first, crop second, ass subtitle third."""
        args = build_ffmpeg_render_args(
            source_video="in.mp4",
            output_file="out.mp4",
            start_ms=1000,
            end_ms=3000,
            crop_filter="crop=608:1080:656:0,scale=1080:1920",
            ass_path="sub.ass",
        )

        vf_index = args.index("-vf")
        filter_str = args[vf_index + 1]

        # Verify chaining in a single pass
        assert filter_str.startswith("setpts=PTS-STARTPTS")
        assert "crop=608:1080" in filter_str
        assert "scale=1080:1920" in filter_str
        assert "ass=filename=" in filter_str

        # Confirm single -vf argument (single-pass filtergraph)
        assert args.count("-vf") == 1

    def test_adv_render_ffprobe_comprehensive_validation_1080p(self, tmp_path: Path, sample_ass_content: str):
        """EMPIRICAL TEST: Full 1080x1920 vertical render verified via ffprobe, pixel variance, and decode check."""
        require_ffmpeg()
        require_ffprobe()

        src = generate_synthetic_mp4(tmp_path / "source_1080.mp4", duration_s=2.5, width=1920, height=1080)
        ass = tmp_path / "captions.ass"
        ass.write_text(sample_ass_content, encoding="utf-8")
        out = tmp_path / "final_vertical_1080.mp4"

        # Render clip
        render_clip(
            source_video=src,
            start_ms=0,
            end_ms=2500,
            crop_filter="crop=608:1080:656:0,scale=1080:1920",
            ass_path=ass,
            output_file=out,
        )

        assert out.exists()

        # 1. Probe resolution and aspect ratio (1080x1920)
        probe = probe_media(out)
        assert_vertical_9_16(probe, target_w=1080, target_h=1920)

        # 2. Probe video & audio codecs (H.264 + AAC + yuv420p)
        assert_valid_h264_aac(probe)

        # 3. Audio/video duration synchronization
        assert_audio_duration_sync(probe, tolerance_s=0.25)

        # 4. Burned subtitle pixel variance check
        assert_burned_subtitles_present(out, timestamp_s=1.0)

        # 5. Full media stream decode integrity (zero corrupted frames)
        assert_media_integrity(out)

    def test_adv_render_ffprobe_validation_720p(self, tmp_path: Path, sample_ass_content: str):
        """EMPIRICAL TEST: 720x1280 vertical render verified via ffprobe."""
        require_ffmpeg()
        require_ffprobe()

        src = generate_synthetic_mp4(tmp_path / "source_720.mp4", duration_s=2.0, width=1280, height=720)
        ass = tmp_path / "captions_720.ass"
        ass.write_text(sample_ass_content, encoding="utf-8")
        out = tmp_path / "final_vertical_720.mp4"

        render_clip(
            source_video=src,
            start_ms=0,
            end_ms=2000,
            crop_filter="crop=405:720:437:0,scale=720:1280",
            ass_path=ass,
            output_file=out,
        )

        assert out.exists()
        probe = probe_media(out)
        assert_vertical_9_16(probe, target_w=720, target_h=1280)
        assert_valid_h264_aac(probe)

    def test_adv_render_faststart_moov_atom_position(self, tmp_path: Path, sample_ass_content: str):
        """EMPIRICAL TEST: Moov atom is located before mdat atom in output MP4 container."""
        require_ffmpeg()

        src = generate_synthetic_mp4(tmp_path / "source_fs.mp4", duration_s=1.5)
        ass = tmp_path / "captions_fs.ass"
        ass.write_text(sample_ass_content, encoding="utf-8")
        out = tmp_path / "final_faststart.mp4"

        render_clip(
            source_video=src,
            start_ms=0,
            end_ms=1500,
            crop_filter="crop=608:1080:656:0,scale=1080:1920",
            ass_path=ass,
            output_file=out,
        )

        data = out.read_bytes()
        pos_moov = data.find(b"moov")
        pos_mdat = data.find(b"mdat")

        assert pos_moov > 0, "moov atom not found in MP4 header"
        assert pos_mdat > 0, "mdat atom not found in MP4 payload"
        assert pos_moov < pos_mdat, f"moov atom ({pos_moov}) must appear before mdat atom ({pos_mdat}) for faststart"
