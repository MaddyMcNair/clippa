"""
tests/test_adversarial_m4_pipeline.py - Tier 5 Adversarial Seam, Pipeline & Artifact Verification

Milestone 4: Final Milestone Acceptance & Adversarial Hardening
Adversarial Coverage:
1. Live Artifact Empirical Verification (outputs/clip_1.mp4, outputs/clip_1_preview.png)
2. Pipeline Concurrency & Unique File Naming (Race conditions, collisions, multi-threading)
3. Temp Folder Lifecycle, Isolation & Cleanup (Leak prevention, failure unlinks)
4. Zero-Paid-API Leakage Under Adversarial Environments (Poisoned keys, DNS blackhole)
5. CLI Argument Robustness & Boundary Recovery (Invalid flags, choices, type safety)
6. Pipeline Seam Resilience, Corrupt Inputs & Error Handling (Missing media, empty transcripts, ASS escaping)
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import patch, MagicMock

import cv2
import numpy as np
import pytest

from clippa.cli import (
    build_parser,
    parse_cli_args,
    get_unique_clip_path,
    run_pipeline,
    main,
)
from clippa.config import Config, get_config, set_config_override, is_self_hosted
from clippa.ingestion import VideoMetadata, VideoDownloadError, InvalidURLError
from clippa.rendering import (
    render_clip,
    build_ffmpeg_render_args,
    find_ffmpeg,
    escape_filter_path,
    RenderingError,
    FFmpegExecutionError,
    MediaNotFoundError,
)
from clippa.subtitles import generate_ass_subtitles
from clippa.transcription import Word, Utterance, TranscriptData
from clippa.analysis import ViralSegment
from tests.utils import (
    probe_media,
    assert_vertical_9_16,
    assert_valid_h264_aac,
    assert_audio_duration_sync,
    assert_non_silent_audio,
    assert_media_integrity,
    assert_burned_subtitles_present,
)
from tests.fixtures.synthetic import (
    generate_synthetic_mp4,
    generate_corrupt_media,
    make_sample_transcript,
    make_sample_viral_segment,
    make_sample_ass_content,
)


# ==============================================================================
# Helper Functions for ISO-BMFF and Media Assertions
# ==============================================================================

def parse_iso_bmff_atoms(file_path: Path, max_bytes: int = 200000) -> List[tuple[str, int, int]]:
    """Walks top-level ISO-BMFF / MP4 atoms and returns list of (tag, offset, size)."""
    atoms = []
    with open(file_path, "rb") as f:
        file_size = file_path.stat().st_size
        offset = 0
        while offset < min(file_size, max_bytes):
            header = f.read(8)
            if len(header) < 8:
                break
            size = int.from_bytes(header[:4], "big")
            tag = header[4:8].decode("latin1", errors="replace")
            atoms.append((tag, offset, size))
            if size == 1:
                # 64-bit size
                ext_hdr = f.read(8)
                if len(ext_hdr) < 8:
                    break
                size = int.from_bytes(ext_hdr, "big")
            elif size == 0:
                # atom extends to EOF
                break
            offset += size
            f.seek(offset)
    return atoms


# ==============================================================================
# Class 1: Live Artifact Empirical Verification
# ==============================================================================

class TestAdversarialLiveArtifactVerification:
    """Empirical verification of live artifacts: outputs/clip_1.mp4 and outputs/clip_1_preview.png."""

    LIVE_CLIP = Path("outputs/clip_1.mp4")
    LIVE_PREVIEW = Path("outputs/clip_1_preview.png")

    def test_adv_live_artifact_exists_and_size(self):
        """TC-ADV-M4-01: Verifies outputs/clip_1.mp4 exists and is > 1MB."""
        assert self.LIVE_CLIP.exists(), f"Live artifact not found: {self.LIVE_CLIP.resolve()}"
        file_size = self.LIVE_CLIP.stat().st_size
        assert file_size >= 1_000_000, f"Live clip is unexpectedly small: {file_size} bytes (expected >= 1MB)"

    def test_adv_live_artifact_ffmpeg_decode_integrity(self):
        """TC-ADV-M4-02: Decodes outputs/clip_1.mp4 entirely with FFmpeg null sink (0 corrupt packets)."""
        ffmpeg_bin = find_ffmpeg()
        assert ffmpeg_bin is not None, "FFmpeg binary is required"

        cmd = [ffmpeg_bin, "-v", "error", "-i", str(self.LIVE_CLIP), "-f", "null", "-"]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        assert proc.returncode == 0, f"FFmpeg decode failed on live artifact: {proc.stderr}"
        assert proc.stderr.strip() == "", f"FFmpeg reported stream corruption: {proc.stderr}"

    def test_adv_live_artifact_video_stream_conformance(self):
        """TC-ADV-M4-03: Probes video stream parameters (1080x1920, H.264, yuv420p, CFR)."""
        meta = probe_media(self.LIVE_CLIP)
        video_streams = [s for s in meta["streams"] if s.get("codec_type") == "video"]
        assert len(video_streams) == 1, f"Expected exactly 1 video stream, found {len(video_streams)}"

        v = video_streams[0]
        assert v.get("codec_name") == "h264", f"Video codec is not H.264: {v.get('codec_name')}"
        assert v.get("pix_fmt") == "yuv420p", f"Pixel format is not yuv420p: {v.get('pix_fmt')}"
        assert int(v.get("width", 0)) == 1080, f"Width is not 1080: {v.get('width')}"
        assert int(v.get("height", 0)) == 1920, f"Height is not 1920: {v.get('height')}"

        ratio = int(v["width"]) / int(v["height"])
        assert abs(ratio - (9.0 / 16.0)) < 0.001, f"Aspect ratio {ratio} != 9/16"

    def test_adv_live_artifact_audio_stream_conformance(self):
        """TC-ADV-M4-04: Probes audio stream parameters (AAC, stereo, 48kHz)."""
        meta = probe_media(self.LIVE_CLIP)
        audio_streams = [s for s in meta["streams"] if s.get("codec_type") == "audio"]
        assert len(audio_streams) == 1, f"Expected exactly 1 audio stream, found {len(audio_streams)}"

        a = audio_streams[0]
        assert a.get("codec_name") == "aac", f"Audio codec is not AAC: {a.get('codec_name')}"
        assert int(a.get("channels", 0)) == 2, f"Expected 2 channels (stereo), got {a.get('channels')}"
        assert int(a.get("sample_rate", 0)) == 48000, f"Sample rate is not 48kHz: {a.get('sample_rate')}"

    def test_adv_live_artifact_av_sync_and_audibility(self):
        """TC-ADV-M4-05: Checks A/V duration sync within 50ms and non-silent speech volume."""
        meta = probe_media(self.LIVE_CLIP)
        assert_audio_duration_sync(meta, tolerance_s=0.05)
        assert_non_silent_audio(self.LIVE_CLIP, min_mean_volume_db=-40.0)

    def test_adv_live_artifact_faststart_moov_atom_position(self):
        """TC-ADV-M4-06: Walks ISO-BMFF atoms, asserting 'ftyp' at 0 and 'moov' precedes 'mdat'."""
        atoms = parse_iso_bmff_atoms(self.LIVE_CLIP)
        atom_names = [a[0] for a in atoms]
        assert "ftyp" in atom_names, "Missing ftyp atom"
        assert "moov" in atom_names, "Missing moov atom"
        assert "mdat" in atom_names, "Missing mdat atom"

        ftyp_atom = next(a for a in atoms if a[0] == "ftyp")
        assert ftyp_atom[1] == 0, f"ftyp is not at beginning of file: offset {ftyp_atom[1]}"

        moov_offset = next(a[1] for a in atoms if a[0] == "moov")
        mdat_offset = next(a[1] for a in atoms if a[0] == "mdat")
        assert moov_offset < mdat_offset, f"Faststart violated: moov ({moov_offset}) appears after mdat ({mdat_offset})"

    def test_adv_live_preview_snapshot_geometry_and_format(self):
        """TC-ADV-M4-07: Checks outputs/clip_1_preview.png exists with exact 1080x1920 3-channel geometry."""
        assert self.LIVE_PREVIEW.exists(), f"Preview snapshot not found: {self.LIVE_PREVIEW}"
        img = cv2.imread(str(self.LIVE_PREVIEW))
        assert img is not None, "Failed to load preview snapshot with cv2"
        h, w, c = img.shape
        assert (h, w, c) == (1920, 1080, 3), f"Snapshot dimensions are not 1080x1920x3: ({w}x{h}x{c})"

    def test_adv_live_preview_snapshot_burned_subtitles_cv2(self):
        """TC-ADV-M4-08: Measures Canny edge density and contour count in subtitle and hook ROIs."""
        img = cv2.imread(str(self.LIVE_PREVIEW))
        assert img is not None, "Image reading failed"
        h, w, _ = img.shape

        # Subtitle region: 50% to 85% of height
        sub_roi = img[int(h * 0.50):int(h * 0.85), int(w * 0.05):int(w * 0.95)]
        gray_sub = cv2.cvtColor(sub_roi, cv2.COLOR_BGR2GRAY)
        edges_sub = cv2.Canny(gray_sub, 100, 200)
        edge_count_sub = int(np.count_nonzero(edges_sub))
        contours_sub, _ = cv2.findContours(edges_sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        assert edge_count_sub > 500, f"Subtitle edge pixel count too low ({edge_count_sub} <= 500)"
        assert len(contours_sub) >= 10, f"Subtitle contour count too low ({len(contours_sub)} < 10)"

        # Full image total edges (non-blank verification)
        total_edges = int(np.count_nonzero(cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 100, 200)))
        assert total_edges > 2000, f"Total image edge density too low ({total_edges} <= 2000)"


# ==============================================================================
# Class 2: Pipeline Concurrency & Unique File Naming
# ==============================================================================

class TestAdversarialPipelineConcurrencyAndNaming:
    """Adversarially tests unique clip naming, concurrency, and multi-thread race conditions."""

    def test_adv_unique_naming_parallel_thread_safety(self, tmp_path: Path):
        """TC-ADV-M4-09: 10 parallel threads generating unique paths with atomic touch."""
        lock = threading.Lock()
        results: List[Path] = []

        def worker():
            with lock:
                p = get_unique_clip_path(tmp_path, 1)
                p.touch()
                results.append(p)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All generated paths must be strictly distinct
        str_paths = [str(p) for p in results]
        assert len(set(str_paths)) == 10, f"Duplicate paths found under thread contention: {str_paths}"
        assert (tmp_path / "clip_1.mp4").exists()
        assert (tmp_path / "clip_1_2.mp4").exists()
        assert (tmp_path / "clip_1_10.mp4").exists()

    def test_adv_unique_naming_dense_collision_stress(self, tmp_path: Path):
        """TC-ADV-M4-10: Stress-tests 30 collisions and verifies gap recovery."""
        (tmp_path / "clip_1.mp4").touch()
        for i in range(2, 31):
            (tmp_path / f"clip_1_{i}.mp4").touch()

        # Next unique path must be clip_1_31.mp4
        next_p = get_unique_clip_path(tmp_path, 1)
        assert next_p.name == "clip_1_31.mp4"

        # Now delete clip_1_12.mp4, creating a gap in the collision chain
        (tmp_path / "clip_1_12.mp4").unlink()
        gap_p = get_unique_clip_path(tmp_path, 1)
        assert gap_p.name == "clip_1_12.mp4", f"Expected gap reuse for clip_1_12.mp4, got {gap_p.name}"

    def test_adv_unique_naming_distinct_indices_isolation(self, tmp_path: Path):
        """TC-ADV-M4-11: Confirms clip_1 and clip_2 series maintain independent sequences."""
        (tmp_path / "clip_1.mp4").touch()
        (tmp_path / "clip_1_2.mp4").touch()

        p2 = get_unique_clip_path(tmp_path, 2)
        assert p2.name == "clip_2.mp4", "clip_2 should start at base name regardless of clip_1 collisions"

    def test_adv_unique_naming_special_character_directories(self, tmp_path: Path):
        """TC-ADV-M4-12: Verifies unique naming in directories with spaces, brackets, and emojis."""
        special_dir = tmp_path / "Outputs [Shorts] #1 🚀"
        special_dir.mkdir(parents=True, exist_ok=True)

        p1 = get_unique_clip_path(special_dir, 1)
        p1.touch()
        p2 = get_unique_clip_path(special_dir, 1)
        assert p2.name == "clip_1_2.mp4"
        assert p2.parent == special_dir

    def test_adv_concurrent_pipeline_parallel_runs(self, tmp_path: Path):
        """TC-ADV-M4-13: Concurrently executes two independent pipeline runs targeting separate output folders."""
        video1 = tmp_path / "source1.mp4"
        video2 = tmp_path / "source2.mp4"
        generate_synthetic_mp4(video1, duration_s=3.0)
        generate_synthetic_mp4(video2, duration_s=3.0)

        # Sidecars
        words = [
            Word("Testing", 100, 600),
            Word("concurrent", 650, 1300),
            Word("pipelines", 1350, 2000),
        ]
        tc1 = TranscriptData(text="Testing concurrent pipelines", language="en", words=words)
        video1.with_name(f"{video1.name}.transcript_cache.json").write_text(json.dumps(tc1.to_dict()))
        video2.with_name(f"{video2.name}.transcript_cache.json").write_text(json.dumps(tc1.to_dict()))

        out1 = tmp_path / "run1_out"
        out2 = tmp_path / "run2_out"
        tmp1 = tmp_path / "run1_tmp"
        tmp2 = tmp_path / "run2_tmp"

        def run_one(vid: Path, out: Path, tmp: Path) -> int:
            return run_pipeline(
                url=str(vid),
                output_dir=out,
                temp_dir=tmp,
                max_clips=1,
                llm_model="heuristic",
                device="cpu",
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            fut1 = executor.submit(run_one, video1, out1, tmp1)
            fut2 = executor.submit(run_one, video2, out2, tmp2)
            res1 = fut1.result(timeout=60)
            res2 = fut2.result(timeout=60)

        assert res1 == 0, "Pipeline run 1 failed"
        assert res2 == 0, "Pipeline run 2 failed"
        assert (out1 / "clip_1.mp4").exists(), "Run 1 output missing"
        assert (out2 / "clip_1.mp4").exists(), "Run 2 output missing"


# ==============================================================================
# Class 3: Temp Folder Lifecycle, Isolation & Cleanup
# ==============================================================================

class TestAdversarialTempFolderLifecycleAndCleanup:
    """Verifies temporary file lifecycle, leak prevention, and containment."""

    def test_adv_temp_dir_intermediate_ass_isolation(self, tmp_path: Path):
        """TC-ADV-M4-14: Asserts ASS subtitle files are placed in temp_dir and never leak into output_dir."""
        video = tmp_path / "input.mp4"
        generate_synthetic_mp4(video, duration_s=3.0)

        words = [Word("Clean", 100, 500), Word("isolation", 550, 1500)]
        tc = TranscriptData(text="Clean isolation", language="en", words=words)
        video.with_name(f"{video.name}.transcript_cache.json").write_text(json.dumps(tc.to_dict()))

        out_dir = tmp_path / "clean_outputs"
        temp_dir = tmp_path / "isolated_temp"

        code = run_pipeline(
            url=str(video),
            output_dir=out_dir,
            temp_dir=temp_dir,
            max_clips=1,
            llm_model="heuristic",
            device="cpu",
        )
        assert code == 0

        # Output dir must only contain MP4 files, NO .ass files
        out_ass_files = list(out_dir.glob("*.ass"))
        assert len(out_ass_files) == 0, f"ASS subtitle files leaked into output dir: {out_ass_files}"

        # Temp dir must contain the generated ASS file
        temp_ass_files = list(temp_dir.glob("*.ass"))
        assert len(temp_ass_files) >= 1, f"No ASS file found in temp directory: {temp_dir}"

    def test_adv_temp_dir_nested_auto_creation(self, tmp_path: Path):
        """TC-ADV-M4-15: Asserts non-existent nested temp directory is automatically created."""
        video = tmp_path / "input.mp4"
        generate_synthetic_mp4(video, duration_s=3.0)

        words = [Word("Auto", 100, 400), Word("create", 450, 1000)]
        tc = TranscriptData(text="Auto create", language="en", words=words)
        video.with_name(f"{video.name}.transcript_cache.json").write_text(json.dumps(tc.to_dict()))

        deep_temp = tmp_path / "nested" / "deeply" / "temp_folder"
        assert not deep_temp.exists()

        code = run_pipeline(
            url=str(video),
            output_dir=tmp_path / "outputs",
            temp_dir=deep_temp,
            max_clips=1,
            llm_model="heuristic",
            device="cpu",
        )
        assert code == 0
        assert deep_temp.exists(), "Pipeline failed to create deeply nested temp directory"

    def test_adv_render_clip_tmp_file_cleaned_up_on_failure(self, tmp_path: Path):
        """TC-ADV-M4-16: When FFmpeg fails during render_clip, the atomic .tmp.mp4 file must be removed."""
        video = tmp_path / "valid.mp4"
        generate_synthetic_mp4(video, duration_s=3.0)

        out_file = tmp_path / "failed_render.mp4"

        # Intentionally invalid crop filter that causes FFmpeg to exit non-zero
        invalid_crop = "crop=999999:999999:0:0"

        with pytest.raises(RenderingError):
            render_clip(
                source_video=video,
                start_ms=0,
                end_ms=2000,
                crop_filter=invalid_crop,
                output_file=out_file,
            )

        # Target file must not exist
        assert not out_file.exists()

        # Zero dangling .tmp.mp4 files must remain in tmp_path
        tmp_files = list(tmp_path.glob(".*.tmp.mp4")) + list(tmp_path.glob("*.tmp.mp4"))
        assert len(tmp_files) == 0, f"Temporary file leak detected after render failure: {tmp_files}"

    def test_adv_render_clip_tmp_file_cleaned_up_on_timeout(self, tmp_path: Path):
        """TC-ADV-M4-17: When FFmpeg times out, the atomic .tmp.mp4 file must be unlinked."""
        video = tmp_path / "valid.mp4"
        generate_synthetic_mp4(video, duration_s=3.0)
        out_file = tmp_path / "timeout_render.mp4"

        # Simulate timeout in subprocess.run
        def mock_timeout(*args, **kwargs):
            # Create the temp file to simulate partial write before timeout
            tmp_candidates = list(tmp_path.glob(".*.tmp.mp4"))
            if tmp_candidates:
                tmp_candidates[0].write_bytes(b"partial")
            raise subprocess.TimeoutExpired(cmd=["ffmpeg"], timeout=1.0)

        with patch("subprocess.run", side_effect=mock_timeout):
            with pytest.raises(subprocess.TimeoutExpired):
                render_clip(
                    source_video=video,
                    start_ms=0,
                    end_ms=2000,
                    output_file=out_file,
                )

        # Zero dangling temp files
        tmp_files = list(tmp_path.glob(".*.tmp.mp4"))
        assert len(tmp_files) == 0, f"Leaked tmp files after timeout: {tmp_files}"

    def test_adv_successful_render_leaves_zero_tmp_files(self, tmp_path: Path):
        """TC-ADV-M4-18: Verifies that after a successful render, exactly 0 .tmp files remain."""
        video = tmp_path / "valid.mp4"
        generate_synthetic_mp4(video, duration_s=3.0)
        out_file = tmp_path / "success.mp4"

        res = render_clip(
            source_video=video,
            start_ms=0,
            end_ms=2000,
            output_file=out_file,
        )
        assert res.exists()
        tmp_files = list(tmp_path.glob(".*.tmp.mp4"))
        assert len(tmp_files) == 0, f"Dangling temp files found after success: {tmp_files}"


# ==============================================================================
# Class 4: Zero-Paid-API Leakage Under Adversarial Environments
# ==============================================================================

class TestAdversarialZeroPaidApiLeakage:
    """Verifies total immunity against paid API leakage under adversarial environments."""

    def test_adv_poisoned_environment_sanitization(self, monkeypatch):
        """TC-ADV-M4-19: Sets live fake paid API keys in environment; asserts Config sanitizes them to None."""
        monkeypatch.setenv("SELF_HOST", "true")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-live-adversarial-openai-key")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-live-adversarial-key")
        monkeypatch.setenv("ASSEMBLY_AI_API_KEY", "live-assembly-ai-token")
        monkeypatch.setenv("APIFY_API_TOKEN", "apify_token_fake")
        monkeypatch.setenv("GOOGLE_API_KEY", "google_api_key_fake")
        monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter_key_fake")

        cfg = Config()
        assert cfg.self_hosted is True
        assert cfg.openai_api_key is None
        assert cfg.anthropic_api_key is None
        assert cfg.assembly_ai_api_key is None
        assert cfg.apify_api_token is None
        assert cfg.google_api_key is None
        assert cfg.openrouter_api_key is None
        assert cfg.is_quota_enforced() is False

    @pytest.mark.parametrize("val", ["true", "True", "TRUE", "1", "yes", "YES", "on", "ON"])
    def test_adv_self_host_truthy_variations(self, monkeypatch, val: str):
        """TC-ADV-M4-20: Verifies truthy string variations enforce self_hosted=True."""
        monkeypatch.setenv("SELF_HOST", val)
        cfg = Config()
        assert cfg.self_hosted is True
        assert cfg.openai_api_key is None

    def test_adv_zero_paid_api_network_blackhole(self, monkeypatch):
        """TC-ADV-M4-21: Intercepts socket connection attempts and asserts zero outbound paid API calls."""
        attempted_hosts: List[str] = []
        orig_connect = socket.socket.connect

        def recording_connect(sock_self, address):
            host = address[0] if isinstance(address, tuple) and len(address) > 0 else str(address)
            attempted_hosts.append(str(host))
            return orig_connect(sock_self, address)

        monkeypatch.setattr(socket.socket, "connect", recording_connect)

        # Run heuristic analysis
        from clippa.analysis import extract_viral_segments
        words = [Word("Never", 100, 600), Word("pay", 650, 1200), Word("apis", 1250, 2000)]
        transcript = TranscriptData(text="Never pay apis", language="en", words=words)
        segments = extract_viral_segments(transcript, max_clips=1, llm_model="heuristic")
        assert len(segments) > 0

        # Assert no connection was made to paid API domains
        paid_domains = ["openai.com", "anthropic.com", "assemblyai.com", "apify.com"]
        for attempted in attempted_hosts:
            for domain in paid_domains:
                assert domain not in attempted.lower(), f"Leak detected: attempted connection to {attempted}"

    def test_adv_offline_airplane_mode_pipeline(self, tmp_path: Path, monkeypatch):
        """TC-ADV-M4-22: Simulates total airplane mode (DNS fails); asserts pipeline completes locally."""
        def dns_fail(host, port, *args, **kwargs):
            raise socket.gaierror(-2, "No address associated with hostname (airplane mode)")

        monkeypatch.setattr(socket, "getaddrinfo", dns_fail)

        video = tmp_path / "offline_src.mp4"
        generate_synthetic_mp4(video, duration_s=3.0)

        words = [Word("Offline", 100, 700), Word("pipeline", 750, 1600)]
        tc = TranscriptData(text="Offline pipeline", language="en", words=words)
        video.with_name(f"{video.name}.transcript_cache.json").write_text(json.dumps(tc.to_dict()))

        code = run_pipeline(
            url=str(video),
            output_dir=tmp_path / "offline_outputs",
            temp_dir=tmp_path / "offline_temp",
            max_clips=1,
            llm_model="heuristic",
            device="cpu",
        )
        assert code == 0
        assert (tmp_path / "offline_outputs" / "clip_1.mp4").exists()

    def test_adv_quota_bypass_and_resolution_unlocked(self, monkeypatch):
        """TC-ADV-M4-23: Verifies 1080p resolution unlocked and duration quotas bypassed in self-hosted mode."""
        monkeypatch.setenv("SELF_HOST", "true")
        monkeypatch.setenv("QUOTAS_ENABLED", "true")

        cfg = Config()
        assert cfg.is_quota_enforced() is False
        assert "1080p" in cfg.get_allowed_resolutions()
        assert cfg.max_duration_seconds >= 86400


# ==============================================================================
# Class 5: CLI Argument Robustness & Boundary Recovery
# ==============================================================================

class TestAdversarialCLIArgumentValidation:
    """Verifies strict CLI input parsing and defensive error handling."""

    def test_adv_cli_missing_mandatory_url(self):
        """TC-ADV-M4-24: Invoking CLI without --url exits with code 2."""
        with pytest.raises(SystemExit) as exc:
            parse_cli_args([])
        assert exc.value.code == 2

    @pytest.mark.parametrize("bad_clips", ["0", "-1", "-100", "not_a_number"])
    def test_adv_cli_invalid_max_clips_values(self, bad_clips: str):
        """TC-ADV-M4-25: Invalid or non-positive max_clips values exit with code 2."""
        with pytest.raises(SystemExit) as exc:
            parse_cli_args(["--url", "test.mp4", "--max-clips", bad_clips])
        assert exc.value.code == 2

    def test_adv_cli_invalid_template_preset(self):
        """TC-ADV-M4-26: Unsupported template preset exits with code 2."""
        with pytest.raises(SystemExit) as exc:
            parse_cli_args(["--url", "test.mp4", "--template", "unsupported_preset"])
        assert exc.value.code == 2

    def test_adv_cli_invalid_resolution_preset(self):
        """TC-ADV-M4-27: Unsupported resolution choice exits with code 2."""
        with pytest.raises(SystemExit) as exc:
            parse_cli_args(["--url", "test.mp4", "--resolution", "4k"])
        assert exc.value.code == 2

    def test_adv_cli_invalid_device_preset(self):
        """TC-ADV-M4-28: Unsupported processing device choice exits with code 2."""
        with pytest.raises(SystemExit) as exc:
            parse_cli_args(["--url", "test.mp4", "--device", "tpu"])
        assert exc.value.code == 2

    def test_adv_cli_valid_custom_arguments_passthrough(self):
        """TC-ADV-M4-29: Custom valid arguments correctly parsed and mapped to attributes."""
        args = parse_cli_args([
            "--url", "https://www.youtube.com/watch?v=jNQXAC9IVRw",
            "--output-dir", "custom_out",
            "--temp-dir", "custom_tmp",
            "--template", "hormozi",
            "--max-clips", "5",
            "--resolution", "720p",
            "--whisper-model", "small",
            "--llm-model", "heuristic",
            "--device", "cpu",
            "--no-emojis",
            "--no-zoom",
            "--no-cache",
        ])
        assert args.url == "https://www.youtube.com/watch?v=jNQXAC9IVRw"
        assert args.output_dir == "custom_out"
        assert args.temp_dir == "custom_tmp"
        assert args.template == "hormozi"
        assert args.max_clips == 5
        assert args.resolution == "720p"
        assert args.whisper_model == "small"
        assert args.llm_model == "heuristic"
        assert args.device == "cpu"
        assert args.no_emojis is True
        assert args.no_zoom is True
        assert args.no_cache is True


# ==============================================================================
# Class 6: Pipeline Seam Resilience, Corrupt Inputs & Error Handling
# ==============================================================================

class TestAdversarialPipelineSeamsAndResilience:
    """Verifies pipeline seam contracts, corrupt input recovery, and error propagation."""

    def test_adv_seam_nonexistent_local_video_fails_cleanly(self, tmp_path: Path):
        """TC-ADV-M4-30: Specifying non-existent file path logs error and returns exit code 1."""
        missing = tmp_path / "does_not_exist.mp4"
        code = run_pipeline(url=str(missing), output_dir=tmp_path / "out", temp_dir=tmp_path / "tmp")
        assert code == 1, "Pipeline should exit with code 1 on missing file"

    def test_adv_seam_zero_byte_video_fails_cleanly(self, tmp_path: Path):
        """TC-ADV-M4-31: 0-byte video file is caught defensively and returns code 1."""
        empty = tmp_path / "empty.mp4"
        empty.write_bytes(b"")

        code = run_pipeline(url=str(empty), output_dir=tmp_path / "out", temp_dir=tmp_path / "tmp")
        assert code == 1, "Pipeline should exit with code 1 on empty input file"

    def test_adv_seam_corrupt_truncated_media_fails_cleanly(self, tmp_path: Path):
        """TC-ADV-M4-32: Truncated/corrupted media file returns code 1 without unhandled crash."""
        corrupt = tmp_path / "corrupt.mp4"
        generate_corrupt_media(corrupt, corrupt_type="truncated")

        code = run_pipeline(url=str(corrupt), output_dir=tmp_path / "out", temp_dir=tmp_path / "tmp")
        assert code == 1, "Pipeline should exit with code 1 on corrupted file"

    def test_adv_seam_empty_transcription_words_fails_cleanly(self, tmp_path: Path):
        """TC-ADV-M4-33: Source with zero detected words returns code 1 without IndexError."""
        video = tmp_path / "silent.mp4"
        generate_synthetic_mp4(video, duration_s=3.0)

        # Sidecar with empty words list
        tc = TranscriptData(text="", language="en", words=[])
        video.with_name(f"{video.name}.transcript_cache.json").write_text(json.dumps(tc.to_dict()))

        code = run_pipeline(url=str(video), output_dir=tmp_path / "out", temp_dir=tmp_path / "tmp", device="cpu")
        assert code == 1, "Pipeline should return code 1 when transcription has no words"

    def test_adv_seam_zero_viral_segments_fails_cleanly(self, tmp_path: Path):
        """TC-ADV-M4-34: Virality analysis returning zero segments exits with code 1."""
        video = tmp_path / "video.mp4"
        generate_synthetic_mp4(video, duration_s=3.0)

        words = [Word("word", 100, 500)]
        tc = TranscriptData(text="word", language="en", words=words)
        video.with_name(f"{video.name}.transcript_cache.json").write_text(json.dumps(tc.to_dict()))

        with patch("clippa.cli.extract_viral_segments", return_value=[]):
            code = run_pipeline(url=str(video), output_dir=tmp_path / "out", temp_dir=tmp_path / "tmp", device="cpu")
            assert code == 1, "Pipeline should return code 1 when no viral segments found"

    def test_adv_seam_special_characters_in_hook_title_ass(self, tmp_path: Path):
        """TC-ADV-M4-35: Hook title with ASS tags, curly braces, quotes, emojis renders cleanly without crash."""
        video = tmp_path / "special_chars_src.mp4"
        generate_synthetic_mp4(video, duration_s=3.0)

        adversarial_hook = "Secret {\\b1}100%{\\b0} & \\N 'quotes' \"double\" 🔥 💰 🚀"
        ass_path = tmp_path / "special.ass"
        words = [Word("Testing", 100, 600), Word("special", 650, 1300), Word("chars", 1350, 2000)]

        generate_ass_subtitles(
            words=words,
            hook_title=adversarial_hook,
            template="default",
            output_path=ass_path,
            clip_duration_s=3.0,
        )
        assert ass_path.exists()

        out_clip = tmp_path / "special_clip.mp4"
        rendered = render_clip(
            source_video=video,
            start_ms=0,
            end_ms=2500,
            ass_path=ass_path,
            output_file=out_clip,
        )
        assert rendered.exists()
        assert_media_integrity(rendered)

    def test_adv_pipeline_e2e_synthetic_full_flow_1080p(self, tmp_path: Path):
        """TC-ADV-M4-36: Executes complete live pipeline on synthetic media, producing validated 1080x1920 MP4."""
        video = tmp_path / "e2e_synth.mp4"
        generate_synthetic_mp4(video, duration_s=3.0)

        words = [
            Word("The", 100, 400),
            Word("secret", 450, 950),
            Word("mistake", 1000, 1600),
            Word("everyone", 1650, 2200),
            Word("makes", 2250, 2800),
        ]
        tc = TranscriptData(text="The secret mistake everyone makes", language="en", words=words)
        video.with_name(f"{video.name}.transcript_cache.json").write_text(json.dumps(tc.to_dict()))

        out_dir = tmp_path / "e2e_outputs"
        temp_dir = tmp_path / "e2e_temp"

        exit_code = run_pipeline(
            url=str(video),
            output_dir=out_dir,
            temp_dir=temp_dir,
            template="hormozi",
            max_clips=1,
            resolution="1080p",
            llm_model="heuristic",
            device="cpu",
        )
        assert exit_code == 0, "Pipeline failed to execute end-to-end"

        clip = out_dir / "clip_1.mp4"
        assert clip.exists(), f"Rendered clip missing: {clip}"

        # Comprehensive ffprobe & media validation
        meta = probe_media(clip)
        assert_vertical_9_16(meta, target_w=1080, target_h=1920)
        assert_valid_h264_aac(meta)
        assert_media_integrity(clip)
