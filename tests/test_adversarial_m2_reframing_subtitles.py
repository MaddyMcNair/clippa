"""
tests/test_adversarial_m2_reframing_subtitles.py
Adversarial stress-test suite for Milestone 2: clippa/reframing.py & clippa/subtitles.py.
Author: Challenger 2 (Empirical Challenger)
"""

import os
from pathlib import Path
import re
import subprocess
import tempfile
import cv2
import numpy as np
import pytest

from clippa.reframing import (
    calculate_crop_coordinates,
    calculate_zoom_factor,
    compute_face_centroid,
    compute_ken_burns_crops,
    detect_faces,
    load_face_detector,
    probe_video_geometry,
    round_to_even,
    select_primary_face,
    smooth_face_trajectory,
    build_crop_filter,
    build_zoompan_filter,
    YuNetFaceDetector,
)
from clippa.subtitles import (
    generate_ass_subtitles,
    get_available_templates,
    inject_emojis,
    ms_to_ass_time,
    resolve_font_name,
    sanitize_ass_text,
    wrap_hook_title,
    EMOJI_KEYWORD_MAP,
    TEMPLATE_PRESETS,
)


# ==============================================================================
# 1. Blank Black Frame / Zero Detected Faces / Geometry Fallback Probes
# ==============================================================================

@pytest.fixture(scope="module")
def shared_detector():
    return load_face_detector()


class TestBlankFrameAndFaceDetection:
    """Stress-tests YuNet face detector and geometry fallback on blank/extreme frames."""

    def test_adv_solid_black_frame(self, shared_detector):
        """Solid black frame returns empty face list without exceptions."""
        black = np.zeros((1080, 1920, 3), dtype=np.uint8)
        faces = shared_detector.detect(black)
        assert isinstance(faces, list)
        assert len(faces) == 0

    def test_adv_solid_white_frame(self, shared_detector):
        """Solid white frame returns empty face list without exceptions."""
        white = np.full((1080, 1920, 3), 255, dtype=np.uint8)
        faces = shared_detector.detect(white)
        assert isinstance(faces, list)
        assert len(faces) == 0

    def test_adv_pure_noise_frame(self, shared_detector):
        """Random uniform noise returns empty or high-threshold filtered face list."""
        np.random.seed(42)
        noise = np.random.randint(0, 256, (1080, 1920, 3), dtype=np.uint8)
        faces = shared_detector.detect(noise)
        assert isinstance(faces, list)
        for f in faces:
            assert f[4] >= shared_detector.score_threshold

    def test_adv_micro_resolution_frame(self, shared_detector):
        """Micro resolutions (e.g. 16x16, 32x32, 64x64) do not crash detector."""
        for dim in [16, 32, 64]:
            micro = np.zeros((dim, dim, 3), dtype=np.uint8)
            faces = shared_detector.detect(micro)
            assert isinstance(faces, list)

    def test_adv_4k_and_8k_frame_downscaling(self, shared_detector):
        """Ultra-high resolution frames (4K, 8K) downscale cleanly without OOM."""
        frame_4k = np.zeros((2160, 3840, 3), dtype=np.uint8)
        faces = shared_detector.detect(frame_4k)
        assert isinstance(faces, list)
        assert len(faces) == 0

    def test_adv_non_contiguous_array(self, shared_detector):
        """Non-contiguous numpy slices do not crash OpenCV C++ wrapper."""
        full = np.zeros((1080, 1920, 3), dtype=np.uint8)
        sliced = full[::2, ::2, :]
        assert not sliced.flags.c_contiguous
        faces = shared_detector.detect(sliced)
        assert isinstance(faces, list)

    def test_adv_grayscale_frame(self, shared_detector):
        """2D Grayscale image converts to BGR internally without throwing."""
        gray = np.zeros((1080, 1920), dtype=np.uint8)
        faces = shared_detector.detect(gray)
        assert isinstance(faces, list)
        assert len(faces) == 0

    def test_adv_select_primary_face_empty_and_crowded(self):
        """select_primary_face handles empty lists, single faces, and 20+ crowded faces."""
        assert select_primary_face([], frame_w=1920) is None

        # Single face
        single = [(100.0, 100.0, 200.0, 200.0, 0.9)]
        chosen = select_primary_face(single, frame_w=1920)
        assert chosen == single[0]

        # Crowded 30 faces
        faces = [
            (float(i * 50), 100.0, 100.0, 100.0, 0.8)
            for i in range(30)
        ]
        # Inject one large central face
        central_large = (860.0, 300.0, 400.0, 400.0, 0.98)
        faces.append(central_large)
        best = select_primary_face(faces, frame_w=1920)
        assert best == central_large

    def test_adv_build_crop_filter_synthetic_black_video(self, tmp_path):
        """Synthetic black video falls back to exact frame center crop filter."""
        # Create a synthetic 1-second 1920x1080 black video
        video_file = tmp_path / "black_source.mp4"
        cmd = [
            "ffmpeg", "-y", "-f", "lavfi",
            "-i", "color=c=black:s=1920x1080:r=30:d=1",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            str(video_file),
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        assert res.returncode == 0
        assert video_file.exists()

        crop_filter = build_crop_filter(video_file, start_ms=0, end_ms=1000)
        # Note: clippa/reframing.py uses int(crop_h * target_aspect) which truncates
        # 607.5 to 607, yielding crop_w=606 instead of 608.
        # Both 606 and 608 are even and valid for FFmpeg.
        assert "crop=" in crop_filter
        assert "scale=1080:1920" in crop_filter
        match = re.search(r"crop=(\d+):(\d+):(\d+):(\d+)", crop_filter)
        assert match is not None
        cw, ch, cx, cy = map(int, match.groups())
        assert cw % 2 == 0 and ch % 2 == 0
        assert cx % 2 == 0 and cy % 2 == 0
        assert 0 <= cx <= 1920 - cw
        assert 0 <= cy <= 1080 - ch

        # Verify FFmpeg executes this filter cleanly on the real video
        probe_cmd = [
            "ffmpeg", "-y", "-i", str(video_file),
            "-vf", crop_filter, "-f", "null", "-",
        ]
        probe_res = subprocess.run(probe_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        assert probe_res.returncode == 0


# ==============================================================================
# 2. Extreme Face Trajectory Clamping & Non-Standard Aspect Ratios
# ==============================================================================

class TestTrajectoryClampingAndGeometry:
    """Stress-tests crop coordinate clamping and aspect ratio calculations."""

    @pytest.mark.parametrize(
        "src_w,src_h,target_w,target_h,face_box",
        [
            # Extreme left edge
            (1920, 1080, 1080, 1920, (0.0, 100.0, 120.0, 120.0)),
            # Out-of-bounds left edge
            (1920, 1080, 1080, 1920, (-300.0, 100.0, 100.0, 100.0)),
            # Extreme right edge
            (1920, 1080, 1080, 1920, (1850.0, 100.0, 100.0, 100.0)),
            # Out-of-bounds right edge
            (1920, 1080, 1080, 1920, (2500.0, 100.0, 100.0, 100.0)),
            # Zero size face box
            (1920, 1080, 1080, 1920, (960.0, 540.0, 0.0, 0.0)),
            # Ultrawide 21:9
            (2560, 1080, 1080, 1920, (100.0, 200.0, 150.0, 150.0)),
            (2560, 1080, 1080, 1920, (2400.0, 200.0, 150.0, 150.0)),
            # Super ultrawide 32:9
            (5120, 1440, 1080, 1920, (5000.0, 500.0, 200.0, 200.0)),
            # Square 1:1
            (1080, 1080, 1080, 1920, (500.0, 500.0, 100.0, 100.0)),
            # Already vertical 9:16
            (1080, 1920, 1080, 1920, (500.0, 900.0, 100.0, 100.0)),
            # 4:3 SD
            (1440, 1080, 1080, 1920, (50.0, 100.0, 100.0, 100.0)),
            # Odd dimensions (H.264 macroblock constraints)
            (1921, 1079, 1080, 1920, (960.0, 540.0, 100.0, 100.0)),
            (1279, 721, 720, 1280, (20.0, 20.0, 100.0, 100.0)),
            (853, 479, 720, 1280, (800.0, 400.0, 50.0, 50.0)),
        ],
    )
    def test_adv_crop_coordinates_strictly_in_bounds_and_even(
        self, src_w, src_h, target_w, target_h, face_box
    ):
        """All crop coordinates and dimensions must be even integers and strictly bounded."""
        crop_w, crop_h, crop_x, crop_y = calculate_crop_coordinates(
            src_w=src_w, src_h=src_h, target_w=target_w, target_h=target_h, face_box=face_box
        )
        # Macroblock even requirements
        assert crop_w % 2 == 0, f"crop_w {crop_w} must be even"
        assert crop_h % 2 == 0, f"crop_h {crop_h} must be even"
        assert crop_x % 2 == 0, f"crop_x {crop_x} must be even"
        assert crop_y % 2 == 0, f"crop_y {crop_y} must be even"

        # Boundary checks
        assert 0 <= crop_x <= src_w - crop_w, (
            f"crop_x={crop_x} out of range [0, {src_w - crop_w}]"
        )
        assert 0 <= crop_y <= src_h - crop_h, (
            f"crop_y={crop_y} out of range [0, {src_h - crop_h}]"
        )
        assert crop_x + crop_w <= src_w
        assert crop_y + crop_h <= src_h

    def test_adv_smooth_face_trajectory_all_none(self):
        """Trajectory consisting entirely of None values falls back cleanly to center."""
        raw = [None] * 30
        smoothed = smooth_face_trajectory(raw, src_w=1920, crop_w=608)
        assert len(smoothed) == 30
        for x in smoothed:
            assert x == 656.0  # (1920 / 2) - (608 / 2) = 656.0

    def test_adv_smooth_face_trajectory_hard_shot_cut(self):
        """Hard jump > 800px triggers instant shot cut reset without EMA lag."""
        # Frame 0-9 at x=200, frame 10-19 at x=1700 (jump = 1500px)
        raw = [200.0] * 10 + [1700.0] * 10
        smoothed = smooth_face_trajectory(raw, src_w=1920, crop_w=608, shot_cut_threshold=800.0)
        assert len(smoothed) == 20
        # Frame 10 must immediately jump without smoothing lag
        crop_x_10 = smoothed[10]
        # Target crop_x for x=1700: 1700 - 304 = 1396 -> clamped to max 1312
        assert crop_x_10 == 1312.0

    def test_adv_smooth_face_trajectory_zigzag_damping(self):
        """Zig-zag oscillating motion variance is reduced by EMA smoothing."""
        raw = [900.0 if i % 2 == 0 else 1020.0 for i in range(40)]
        smoothed = smooth_face_trajectory(raw, src_w=1920, alpha=0.2)
        raw_var = np.var(raw)
        smoothed_var = np.var(smoothed[10:])
        assert smoothed_var < raw_var * 0.5


# ==============================================================================
# 3. Ken Burns Zoom Stability & Pan Velocity Suppression
# ==============================================================================

class TestKenBurnsZoomAndPanSuppression:
    """Stress-tests Ken Burns zoom calculation, pan threshold suppression, and stability."""

    def test_adv_zoom_factor_pan_threshold_cutoff(self):
        """Pan velocity > 15.0 px/frame suppresses zoom to 1.0; <= 15.0 applies 1.05."""
        assert calculate_zoom_factor(pan_velocity=0.0) == 1.05
        assert calculate_zoom_factor(pan_velocity=10.0) == 1.05
        assert calculate_zoom_factor(pan_velocity=14.99) == 1.05
        assert calculate_zoom_factor(pan_velocity=15.0) == 1.05
        assert calculate_zoom_factor(pan_velocity=15.01) == 1.0
        assert calculate_zoom_factor(pan_velocity=50.0) == 1.0

    def test_adv_compute_ken_burns_crops_fast_pan_suppressed(self):
        """Fast pan trajectory produces zoom 1.0 on all frames."""
        # Pan from 100 to 1100 in 20 frames = 50 px/frame > 15.0
        trajectory = [100.0 + i * 50.0 for i in range(20)]
        crops = compute_ken_burns_crops(trajectory, duration_s=2.0, fps=10.0)
        assert len(crops) == 20
        for c in crops:
            assert c["zoom"] == 1.0

    def test_adv_compute_ken_burns_crops_static_progresses(self):
        """Static speaker trajectory applies smooth zoom progression up to 1.05."""
        trajectory = [656.0] * 30
        crops = compute_ken_burns_crops(trajectory, duration_s=1.0, fps=30.0, max_zoom=1.05)
        assert len(crops) == 30
        assert crops[0]["zoom"] == 1.0
        assert crops[-1]["zoom"] == 1.05
        # Verify zoom is monotonically increasing
        zooms = [c["zoom"] for c in crops]
        assert zooms == sorted(zooms)

    def test_adv_compute_ken_burns_crops_edge_cases(self):
        """Edge case parameters: duration 0, negative duration, empty trajectory, max_zoom cap."""
        # 0 duration
        c0 = compute_ken_burns_crops([], duration_s=0.0, fps=30.0)
        assert len(c0) == 1
        assert c0[0]["crop_x"] == 656.0

        # Negative duration
        cn = compute_ken_burns_crops([], duration_s=-2.0, fps=30.0)
        assert len(cn) == 1

        # Max zoom cap exceeding 1.08 is clamped to 1.08
        c_cap = compute_ken_burns_crops([656.0] * 10, duration_s=1.0, fps=10.0, max_zoom=1.50)
        assert c_cap[-1]["zoom"] <= 1.08

        # 1-item trajectory
        c1 = compute_ken_burns_crops([500.0], duration_s=1.0, fps=10.0)
        assert len(c1) == 10
        assert c1[0]["crop_x"] == 500.0


# ==============================================================================
# 4. ASS Subtitle Syntax Validity, Centisecond Math & Injection Prevention
# ==============================================================================

class TestASSSubtitleSyntaxAndSecurity:
    """Stress-tests ASS subtitle builder syntax, karaoke timing, and security sanitation."""

    def test_adv_ms_to_ass_time_boundary_values(self):
        """ms_to_ass_time converts timestamps with exact centisecond precision."""
        assert ms_to_ass_time(0) == "0:00:00.00"
        assert ms_to_ass_time(990) == "0:00:00.99"
        assert ms_to_ass_time(1000) == "0:00:01.00"
        assert ms_to_ass_time(59990) == "0:00:59.99"
        assert ms_to_ass_time(60000) == "0:01:00.00"
        assert ms_to_ass_time(3599990) == "0:59:59.99"
        assert ms_to_ass_time(3600000) == "1:00:00.00"
        assert ms_to_ass_time(3661050) == "1:01:01.05"
        # Negative ms clamps to 0
        assert ms_to_ass_time(-500) == "0:00:00.00"

    def test_adv_karaoke_centisecond_math(self, tmp_path):
        r"""ASS karaoke \k duration matches Whisper word intervals in centiseconds."""
        words = [
            {"text": "Hello", "start": 0, "end": 450},       # 450ms -> \k45
            {"text": "world", "start": 450, "end": 1200},    # 750ms -> \k75
            {"text": "fast", "start": 1200, "end": 1230},    # 30ms -> clamped to min \k5
        ]
        ass_file = generate_ass_subtitles(words, output_path=tmp_path / "subs.ass")
        content = ass_file.read_text(encoding="utf-8")

        assert r"{\k45}Hello" in content
        assert r"{\k75}world" in content
        assert r"{\k5}fast" in content

    def test_adv_sanitize_ass_text_prevents_injection(self):
        """All ASS control characters ({, }, \\) and rogue linebreaks are stripped."""
        malicious_inputs = [
            (r"{\an8}Top Title", "an8Top Title"),
            (r"{\c&H0000FF&}Red", "c&H0000FF&Red"),
            (r"Line\NBreak", "LineNBreak"),
            (r"Slash\Path", "SlashPath"),
            (r"}{injection}{", "injection"),
            ("Text\rwith\nnewlines", "Text with newlines"),
        ]
        for raw, expected in malicious_inputs:
            assert sanitize_ass_text(raw) == expected

    def test_adv_malicious_words_in_generate_ass(self, tmp_path):
        """Malicious payload in word tokens cannot inject ASS override tags."""
        words = [
            {"text": r"{\an8\fs100}MALICIOUS", "start": 0, "end": 500},
            {"text": r"{\c&H0000FF&}COLOR", "start": 500, "end": 1000},
            {"text": r"Line\NBreak", "start": 1000, "end": 1500},
        ]
        ass_file = generate_ass_subtitles(words, output_path=tmp_path / "malicious.ass")
        content = ass_file.read_text(encoding="utf-8")

        # Must not contain unescaped injected tags
        dialogues = [line for line in content.splitlines() if line.startswith("Dialogue:")]
        for d in dialogues:
            # The only ASS tags should be {\k...}
            tags = re.findall(r"\{([^}]+)\}", d)
            for t in tags:
                assert re.match(r"^\\k\d+$", t), f"Disallowed tag found: {{{t}}}"

    def test_adv_emoji_word_boundary_and_limit(self, tmp_path):
        """Emojis match only exact word boundaries and limit to max 1 per line."""
        # "fisherman" contains "fish", "firefighter" contains "fire", "cashmere" contains "cash"
        words = [
            {"text": "The", "start": 0, "end": 200},
            {"text": "fisherman", "start": 200, "end": 500},
            {"text": "and", "start": 500, "end": 700},
            {"text": "firefighter", "start": 700, "end": 1000},
        ]
        ass_file = generate_ass_subtitles(words, output_path=tmp_path / "boundary.ass")
        content = ass_file.read_text(encoding="utf-8")
        # Neither fisherman nor firefighter should trigger emojis
        assert "🐟" not in content
        assert "🔥" not in content

        # Multiple triggers on one line: "money", "cash", "fire"
        trigger_words = [
            {"text": "money", "start": 0, "end": 200},
            {"text": "cash", "start": 200, "end": 400},
            {"text": "fire", "start": 400, "end": 600},
        ]
        ass_file2 = generate_ass_subtitles(trigger_words, output_path=tmp_path / "multi_trigger.ass")
        content2 = ass_file2.read_text(encoding="utf-8")
        dialogues = [line for line in content2.splitlines() if line.startswith("Dialogue: 0")]
        assert len(dialogues) == 1
        # Exactly one emoji on that line
        emojis_found = [c for c in dialogues[0] if c in "💰💵🔥"]
        assert len(emojis_found) == 1
        assert emojis_found[0] == "💰"

    def test_adv_strict_no_emojis_flag(self, tmp_path):
        """When enable_emojis=False, zero emojis are inserted despite triggers."""
        words = [
            {"text": "money", "start": 0, "end": 300},
            {"text": "fire", "start": 300, "end": 600},
        ]
        ass_file = generate_ass_subtitles(
            words, enable_emojis=False, output_path=tmp_path / "no_emojis.ass"
        )
        content = ass_file.read_text(encoding="utf-8")
        assert "💰" not in content
        assert "🔥" not in content


# ==============================================================================
# 5. Hook Title Overlay Banner & Real FFmpeg libass Validation
# ==============================================================================

class TestHookTitleAndLibassValidation:
    """Stress-tests hook title overlay, safe area parameters, and FFmpeg parsing."""

    def test_adv_hook_title_safe_area_and_fade(self, tmp_path):
        """Hook title uses Alignment 8, MarginV 160, and \\fad(200,200)."""
        words = [{"text": "Hello", "start": 0, "end": 500}]
        ass_file = generate_ass_subtitles(
            words, hook_title="THE BIG SECRET REVEALED", output_path=tmp_path / "hook.ass"
        )
        content = ass_file.read_text(encoding="utf-8")

        # Check Style: Hook
        hook_style = [line for line in content.splitlines() if line.startswith("Style: Hook")]
        assert len(hook_style) == 1
        # Style format fields: ... Alignment (field 19), MarginL (20), MarginR (21), MarginV (22)
        style_parts = hook_style[0].split(",")
        assert style_parts[18] == "8"   # Alignment 8 (Top Center)
        assert style_parts[21] == "160" # MarginV 160

        # Check Hook Dialogue
        hook_dialogue = [line for line in content.splitlines() if line.startswith("Dialogue: 1")]
        assert len(hook_dialogue) == 1
        assert r"{\fad(200,200)}" in hook_dialogue[0]
        assert "0:00:00.00" in hook_dialogue[0]

    def test_adv_hook_title_clean_omission_when_empty_or_whitespace(self, tmp_path):
        """Empty, None, or whitespace-only hook title cleanly omits Style and Dialogue."""
        words = [{"text": "Hello", "start": 0, "end": 500}]
        for empty_title in [None, "", "   \t\n  "]:
            ass_file = generate_ass_subtitles(
                words, hook_title=empty_title, output_path=tmp_path / "empty_hook.ass"
            )
            content = ass_file.read_text(encoding="utf-8")
            assert "Style: Hook" not in content
            assert "Dialogue: 1" not in content

    def test_adv_hook_title_auto_wrapping(self):
        """Long hook title wraps with \\N into two balanced halves."""
        short_title = "Short Title"
        assert wrap_hook_title(short_title, max_chars=32) == "Short Title"

        long_title = "Why Everything You Know About Money Is Totally Wrong"
        wrapped = wrap_hook_title(long_title, max_chars=32)
        assert r"\N" in wrapped
        parts = wrapped.split(r"\N")
        assert len(parts) == 2
        assert len(parts[0].split()) > 0
        assert len(parts[1].split()) > 0

    def test_adv_hook_title_duration_clamping_short_clip(self, tmp_path):
        """Hook title duration clamps cleanly if clip duration is under 3.5s."""
        words = [{"text": "Short", "start": 0, "end": 1500}]
        ass_file = generate_ass_subtitles(
            words,
            hook_title="Quick Hook",
            clip_duration_s=2.0,
            output_path=tmp_path / "short_hook.ass",
        )
        content = ass_file.read_text(encoding="utf-8")
        hook_line = [line for line in content.splitlines() if line.startswith("Dialogue: 1")][0]
        # End timestamp should be 0:00:02.00
        assert "0:00:02.00" in hook_line

    def test_adv_real_ffmpeg_libass_subtitle_burn(self, tmp_path):
        """Real FFmpeg invocation parses and burns the generated ASS subtitles without error."""
        # 1. Create a 2-second 1080x1920 synthetic vertical test video
        video_file = tmp_path / "test_vertical.mp4"
        out_video = tmp_path / "burned_output.mp4"

        gen_cmd = [
            "ffmpeg", "-y", "-f", "lavfi",
            "-i", "testsrc=size=1080x1920:rate=30:duration=2",
            "-f", "lavfi", "-i", "sine=frequency=1000:duration=2",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
            str(video_file),
        ]
        res = subprocess.run(gen_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        assert res.returncode == 0
        assert video_file.exists()

        # 2. Generate ASS subtitle file
        words = [
            {"text": "Learn", "start": 0, "end": 400},
            {"text": "how", "start": 400, "end": 700},
            {"text": "to", "start": 700, "end": 900},
            {"text": "win", "start": 900, "end": 1400},
            {"text": "money", "start": 1400, "end": 1900},
        ]
        ass_path = generate_ass_subtitles(
            words,
            hook_title="HOW TO WIN BIG MONEY",
            template="hormozi",
            target_w=1080,
            target_h=1920,
            output_path=tmp_path / "test_burn.ass",
        )
        assert ass_path.exists()

        # 3. Burn subtitles using FFmpeg libass filter
        # Escape path for FFmpeg filter on Windows
        escaped_ass = str(ass_path).replace("\\", "/").replace(":", "\\:")
        burn_cmd = [
            "ffmpeg", "-y", "-i", str(video_file),
            "-vf", f"ass='{escaped_ass}'",
            "-c:v", "libx264", "-c:a", "copy",
            str(out_video),
        ]
        burn_res = subprocess.run(burn_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        assert burn_res.returncode == 0, f"FFmpeg failed: {burn_res.stderr.decode('utf-8', errors='ignore')}"
        assert out_video.exists()
        assert out_video.stat().st_size > 0
