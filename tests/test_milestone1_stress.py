"""Milestone 1 Adversarial Stress and Empirical Validation Tests.

Authored by Challenger 1.
Verifies:
1. GPU CUDA progressive tensor allocation, GEMM throughput, and OOM recovery.
2. OpenCV YuNet face detection on synthetic image, bbox format, resolution scaling, and memory leak checks.
3. Real FFmpeg + libass vertical video subtitle burning using bundled backend fonts.
4. Ollama LLM structured JSON schema enforcement, token generation throughput, and prompt processing latency.
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path

import cv2
import numpy as np
import pytest
import requests
import torch

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = WORKSPACE_ROOT / "backend"
FONTS_DIR = BACKEND_DIR / "fonts"
YUNET_MODEL_PATH = BACKEND_DIR / "src" / "media" / "models" / "face_detection_yunet_2023mar.onnx"


def test_gpu_tensor_stress_and_oom_recovery():
    """Verify GPU VRAM stability, GEMM compute, and graceful OOM recovery."""
    assert torch.cuda.is_available(), "CUDA device not available"
    dev = torch.device("cuda:0")

    # Allocation test: 1.0 GB and 2.0 GB allocations
    t1 = torch.ones((256, 1024, 1024), dtype=torch.float32, device=dev) # 1.0 GB
    assert t1.shape == (256, 1024, 1024)
    del t1
    torch.cuda.empty_cache()

    # Sustained GEMM compute test: 30 iterations of 4096 x 4096
    a = torch.randn(4096, 4096, dtype=torch.float16, device=dev)
    b = torch.randn(4096, 4096, dtype=torch.float16, device=dev)
    torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(30):
        c = torch.matmul(a, b)
    torch.cuda.synchronize()
    dt = time.time() - t0
    tflops = (2 * (4096**3) * 30) / (dt * 1e12)
    assert tflops > 3.0, f"Expected >3 TFLOPS on RTX 4050, got {tflops:.2f}"

    del a, b, c
    torch.cuda.empty_cache()

    # OOM boundary stress test
    with pytest.raises(torch.cuda.OutOfMemoryError):
        _ = torch.empty((100_000, 100_000, 50), dtype=torch.float32, device=dev)

    # Recovery test: Ensure CUDA context remains valid and responsive after OOM
    torch.cuda.empty_cache()
    rec = torch.matmul(torch.ones(200, 200, device=dev), torch.ones(200, 200, device=dev))
    assert rec[0, 0].item() == 200.0


def test_yunet_face_detection_synthetic_and_res_scaling():
    """Verify YuNet detection on synthetic face, bounding box format, and memory stability."""
    assert YUNET_MODEL_PATH.is_file(), f"YuNet model missing at {YUNET_MODEL_PATH}"

    detector = cv2.FaceDetectorYN.create(
        str(YUNET_MODEL_PATH),
        "",
        (640, 480),
        0.2, # Low threshold for synthetic drawing
        0.3,
        1000,
    )

    # 1. Blank frame test -> MUST return None (critical downstream contract)
    detector.setInputSize((300, 300))
    _, blank_faces = detector.detect(np.zeros((300, 300, 3), dtype=np.uint8))
    assert blank_faces is None, f"Expected None on blank frame, got {type(blank_faces)}"

    # 2. Synthetic face test
    h, w = 480, 640
    canvas = np.full((h, w, 3), 220, dtype=np.uint8)
    # Draw head, eyes, nose, mouth
    cv2.ellipse(canvas, (320, 240), (100, 140), 0, 0, 360, (180, 180, 220), -1)
    cv2.circle(canvas, (280, 210), 14, (40, 40, 40), -1)
    cv2.circle(canvas, (360, 210), 14, (40, 40, 40), -1)
    cv2.circle(canvas, (320, 250), 8, (100, 100, 100), -1)
    cv2.ellipse(canvas, (320, 290), (40, 15), 0, 0, 180, (40, 40, 160), 4)

    detector.setInputSize((w, h))
    _, synth_faces = detector.detect(canvas)
    assert synth_faces is not None and len(synth_faces) >= 1, "Synthetic face failed to detect"

    face = synth_faces[0]
    assert len(face) == 15, f"Expected 15 elements in face array, got {len(face)}"
    bbox = face[0:4]
    assert bbox[2] > 50 and bbox[3] > 50, f"Unreasonable bounding box: {bbox}"
    score = face[14]
    assert score > 0.3, f"Unexpectedly low face score: {score}"

    # 3. Latency check at 640x360
    detector.setInputSize((640, 360))
    test_frame = np.zeros((360, 640, 3), dtype=np.uint8)
    t0 = time.time()
    for _ in range(10):
        detector.detect(test_frame)
    ms_per_frame = ((time.time() - t0) / 10) * 1000
    assert ms_per_frame < 30.0, f"YuNet 640x360 too slow: {ms_per_frame:.1f}ms"

    # 4. Heap memory stability (leak check)
    tracemalloc.start()
    for _ in range(100):
        detector.detect(test_frame)
    current_mem, _ = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert current_mem < 5 * 1024 * 1024, f"Memory leak detected: {current_mem} bytes"


def test_ffmpeg_libass_bundled_font_rendering():
    """Verify FFmpeg burns ASS subtitles using bundled backend fonts without crashing."""
    ffmpeg_bin = shutil.which("ffmpeg") or r"C:\ffmpeg\bin\ffmpeg.exe"
    ffprobe_bin = shutil.which("ffprobe") or r"C:\ffmpeg\bin\ffprobe.exe"
    assert os.path.exists(ffmpeg_bin), f"FFmpeg missing at {ffmpeg_bin}"
    assert os.path.exists(ffprobe_bin), f"FFprobe missing at {ffprobe_bin}"

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        ass_file = tmp_path / "test.ass"
        output_mp4 = tmp_path / "output_burned.mp4"
        frame_png = tmp_path / "frame.png"

        # Construct ASS with bundled font Anton
        ass_content = (
            "[Script Info]\n"
            "ScriptType: v4.00+\n"
            "PlayResX: 1080\n"
            "PlayResY: 1920\n\n"
            "[V4+ Styles]\n"
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
            "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, "
            "Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
            "Style: Default,Anton,80,&H0000FFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,4,2,2,10,10,200,1\n\n"
            "[Events]\n"
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
            "Dialogue: 0,0:00:00.00,0:00:01.00,Default,,0,0,0,,{\\k20}STRESS {\\k30}TEST {\\k40}BURNING!\n"
        )
        ass_file.write_text(ass_content, encoding="utf-8")

        # On Windows, FFmpeg filtergraph parser treats ':' as an option delimiter.
        # Absolute paths with drive letters (C:) MUST escape colons as '\:' and use forward slashes.
        ass_escaped = str(ass_file).replace("\\", "/").replace(":", r"\:")
        fonts_dir_escaped = str(FONTS_DIR.resolve()).replace("\\", "/").replace(":", r"\:")

        filter_arg = f"ass=filename='{ass_escaped}':fontsdir='{fonts_dir_escaped}'"
        cmd = [
            ffmpeg_bin,
            "-y",
            "-f", "lavfi",
            "-i", "testsrc=duration=1:size=1080x1920:rate=30",
            "-vf", filter_arg,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            str(output_mp4),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        assert proc.returncode == 0, f"FFmpeg failed (rc={proc.returncode}): {proc.stderr}"
        assert output_mp4.is_file() and output_mp4.stat().st_size > 1000

        # ffprobe validation
        probe_cmd = [
            ffprobe_bin,
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,codec_name",
            "-of", "json",
            str(output_mp4),
        ]
        probe_res = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
        probe_data = json.loads(probe_res.stdout)
        stream = probe_data["streams"][0]
        assert stream["codec_name"] == "h264"
        assert stream["width"] == 1080
        assert stream["height"] == 1920

        # Extract frame at 0.5s and verify subtitle pixels
        extract_cmd = [
            ffmpeg_bin,
            "-y",
            "-ss", "00:00:00.500",
            "-i", str(output_mp4),
            "-vframes", "1",
            str(frame_png),
        ]
        subprocess.run(extract_cmd, capture_output=True, text=True, check=True)
        assert frame_png.is_file()

        img = cv2.imread(str(frame_png))
        assert img is not None
        # Subtitle region around y = 1600-1850
        sub_crop = img[1600:1850, :]
        yellow_pixels = np.sum((sub_crop[:, :, 0] < 50) & (sub_crop[:, :, 1] > 200) & (sub_crop[:, :, 2] > 200))
        assert yellow_pixels > 100, f"Subtitles did not render visually onto frame ({yellow_pixels} yellow px)"


def test_ollama_load_and_json_schema_enforcement():
    """Verify Ollama handles prompt under load and strictly emits valid JSON schema."""
    transcript = (
        "Here is the secret to scaling high performance pipelines. Most teams fail because "
        "they load massive models into production without profiling memory footprints or "
        "checking time-to-first-token. By implementing intelligent caching and local routing, "
        "you can achieve 10x higher throughput while keeping infrastructure costs near zero."
    )
    prompt = (
        "You are an expert video editor. Analyze this transcript and return a valid JSON object "
        "with keys: start_time_ms (int), end_time_ms (int), hook_title (string), total_score (int 1-100), reasoning (string).\n\n"
        f"Transcript:\n{transcript}"
    )

    t0 = time.time()
    resp = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3.2:3b",
            "prompt": prompt,
            "format": "json",
            "stream": False,
            "options": {"temperature": 0.2, "num_ctx": 4096},
        },
        timeout=60,
    )
    dt = time.time() - t0
    assert resp.status_code == 200, f"Ollama HTTP error {resp.status_code}"
    res_data = resp.json()

    eval_count = res_data.get("eval_count", 0)
    eval_duration = res_data.get("eval_duration", 1) / 1e9
    tps = eval_count / eval_duration if eval_duration > 0 else 0

    assert dt < 15.0, f"Ollama response too slow: {dt:.2f}s"
    assert tps > 30.0, f"Ollama throughput too low on RTX 4050: {tps:.1f} tok/s"

    raw_response = res_data.get("response", "")
    parsed = json.loads(raw_response)
    item = parsed[0] if isinstance(parsed, list) else parsed
    if "segments" in item:
        item = item["segments"][0]

    for key in ["start_time_ms", "end_time_ms", "hook_title", "total_score"]:
        assert key in item, f"Key '{key}' missing in Ollama structured response: {item}"
