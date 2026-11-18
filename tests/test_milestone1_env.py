"""Milestone 1 Environment and Repository Setup Verification Tests.

Verifies:
1. Python 3.11 virtual environment & PyTorch CUDA acceleration.
2. Core dependencies and clean module imports (no graphite whisper shadowing).
3. Backend repository assets: 22 TTF fonts and YuNet ONNX model hash & load.
4. FFmpeg binaries on PATH with libass & subtitles filter support.
5. Ollama daemon liveness and llama3.2:3b model readiness.
6. Local-first zero-paid-API .env configuration.
"""

import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import dotenv
import pytest
import requests
import torch

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = WORKSPACE_ROOT / "backend"
FONTS_DIR = BACKEND_DIR / "fonts"
YUNET_MODEL_PATH = BACKEND_DIR / "src" / "media" / "models" / "face_detection_yunet_2023mar.onnx"
EXPECTED_YUNET_SHA256 = "8F2383E4DD3CFBB4553EA8718107FC0423210DC964F9F4280604804ED2552FA4".lower()


def test_python_runtime():
    """Verify runtime is Python 3.11 64-bit."""
    assert sys.version_info[:2] == (3, 11), f"Expected Python 3.11, got {sys.version}"
    assert sys.maxsize > 2**32, "Expected 64-bit Python runtime"


def test_pytorch_cuda_acceleration():
    """Verify PyTorch detects NVIDIA GeForce RTX 4050 with CUDA."""
    assert torch.cuda.is_available(), "CUDA is not available in PyTorch"
    device_count = torch.cuda.device_count()
    assert device_count >= 1, "No CUDA devices discovered"
    device_name = torch.cuda.get_device_name(0)
    assert "RTX 4050" in device_name or "NVIDIA" in device_name, f"Unexpected GPU: {device_name}"


def test_core_package_imports():
    """Verify key packages import correctly and whisper is openai-whisper."""
    import whisper
    import pydantic_ai
    import yt_dlp
    import onnxruntime

    assert hasattr(whisper, "load_model"), "whisper package lacks load_model (graphite collision)"
    assert hasattr(whisper, "_MODELS"), "whisper package lacks _MODELS"
    assert hasattr(pydantic_ai, "Agent"), "pydantic_ai package lacks Agent"
    assert hasattr(yt_dlp, "YoutubeDL"), "yt_dlp package lacks YoutubeDL"
    assert hasattr(onnxruntime, "InferenceSession"), "onnxruntime lacks InferenceSession"


def test_backend_assets_fonts():
    """Verify all 22 TTF fonts are present in backend/fonts/."""
    assert FONTS_DIR.is_dir(), f"Fonts directory missing at {FONTS_DIR}"
    ttf_files = list(FONTS_DIR.glob("*.ttf"))
    assert len(ttf_files) == 22, f"Expected 22 fonts, found {len(ttf_files)}: {[f.name for f in ttf_files]}"


def test_backend_yunet_model_integrity():
    """Verify YuNet face detection ONNX model exists, matches SHA-256, and loads in OpenCV."""
    assert YUNET_MODEL_PATH.is_file(), f"YuNet model missing at {YUNET_MODEL_PATH}"
    with open(YUNET_MODEL_PATH, "rb") as f:
        file_hash = hashlib.sha256(f.read()).hexdigest().lower()
    assert file_hash == EXPECTED_YUNET_SHA256, f"YuNet SHA-256 mismatch: {file_hash} != {EXPECTED_YUNET_SHA256}"

    # Verify OpenCV can instantiate the YuNet FaceDetectorYN
    detector = cv2.FaceDetectorYN.create(
        str(YUNET_MODEL_PATH),
        "",
        (320, 320),
        0.6,
        0.3,
        5000,
    )
    assert detector is not None, "Failed to create cv2.FaceDetectorYN from model file"


def test_ffmpeg_and_filters():
    """Verify ffmpeg and ffprobe are executable and support libass/subtitles filters."""
    ffmpeg_path = shutil.which("ffmpeg")
    ffprobe_path = shutil.which("ffprobe")
    assert ffmpeg_path is not None, "ffmpeg executable not found on PATH"
    assert ffprobe_path is not None, "ffprobe executable not found on PATH"

    # Query ffmpeg version
    version_res = subprocess.run([ffmpeg_path, "-version"], capture_output=True, text=True, check=True)
    assert "ffmpeg version" in version_res.stdout, "Unexpected ffmpeg version output"

    # Query filters for ass and subtitles
    filters_res = subprocess.run([ffmpeg_path, "-filters"], capture_output=True, text=True, check=True)
    assert " ass " in filters_res.stdout, "FFmpeg lacks libass filter"
    assert " subtitles " in filters_res.stdout, "FFmpeg lacks subtitles filter"


def test_env_zero_paid_api_configuration():
    """Verify .env configuration enforces local-first zero-paid-API operation."""
    env_file = WORKSPACE_ROOT / ".env"
    assert env_file.is_file(), f".env file missing at {env_file}"

    vals = dotenv.dotenv_values(str(env_file))
    assert vals.get("SELF_HOST") == "true", "SELF_HOST must be 'true'"
    assert vals.get("MONETIZATION_ENABLED") == "false", "MONETIZATION_ENABLED must be 'false'"
    assert vals.get("MEDIA_STORAGE") == "local", "MEDIA_STORAGE must be 'local'"
    assert vals.get("TRANSCRIPTION_PROVIDER") == "whisper", "TRANSCRIPTION_PROVIDER must be 'whisper'"
    assert vals.get("WHISPER_MODEL") == "base", "WHISPER_MODEL must be 'base'"
    assert vals.get("LLM") == "ollama:llama3.2:3b", "LLM must be 'ollama:llama3.2:3b'"
    assert vals.get("RERANK_LLM") == "ollama:llama3.2:3b", "RERANK_LLM must be 'ollama:llama3.2:3b'"
    assert vals.get("OLLAMA_BASE_URL") == "http://localhost:11434/v1", "OLLAMA_BASE_URL must point to local port 11434"
    assert vals.get("YOUTUBE_DOWNLOAD_PROVIDER") == "yt_dlp", "YOUTUBE_DOWNLOAD_PROVIDER must be 'yt_dlp'"
    assert vals.get("YOUTUBE_METADATA_PROVIDER") == "yt_dlp", "YOUTUBE_METADATA_PROVIDER must be 'yt_dlp'"

    # Ensure all external paid API keys are empty
    paid_keys = [
        "OPENAI_API_KEY",
        "ASSEMBLY_AI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GOOGLE_API_KEY",
        "OPENROUTER_API_KEY",
        "APIFY_API_TOKEN",
        "RAPIDAPI_YOUTUBE_API_KEY",
        "PEXELS_API_KEY",
        "STRIPE_SECRET_KEY",
    ]
    for key in paid_keys:
        assert not vals.get(key), f"Paid key {key} must be empty in zero-cost setup, found: {vals.get(key)}"


def test_ollama_daemon_and_model():
    """Verify Ollama daemon responds on localhost:11434 and llama3.2:3b model is available."""
    resp = requests.get("http://localhost:11434/api/tags", timeout=5)
    assert resp.status_code == 200, f"Ollama daemon returned HTTP {resp.status_code}"
    data = resp.json()
    model_names = [m.get("name", "") for m in data.get("models", [])]
    has_llama = any("llama3.2:3b" in name for name in model_names)
    assert has_llama, f"Model llama3.2:3b not found in Ollama models list: {model_names}"


def test_pytorch_cuda_computation():
    """Verify actual matrix multiplication executes on the RTX 4050 GPU."""
    x = torch.randn(1000, 1000, device="cuda")
    y = torch.randn(1000, 1000, device="cuda")
    z = torch.matmul(x, y)
    assert z.is_cuda
    assert z.shape == (1000, 1000)


def test_yunet_inference_execution():
    """Verify YuNet model executes inference on an image matrix."""
    import numpy as np

    detector = cv2.FaceDetectorYN.create(
        str(YUNET_MODEL_PATH),
        "",
        (320, 320),
        0.6,
        0.3,
        5000,
    )
    dummy_img = np.zeros((320, 320, 3), dtype=np.uint8)
    faces = detector.detect(dummy_img)
    assert faces is not None
    assert faces[1] is None or isinstance(faces[1], np.ndarray)


def test_ollama_inference_generation():
    """Verify Ollama generates genuine text from llama3.2:3b model."""
    resp = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "llama3.2:3b", "prompt": "Reply with 'Local AI operational'.", "stream": False},
        timeout=30,
    )
    assert resp.status_code == 200
    text = resp.json().get("response", "").strip()
    assert len(text) > 0, "Empty response from Ollama"


def test_supoclip_config_and_ai_integration():
    """Verify SupoClip native get_config and _build_transcript_model wire to Ollama."""
    sys.path.insert(0, str(BACKEND_DIR))
    from src.config import get_config
    from src.ai import _build_transcript_model
    from pydantic_ai.models.ollama import OllamaModel

    cfg = get_config()
    assert cfg.self_host is True
    assert cfg.transcription_provider == "whisper"
    assert cfg.llm == "ollama:llama3.2:3b"

    ai_model = _build_transcript_model(cfg)
    assert isinstance(ai_model, OllamaModel)
    assert ai_model.model_name == "llama3.2:3b"
