"""
tests/conftest.py - Pytest Root Configuration, Plugins & Shared Fixtures
"""

import http.client
import os
import shutil
import socket
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Set
import pytest

from tests.fixtures.synthetic import (
    Word,
    TranscriptData,
    ViralSegment,
    find_ffmpeg,
    generate_synthetic_mp4,
    generate_aspect_ratio_mp4,
    generate_silent_mp4,
    generate_video_no_audio,
    generate_audio_only,
    generate_moving_face_mp4,
    make_sample_transcript,
    make_sample_viral_segment,
    make_sample_ass_content,
)
from tests.utils import find_ffprobe


# ---------------------------------------------------------------------------
# Zero-Paid-API Transport-Level Network Interceptor
# ---------------------------------------------------------------------------

PAID_API_DOMAINS: Set[str] = {
    "api.openai.com",
    "api.anthropic.com",
    "api.assemblyai.com",
    "api.apify.com",
}


class PaidAPIBlockedError(RuntimeError):
    """Raised when an outbound network call to a paid cloud API domain is intercepted."""
    pass


_ORIG_GETADDRINFO = socket.getaddrinfo
_ORIG_SOCKET_CONNECT = socket.socket.connect
_ORIG_HTTP_CONNECT = http.client.HTTPConnection.connect
_ACTIVE_INTERCEPTORS: List["PaidAPIInterceptor"] = []


def _patched_getaddrinfo(host, port, *args, **kwargs):
    if PaidAPIInterceptor.is_paid_domain(host):
        record = {
            "host": host,
            "port": port,
            "layer": "socket.getaddrinfo",
            "timestamp": time.time(),
        }
        for inc in _ACTIVE_INTERCEPTORS:
            inc.attempts.append(record)
        if any(inc.raise_on_blocked for inc in _ACTIVE_INTERCEPTORS):
            raise PaidAPIBlockedError(
                f"Zero-Paid-API Violation: Blocked DNS resolution for paid API domain '{host}' (port {port})"
            )
        raise socket.gaierror(-2, f"Name or service not known: {host} (Blocked by PaidAPIInterceptor)")
    return _ORIG_GETADDRINFO(host, port, *args, **kwargs)


def _patched_socket_connect(sock_self, address):
    host = address[0] if isinstance(address, tuple) and len(address) > 0 else None
    if PaidAPIInterceptor.is_paid_domain(host):
        record = {
            "host": host,
            "address": address,
            "layer": "socket.socket.connect",
            "timestamp": time.time(),
        }
        for inc in _ACTIVE_INTERCEPTORS:
            inc.attempts.append(record)
        if any(inc.raise_on_blocked for inc in _ACTIVE_INTERCEPTORS):
            raise PaidAPIBlockedError(
                f"Zero-Paid-API Violation: Blocked socket connection to paid API domain '{host}'"
            )
        raise ConnectionRefusedError(f"Connection to paid API domain '{host}' refused")
    return _ORIG_SOCKET_CONNECT(sock_self, address)


def _patched_http_connect(http_self):
    host = getattr(http_self, "host", None)
    if PaidAPIInterceptor.is_paid_domain(host):
        record = {
            "host": host,
            "port": getattr(http_self, "port", None),
            "layer": "http.client.HTTPConnection.connect",
            "timestamp": time.time(),
        }
        for inc in _ACTIVE_INTERCEPTORS:
            inc.attempts.append(record)
        if any(inc.raise_on_blocked for inc in _ACTIVE_INTERCEPTORS):
            raise PaidAPIBlockedError(
                f"Zero-Paid-API Violation: Blocked HTTP connection to paid API domain '{host}'"
            )
        raise ConnectionRefusedError(f"HTTP connection to paid API domain '{host}' refused")
    return _ORIG_HTTP_CONNECT(http_self)


class PaidAPIInterceptor:
    """
    Socket-level and HTTP-client interceptor that monitors and blocks outbound
    network requests to paid third-party API domains.
    Provides verifiable call tracking, blocked call records, and assertion utilities.
    """

    def __init__(self, raise_on_blocked: bool = True):
        self.attempts: List[Dict[str, Any]] = []
        self.raise_on_blocked = raise_on_blocked
        self._is_active = False

    @staticmethod
    def is_paid_domain(host: Optional[str]) -> bool:
        if not host or not isinstance(host, str):
            return False
        h = host.lower().strip()
        for domain in PAID_API_DOMAINS:
            if h == domain or h.endswith("." + domain):
                return True
            base = domain.removeprefix("api.")
            if h == base or h.endswith("." + base):
                return True
        return False

    def start(self):
        if self._is_active:
            return
        if len(_ACTIVE_INTERCEPTORS) == 0:
            socket.getaddrinfo = _patched_getaddrinfo
            socket.socket.connect = _patched_socket_connect
            http.client.HTTPConnection.connect = _patched_http_connect
        _ACTIVE_INTERCEPTORS.append(self)
        self._is_active = True

    def stop(self):
        if not self._is_active:
            return
        if self in _ACTIVE_INTERCEPTORS:
            _ACTIVE_INTERCEPTORS.remove(self)
        if len(_ACTIVE_INTERCEPTORS) == 0:
            socket.getaddrinfo = _ORIG_GETADDRINFO
            socket.socket.connect = _ORIG_SOCKET_CONNECT
            http.client.HTTPConnection.connect = _ORIG_HTTP_CONNECT
        self._is_active = False

    @property
    def call_count(self) -> int:
        return len(self.attempts)

    def assert_zero_calls(self):
        assert len(self.attempts) == 0, (
            f"Zero-Paid-API Assertion Failed! Expected 0 outbound calls to paid APIs, "
            f"but detected {len(self.attempts)} attempted call(s): {self.attempts}"
        )


@pytest.fixture
def paid_api_interceptor() -> Generator[PaidAPIInterceptor, None, None]:
    """
    Pytest fixture providing active socket-level and HTTP-client network interception
    for paid API domains. Tests can inspect interceptor.attempts and assert zero calls.
    """
    interceptor = PaidAPIInterceptor(raise_on_blocked=True)
    interceptor.start()
    try:
        yield interceptor
    finally:
        interceptor.stop()


@pytest.fixture(autouse=True)
def zero_paid_api_guard(request) -> Generator[None, None, None]:
    """
    Autouse fixture that runs for every test in the suite, guaranteeing that
    no test accidentally contacts paid cloud services without being detected.
    """
    interceptor = PaidAPIInterceptor(raise_on_blocked=True)
    interceptor.start()
    try:
        yield
    finally:
        interceptor.stop()


def pytest_configure(config):
    """Register custom markers for tier selection and requirement checks."""
    config.addinivalue_line("markers", "tier1: Tier 1 Category-Partition feature test")
    config.addinivalue_line("markers", "tier2: Tier 2 Boundary value and corner case test")
    config.addinivalue_line("markers", "tier3: Tier 3 Pairwise cross-feature interaction test")
    config.addinivalue_line("markers", "tier4: Tier 4 Real-world application scenario test")
    config.addinivalue_line("markers", "requires_ffmpeg: Test requires FFmpeg binary installed on system")
    config.addinivalue_line("markers", "requires_ffprobe: Test requires FFprobe binary installed on system")
    config.addinivalue_line("markers", "requires_ollama: Test requires local Ollama daemon active")
    config.addinivalue_line("markers", "requires_cuda: Test requires NVIDIA GPU with CUDA support")
    config.addinivalue_line("markers", "live_network: Test requires active outbound internet connection")


@pytest.fixture(autouse=True)
def check_test_requirements(request):
    """Automatically skips tests if explicit markers require unavailable dependencies."""
    if request.node.get_closest_marker("requires_ffmpeg"):
        if not find_ffmpeg():
            pytest.skip("Test requires FFmpeg, but ffmpeg binary was not found.")

    if request.node.get_closest_marker("requires_ffprobe"):
        if not find_ffprobe():
            pytest.skip("Test requires FFprobe, but ffprobe binary was not found.")

    if request.node.get_closest_marker("live_network"):
        if os.environ.get("CLIPPA_TIER4_LIVE", "0") != "1":
            pytest.skip("Test requires live network, but --tier4-live / CLIPPA_TIER4_LIVE was not enabled.")


@pytest.fixture(scope="session")
def session_tmp_dir() -> Generator[Path, None, None]:
    """Provides a shared temporary directory for the entire test session."""
    with tempfile.TemporaryDirectory(prefix="clippa_test_session_") as d:
        yield Path(d)


@pytest.fixture
def test_dir(tmp_path: Path) -> Path:
    """Provides a clean temporary directory for an individual test."""
    return tmp_path


@pytest.fixture(scope="session")
def sample_1080p_mp4(session_tmp_dir: Path) -> Path:
    """Generates and returns a shared 1920x1080 MP4 video (3s) with audio."""
    if not find_ffmpeg():
        pytest.skip("FFmpeg not installed; cannot generate synthetic sample_1080p_mp4.")
    p = session_tmp_dir / "sample_1080p_3s.mp4"
    if not p.exists():
        generate_synthetic_mp4(p, duration_s=3.0, width=1920, height=1080)
    return p


@pytest.fixture(scope="session")
def sample_9_16_mp4(session_tmp_dir: Path) -> Path:
    """Generates and returns a shared 1080x1920 9:16 vertical MP4 video (3s)."""
    if not find_ffmpeg():
        pytest.skip("FFmpeg not installed; cannot generate synthetic sample_9_16_mp4.")
    p = session_tmp_dir / "sample_9_16_3s.mp4"
    if not p.exists():
        generate_aspect_ratio_mp4(p, aspect="9:16", duration_s=3.0)
    return p


@pytest.fixture(scope="session")
def sample_silent_mp4(session_tmp_dir: Path) -> Path:
    """Generates and returns a video with silent audio track (3s)."""
    if not find_ffmpeg():
        pytest.skip("FFmpeg not installed; cannot generate synthetic sample_silent_mp4.")
    p = session_tmp_dir / "sample_silent_3s.mp4"
    if not p.exists():
        generate_silent_mp4(p, duration_s=3.0)
    return p


@pytest.fixture(scope="session")
def sample_no_audio_mp4(session_tmp_dir: Path) -> Path:
    """Generates and returns a video without any audio stream (3s)."""
    if not find_ffmpeg():
        pytest.skip("FFmpeg not installed; cannot generate synthetic sample_no_audio_mp4.")
    p = session_tmp_dir / "sample_no_audio_3s.mp4"
    if not p.exists():
        generate_video_no_audio(p, duration_s=3.0)
    return p


@pytest.fixture(scope="session")
def sample_audio_mp3(session_tmp_dir: Path) -> Path:
    """Generates and returns a 16kHz mono MP3 audio file (3s)."""
    if not find_ffmpeg():
        pytest.skip("FFmpeg not installed; cannot generate synthetic sample_audio_mp3.")
    p = session_tmp_dir / "sample_audio_16k.mp3"
    if not p.exists():
        generate_audio_only(p, duration_s=3.0, format="mp3")
    return p


@pytest.fixture(scope="session")
def sample_moving_face_mp4(session_tmp_dir: Path) -> Path:
    """Generates and returns a video with moving box simulating face tracking (4s)."""
    if not find_ffmpeg():
        pytest.skip("FFmpeg not installed; cannot generate synthetic sample_moving_face_mp4.")
    p = session_tmp_dir / "sample_moving_face_4s.mp4"
    if not p.exists():
        generate_moving_face_mp4(p, duration_s=4.0)
    return p


@pytest.fixture
def sample_transcript() -> TranscriptData:
    """Provides a deterministic 10-word sample TranscriptData."""
    return make_sample_transcript(duration_ms=3000)


@pytest.fixture
def sample_viral_segment() -> ViralSegment:
    """Provides a deterministic valid ViralSegment."""
    return make_sample_viral_segment(start_ms=0, end_ms=30000)


@pytest.fixture
def sample_ass_file(test_dir: Path) -> Path:
    """Writes a valid .ass subtitle file and returns its path."""
    ass_path = test_dir / "sample_subtitle.ass"
    ass_path.write_text(make_sample_ass_content(hook_title="Test Hook Title"), encoding="utf-8")
    return ass_path
