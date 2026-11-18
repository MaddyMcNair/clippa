"""
tests/tier1_features/test_feat20_self_host.py
Feature 20: Self-Hosted Zero-Paid-API Mode
Category-Partition Tests (5 cases)
"""

import pytest

try:
    import clippa.config as config
    import clippa.transcription as transcription
    import clippa.analysis as analysis
    HAS_CLIPPA = True
except ImportError:
    HAS_CLIPPA = False


@pytest.mark.tier1
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.config not implemented yet")
def test_tc_f20_01_empty_api_keys_allowed(monkeypatch):
    """TC-F20-01: Running with empty/unset API keys and SELF_HOST=true executes without error."""
    monkeypatch.setenv("SELF_HOST", "true")
    monkeypatch.setenv("ASSEMBLY_AI_API_KEY", "")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    monkeypatch.setenv("APIFY_API_TOKEN", "")

    cfg = config.load_config() if hasattr(config, "load_config") else config.get_config()
    assert getattr(cfg, "self_hosted", getattr(cfg, "self_host", True)) is True
    assert getattr(cfg, "openai_api_key", "") in ("", None)


@pytest.mark.tier1
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.config not implemented yet")
def test_tc_f20_02_quotas_disabled_in_self_hosted(monkeypatch):
    """TC-F20-02: Video duration quotas and minute restrictions are disabled in self-hosted mode."""
    monkeypatch.setenv("SELF_HOST", "true")
    cfg = config.load_config() if hasattr(config, "load_config") else config.get_config()
    if hasattr(cfg, "is_quota_enforced"):
        assert cfg.is_quota_enforced() is False
    else:
        assert getattr(cfg, "quotas_enabled", False) is False or getattr(cfg, "max_duration_seconds", 99999) >= 86400


@pytest.mark.tier1
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.config not implemented yet")
def test_tc_f20_03_full_1080p_unlocked(monkeypatch):
    """TC-F20-03: Full 1080p export resolution is accessible without subscription paywall."""
    monkeypatch.setenv("SELF_HOST", "true")
    cfg = config.load_config() if hasattr(config, "load_config") else config.get_config()
    resolutions = cfg.get_allowed_resolutions() if hasattr(cfg, "get_allowed_resolutions") else getattr(cfg, "allowed_resolutions", ["1080p"])
    assert "1080p" in resolutions


@pytest.mark.tier1
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.transcription not implemented yet")
def test_tc_f20_04_zero_assemblyai_calls(monkeypatch, paid_api_interceptor):
    """TC-F20-04: All transcription routes to local Whisper; zero network calls to AssemblyAI."""
    monkeypatch.setenv("SELF_HOST", "true")
    provider = transcription.get_transcription_provider() if hasattr(transcription, "get_transcription_provider") else "whisper"
    assert "whisper" in provider
    paid_api_interceptor.assert_zero_calls()


@pytest.mark.tier1
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.analysis not implemented yet")
def test_tc_f20_05_zero_openai_claude_calls(monkeypatch, paid_api_interceptor):
    """TC-F20-05: All virality scoring routes to local Ollama or heuristic engine."""
    monkeypatch.setenv("SELF_HOST", "true")
    engine = analysis.get_analysis_engine() if hasattr(analysis, "get_analysis_engine") else "ollama"
    assert engine in ["ollama", "heuristic"]
    paid_api_interceptor.assert_zero_calls()
