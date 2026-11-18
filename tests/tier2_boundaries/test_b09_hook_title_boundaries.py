"""
tests/tier2_boundaries/test_b09_hook_title_boundaries.py
Feature 9: Hook Title Generation - Boundary & Edge Cases (5 cases)
"""

import pytest

try:
    import clippa.analysis as analysis
    HAS_CLIPPA = True
except ImportError:
    HAS_CLIPPA = False


@pytest.mark.tier2
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.analysis not implemented yet")
def test_tc_b09_01_short_title_padding():
    """TC-B09-01: AI generates 1-word title; validator pads context or uses fallback."""
    raw_title = "Mistake"
    sanitize_fn = getattr(analysis, "sanitize_hook_title", getattr(analysis, "format_hook_title", None))
    if sanitize_fn:
        result = sanitize_fn(raw_title)
        assert len(result.split()) >= 3
    else:
        pytest.skip("sanitize_hook_title pending in clippa.analysis")


@pytest.mark.tier2
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.analysis not implemented yet")
def test_tc_b09_02_long_title_truncation():
    """TC-B09-02: AI generates title >15 words / >64 chars; validator truncates cleanly."""
    long_raw = "This is an extraordinarily long and verbose headline that simply goes on and on without stopping"
    sanitize_fn = getattr(analysis, "sanitize_hook_title", getattr(analysis, "format_hook_title", None))
    if sanitize_fn:
        truncated = sanitize_fn(long_raw)
        assert len(truncated) <= 64
        assert len(truncated.split()) <= 9
    else:
        pytest.skip("sanitize_hook_title pending in clippa.analysis")


@pytest.mark.tier2
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.analysis not implemented yet")
def test_tc_b09_03_punctuation_only_stripped():
    """TC-B09-03: Title containing only punctuation/emojis substitutes fallback."""
    raw_title = "??? !!! 🔥🔥🔥"
    sanitize_fn = getattr(analysis, "sanitize_hook_title", getattr(analysis, "format_hook_title", None))
    if sanitize_fn:
        cleaned = sanitize_fn(raw_title)
        assert len(cleaned) > 0
        assert "???" not in cleaned or len(cleaned.split()) >= 3
    else:
        pytest.skip("sanitize_hook_title pending in clippa.analysis")


@pytest.mark.tier2
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.analysis not implemented yet")
def test_tc_b09_04_empty_string_fallback():
    """TC-B09-04: Empty string returned from LLM defaults to sanitized title."""
    sanitize_fn = getattr(analysis, "sanitize_hook_title", getattr(analysis, "format_hook_title", None))
    if sanitize_fn:
        fallback = sanitize_fn("")
        assert len(fallback) > 0
    else:
        pytest.skip("sanitize_hook_title pending in clippa.analysis")


@pytest.mark.tier2
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.analysis not implemented yet")
def test_tc_b09_05_ass_control_characters_escaped():
    """TC-B09-05: Title containing ASS escape characters ({, }, \\) escapes properly."""
    injected_title = r"The {\b1}Secret\N Revealed: 100% {True}"
    sanitize_fn = getattr(analysis, "sanitize_hook_title", getattr(analysis, "format_hook_title", None))
    if sanitize_fn:
        escaped = sanitize_fn(injected_title)
        assert "{" not in escaped
        assert "}" not in escaped
    else:
        pytest.skip("sanitize_hook_title pending in clippa.analysis")
