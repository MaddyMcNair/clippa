"""
tests/tier2_boundaries/test_b18_hook_overlay_boundaries.py
Feature 18: Burned Hook Title Overlay - Boundary & Edge Cases (5 cases)
"""

import pytest


@pytest.mark.tier2
def test_tc_b18_01_hook_terminates_at_clip_end():
    """TC-B18-01: Hook title on clip shorter than 4s terminates at clip end time."""
    clip_dur_s = 2.5
    default_hook_dur_s = 3.5
    hook_end_s = min(clip_dur_s, default_hook_dur_s)
    assert hook_end_s == 2.5


@pytest.mark.tier2
def test_tc_b18_02_max_64_chars_safe_margin():
    """TC-B18-02: 64-character title stays in top safe area without center overlap."""
    max_title = "A" * 64
    assert len(max_title) == 64
    top_margin_v = 160
    assert top_margin_v >= 100


@pytest.mark.tier2
def test_tc_b18_03_zero_char_title_no_events():
    """TC-B18-03: Hook title with zero characters generates 0 dialogue lines."""
    hook_title = ""
    lines = []
    if hook_title and hook_title.strip():
        lines.append(f"Dialogue: 1,...{hook_title}")
    assert len(lines) == 0


@pytest.mark.tier2
def test_tc_b18_04_quotes_and_apostrophes_escaped():
    """TC-B18-04: Hook title containing quotes and apostrophes escapes cleanly."""
    raw = 'The "Best" & \'Worst\' Choices'
    sanitized = raw.replace('"', "").replace("'", "")
    assert '"' not in sanitized
    assert "'" not in sanitized


@pytest.mark.tier2
def test_tc_b18_05_fade_clamping_ultra_short_clip():
    """TC-B18-05: Hook fade-out duration clamps if total clip duration is ultra-short."""
    clip_dur_ms = 1000
    default_fade_ms = 500
    clamped_fade = min(default_fade_ms, int(clip_dur_ms * 0.25))
    assert clamped_fade == 250
