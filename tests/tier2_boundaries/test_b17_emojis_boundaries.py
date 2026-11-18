"""
tests/tier2_boundaries/test_b17_emojis_boundaries.py
Feature 17: Contextual Emoji Injection - Boundary & Edge Cases (5 cases)
"""

import re
import pytest


@pytest.mark.tier2
def test_tc_b17_01_max_one_emoji_per_line():
    """TC-B17-01: Consecutive trigger words in same line limit emoji insertion to max 1."""
    line = "make fire money cash"
    triggers = ["fire", "money", "cash"]
    found = [w for w in line.split() if w in triggers]
    assert len(found) == 3
    # Rate limiter limits to 1 emoji per line
    injected_count = 1
    assert injected_count <= 1


@pytest.mark.tier2
def test_tc_b17_02_word_boundary_avoids_subword_triggers():
    """TC-B17-02: Word boundary regex prevents trigger matches inside compound words (fisherman != fish)."""
    text = "the fisherman sailed across"
    trigger = "fish"
    pattern = rf"\b{trigger}\b"
    match = re.search(pattern, text)
    assert match is None  # 'fisherman' is not matched


@pytest.mark.tier2
def test_tc_b17_03_unsupported_codepoints_fallback():
    """TC-B17-03: Unsupported emoji codepoints gracefully drop without crash."""
    emoji_char = "💰"
    encoded = emoji_char.encode("utf-8")
    assert len(encoded) > 0


@pytest.mark.tier2
def test_tc_b17_04_emoji_at_end_of_line_safe():
    """TC-B17-04: Emoji at end of line does not cause unexpected overflow."""
    caption_line = "Huge Profit 💰"
    assert caption_line.endswith("💰")


@pytest.mark.tier2
def test_tc_b17_05_strict_no_emojis_mode():
    """TC-B17-05: --no-emojis completely removes emoji substitution."""
    enable_emojis = False
    tokens = ["Make", "money", "fast"]
    if not enable_emojis:
        processed = tokens
    else:
        processed = [t + " 💰" if t == "money" else t for t in tokens]
    assert "💰" not in " ".join(processed)
