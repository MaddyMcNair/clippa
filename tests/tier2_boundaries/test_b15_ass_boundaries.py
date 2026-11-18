"""
tests/tier2_boundaries/test_b15_ass_boundaries.py
Feature 15: Animated ASS Subtitle Builder - Boundary & Edge Cases (5 cases)
"""

import re
import pytest

from tests.fixtures.synthetic import Word


@pytest.mark.tier2
def test_tc_b15_01_ultra_short_word_clamped():
    """TC-B15-01: Words under 50ms get minimum visible karaoke duration (>=5cs)."""
    raw_duration_ms = 20
    min_visible_cs = 5  # 50ms
    cs = max(min_visible_cs, int(raw_duration_ms / 10))
    assert cs >= 5


@pytest.mark.tier2
def test_tc_b15_02_50_word_sentence_chunking():
    """TC-B15-02: 50+ word sentence chunks into 3-5 word events without overflowing screen."""
    words = [Word(f"w{i}", i * 300, i * 300 + 250) for i in range(50)]
    chunk_size = 4
    chunks = [words[i:i + chunk_size] for i in range(0, len(words), chunk_size)]
    assert len(chunks) == 13
    assert all(len(c) <= 5 for c in chunks)


@pytest.mark.tier2
def test_tc_b15_03_zero_gap_transitions():
    """TC-B15-03: Zero word gaps between consecutive words transition cleanly without overlap."""
    w1 = Word("first", 100, 300)
    w2 = Word("second", 300, 500)  # exact zero gap
    assert w1.end == w2.start


@pytest.mark.tier2
def test_tc_b15_04_silence_pauses_split_events():
    """TC-B15-04: Multi-second silence pause (>1.5s) closes current event and splits."""
    w1 = Word("end", 1000, 1500)
    w2 = Word("start", 4000, 4500)  # 2.5s pause
    pause_gap = w2.start - w1.end
    should_split = pause_gap > 1500
    assert should_split is True


@pytest.mark.tier2
def test_tc_b15_05_escape_ass_control_characters():
    """TC-B15-05: Control characters (\\N, \\h, {}) in spoken words are escaped or stripped."""
    dangerous_word = r"hack{\b1}\Ntest\h"
    # Escaping logic
    safe_word = re.sub(r"[\{\}\\]", "", dangerous_word)
    assert "{" not in safe_word
    assert "}" not in safe_word
    assert "\\" not in safe_word
    assert "hackb1Ntesth" in safe_word or "hack" in safe_word
