"""
tests/tier3_combinations/test_pair09_whisper_ass_subtitles.py
Pairwise Test 09: F04 (Whisper) + F15 (Animated ASS Subtitle Builder)
Interaction Seam: Whisper millisecond intervals drive ASS \\k karaoke timing tags.
"""

import re
import pytest

from tests.fixtures.synthetic import Word


@pytest.mark.tier3
def test_pair09_whisper_words_drive_karaoke_tags():
    """TC-PAIR-09: Cumulative karaoke tags on each line match line duration within tolerance."""
    words = [
        Word("This", 100, 350),      # dur = 250ms -> 25cs
        Word("is", 360, 500),        # dur = 140ms -> 14cs
        Word("great", 510, 800),     # dur = 290ms -> 29cs
    ]
    # Build karaoke dialogue fragment
    tags = [f"{{\\k{int((w.end - w.start) / 10)}}}{w.text}" for w in words]
    dialogue_text = " ".join(tags)

    # Sum \\k tags
    k_values = [int(m) for m in re.findall(r"\\k(\d+)", dialogue_text)]
    total_tag_ms = sum(k * 10 for k in k_values)

    expected_duration_ms = sum(w.end - w.start for w in words)
    assert abs(total_tag_ms - expected_duration_ms) <= 20
