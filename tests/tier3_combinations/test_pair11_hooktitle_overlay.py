"""
tests/tier3_combinations/test_pair11_hooktitle_overlay.py
Pairwise Test 11: F09 (Hook Title) + F18 (Burned Hook Title Overlay)
Interaction Seam: Generated hook title is placed in top safe area dialogue event.
"""

import pytest

from tests.fixtures.synthetic import make_sample_ass_content


@pytest.mark.tier3
def test_pair11_hook_title_top_overlay():
    """TC-PAIR-11: Generated hook title formats into top-safe-area ASS event."""
    title = "The Secret To Real Wealth"
    ass_content = make_sample_ass_content(hook_title=title)

    assert "Style: Hook" in ass_content
    assert title in ass_content
    # Starts at 0s, ends at 3.5s
    assert "0:00:00.00,0:00:03.50,Hook" in ass_content
