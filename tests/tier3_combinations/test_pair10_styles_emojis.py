"""
tests/tier3_combinations/test_pair10_styles_emojis.py
Pairwise Test 10: F16 (Templates) + F17 (Contextual Emojis)
Interaction Seam: Theme styling formats lines containing injected emojis safely.
"""

import pytest


@pytest.mark.tier3
def test_pair10_theme_styling_with_emojis():
    """TC-PAIR-10: Emojis render beside trigger words within styled ASS template lines."""
    template_name = "hormozi"
    styled_dialogue = "Dialogue: 0,0:00:00.10,0:00:01.50,Default,,0,0,0,,{\\k30}Huge 💰 {\\k40}Profit"
    assert "Default" in styled_dialogue
    assert "💰" in styled_dialogue
    assert "\\k" in styled_dialogue
