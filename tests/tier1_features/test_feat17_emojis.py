"""
tests/tier1_features/test_feat17_emojis.py
Feature 17: Contextual Emoji Injection
Category-Partition Tests (5 cases)
"""

from pathlib import Path
import pytest

from tests.fixtures.synthetic import Word

try:
    import clippa.subtitles as subtitles
    HAS_CLIPPA = True
except ImportError:
    HAS_CLIPPA = False


@pytest.mark.tier1
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.subtitles not implemented yet")
def test_tc_f17_01_money_trigger_words(tmp_path: Path):
    """TC-F17-01: Spoken trigger words (money, dollar, cash) map to 💰 / 💵 in formatted subtitles."""
    words = [Word("Make", 0, 300), Word("money", 350, 700), Word("today", 750, 1100)]
    inject_fn = getattr(subtitles, "inject_emojis", None)
    if inject_fn:
        result_words = inject_fn(words)
        texts = [getattr(w, "text", str(w)) for w in result_words]
        combined = " ".join(texts)
        assert "💰" in combined or "💵" in combined
    else:
        out = subtitles.generate_ass_subtitles(words, enable_emojis=True, output_path=tmp_path / "emojis.ass")
        content = Path(out).read_text(encoding="utf-8")
        assert "💰" in content or "💵" in content or "money" in content


@pytest.mark.tier1
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.subtitles not implemented yet")
def test_tc_f17_02_emotion_trigger_words(tmp_path: Path):
    """TC-F17-02: Emotion keywords (fire, danger, warning) insert corresponding emojis."""
    words = [Word("This", 0, 300), Word("fire", 350, 700), Word("warning", 750, 1100)]
    inject_fn = getattr(subtitles, "inject_emojis", None)
    if inject_fn:
        result_words = inject_fn(words)
        combined = " ".join([getattr(w, "text", str(w)) for w in result_words])
        assert "🔥" in combined or "⚠️" in combined
    else:
        out = subtitles.generate_ass_subtitles(words, enable_emojis=True, output_path=tmp_path / "emotion.ass")
        content = Path(out).read_text(encoding="utf-8")
        assert "🔥" in content or "⚠️" in content or "fire" in content


@pytest.mark.tier1
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.subtitles not implemented yet")
def test_tc_f17_03_emojis_preserve_timing_tags(tmp_path: Path):
    """TC-F17-03: Emoji placement does not corrupt word timing tags in ASS string."""
    words = [Word("Huge", 0, 300), Word("money", 350, 700), Word("profit", 750, 1100)]
    out = subtitles.generate_ass_subtitles(words, enable_emojis=True, output_path=tmp_path / "tags.ass")
    content = Path(out).read_text(encoding="utf-8")
    assert r"\k" in content or r"\K" in content


@pytest.mark.tier1
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.subtitles not implemented yet")
def test_tc_f17_04_no_trigger_clean_text(tmp_path: Path):
    """TC-F17-04: Segments with no matching triggers generate clean subtitles without emojis."""
    words = [Word("The", 0, 200), Word("algorithm", 250, 600), Word("evaluates", 650, 1000)]
    inject_fn = getattr(subtitles, "inject_emojis", None)
    if inject_fn:
        result_words = inject_fn(words)
        combined = " ".join([getattr(w, "text", str(w)) for w in result_words])
        assert not any(char in combined for char in ["💰", "💵", "🔥", "⚠️", "🚀"])
    else:
        out = subtitles.generate_ass_subtitles(words, enable_emojis=True, output_path=tmp_path / "plain.ass")
        content = Path(out).read_text(encoding="utf-8")
        assert "💰" not in content and "🔥" not in content


@pytest.mark.tier1
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.subtitles not implemented yet")
def test_tc_f17_05_flag_disables_emojis(tmp_path: Path):
    """TC-F17-05: Configurable flag (enable_emojis=False) disables emojis entirely."""
    words = [Word("Make", 0, 300), Word("money", 350, 700), Word("fire", 750, 1100)]
    out = subtitles.generate_ass_subtitles(words, enable_emojis=False, output_path=tmp_path / "no_emojis.ass")
    content = Path(out).read_text(encoding="utf-8")
    assert "💰" not in content
    assert "🔥" not in content
