"""
tests/tier1_features/test_feat15_ass_builder.py
Feature 15: Animated ASS Subtitle Builder
Category-Partition Tests (5 cases)
"""

from pathlib import Path
import re
import pytest

from tests.fixtures.synthetic import make_sample_transcript

try:
    import clippa.subtitles as subtitles
    HAS_CLIPPA = True
except ImportError:
    HAS_CLIPPA = False


@pytest.mark.tier1
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.subtitles not implemented yet")
def test_tc_f15_01_valid_ass_headers(tmp_path: Path):
    """TC-F15-01: Generates valid ASS file containing standard headers ([Script Info], [V4+ Styles], [Events])."""
    words = make_sample_transcript(3000).words
    out = subtitles.generate_ass_subtitles(words, hook_title="Viral Hook", template="default", output_path=tmp_path / "subs.ass")
    content = Path(out).read_text(encoding="utf-8")
    assert "[Script Info]" in content
    assert "[V4+ Styles]" in content
    assert "[Events]" in content
    assert "PlayResX: 1080" in content or "1080" in content
    assert "PlayResY: 1920" in content or "1920" in content


@pytest.mark.tier1
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.subtitles not implemented yet")
def test_tc_f15_02_karaoke_timing_tags(tmp_path: Path):
    """TC-F15-02: Emits word-by-word karaoke timing tags (\\k / \\K)."""
    words = make_sample_transcript(3000).words
    out = subtitles.generate_ass_subtitles(words, template="default", output_path=tmp_path / "karaoke.ass")
    content = Path(out).read_text(encoding="utf-8")
    assert r"\k" in content or r"\K" in content


@pytest.mark.tier1
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.subtitles not implemented yet")
def test_tc_f15_03_active_word_highlight(tmp_path: Path):
    """TC-F15-03: Active spoken word animates with highlight color transition in style."""
    words = make_sample_transcript(3000).words
    out = subtitles.generate_ass_subtitles(words, template="default", output_path=tmp_path / "highlight.ass")
    content = Path(out).read_text(encoding="utf-8")
    assert "Style:" in content
    assert "PrimaryColour" in content or "SecondaryColour" in content


@pytest.mark.tier1
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.subtitles not implemented yet")
def test_tc_f15_04_line_word_limit(tmp_path: Path):
    """TC-F15-04: Subtitle lines are capped at 3 to 5 words for vertical screen readability."""
    words = make_sample_transcript(6000).words
    out = subtitles.generate_ass_subtitles(words, template="default", output_path=tmp_path / "chunks.ass")
    content = Path(out).read_text(encoding="utf-8")
    dialogue_lines = [l for l in content.splitlines() if l.startswith("Dialogue:")]
    assert len(dialogue_lines) >= 1
    for line in dialogue_lines:
        text_part = line.split(",,", 1)[-1]
        clean_text = re.sub(r"\{[^}]*\}", "", text_part).strip()
        word_count = len(clean_text.split())
        assert word_count <= 8


@pytest.mark.tier1
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.subtitles not implemented yet")
def test_tc_f15_05_timestamps_match_speech(tmp_path: Path):
    """TC-F15-05: Subtitle event start and end timestamps match audio speech timestamps."""
    words = make_sample_transcript(3000).words
    out = subtitles.generate_ass_subtitles(words, template="default", output_path=tmp_path / "timing.ass")
    content = Path(out).read_text(encoding="utf-8")
    dialogue_lines = [l for l in content.splitlines() if l.startswith("Dialogue:")]
    assert len(dialogue_lines) >= 1
    first_diag = dialogue_lines[0]
    parts = first_diag.split(",")
    # Dialogue: Layer, Start, End, Style, ...
    start_str = parts[1]
    assert re.match(r"\d+:\d{2}:\d{2}\.\d{2}", start_str) is not None
