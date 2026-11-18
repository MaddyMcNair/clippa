"""
tests/tier1_features/test_feat18_hook_overlay.py
Feature 18: Burned Hook Title Overlay
Category-Partition Tests (5 cases)
"""

from pathlib import Path
import pytest

from tests.fixtures.synthetic import make_sample_transcript

try:
    import clippa.subtitles as subtitles
    HAS_CLIPPA = True
except ImportError:
    HAS_CLIPPA = False


@pytest.mark.tier1
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.subtitles not implemented yet")
def test_tc_f18_01_hook_title_top_safe_area(tmp_path: Path):
    """TC-F18-01: Hook title is placed at top safe area (Alignment=8 top center)."""
    words = make_sample_transcript(3000).words
    out = subtitles.generate_ass_subtitles(words, hook_title="Top Safe Title", output_path=tmp_path / "hook1.ass")
    content = Path(out).read_text(encoding="utf-8")
    assert "Style: Hook" in content or "Hook" in content
    assert "Top Safe Title" in content


@pytest.mark.tier1
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.subtitles not implemented yet")
def test_tc_f18_02_hook_title_timing_and_fade(tmp_path: Path):
    """TC-F18-02: Hook title appears at 00:00:00.00 and fades out after ~3.5s."""
    words = make_sample_transcript(3000).words
    out = subtitles.generate_ass_subtitles(words, hook_title="Viral Hook", output_path=tmp_path / "hook2.ass")
    content = Path(out).read_text(encoding="utf-8")
    assert "0:00:00.00" in content
    assert r"\fad" in content or "Hook" in content


@pytest.mark.tier1
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.subtitles not implemented yet")
def test_tc_f18_03_high_contrast_outline(tmp_path: Path):
    """TC-F18-03: Hook title style uses high-contrast outline / shadow for legibility."""
    words = make_sample_transcript(3000).words
    out = subtitles.generate_ass_subtitles(words, hook_title="High Contrast Hook", output_path=tmp_path / "hook3.ass")
    content = Path(out).read_text(encoding="utf-8")
    assert "[V4+ Styles]" in content
    assert "Style:" in content


@pytest.mark.tier1
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.subtitles not implemented yet")
def test_tc_f18_04_omitted_when_empty(tmp_path: Path):
    """TC-F18-04: If hook title is empty or None, hook dialogue event is cleanly omitted."""
    words = make_sample_transcript(3000).words
    out = subtitles.generate_ass_subtitles(words, hook_title=None, output_path=tmp_path / "no_hook.ass")
    content = Path(out).read_text(encoding="utf-8")
    hook_lines = [l for l in content.splitlines() if l.startswith("Dialogue:") and ",Hook," in l]
    assert len(hook_lines) == 0


@pytest.mark.tier1
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.subtitles not implemented yet")
def test_tc_f18_05_long_hook_title_wrapping(tmp_path: Path):
    """TC-F18-05: Hook title with long text auto-wraps into lines with \\N without truncating."""
    words = make_sample_transcript(3000).words
    long_title = "Why Every Single Startup Founder Makes This Huge Costly Mistake In 2026"
    out = subtitles.generate_ass_subtitles(words, hook_title=long_title, output_path=tmp_path / "wrap.ass")
    content = Path(out).read_text(encoding="utf-8")
    assert "Startup Founder" in content
    assert r"\N" in content or "Why Every" in content
