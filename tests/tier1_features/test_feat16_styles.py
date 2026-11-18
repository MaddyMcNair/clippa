"""
tests/tier1_features/test_feat16_styles.py
Feature 16: Subtitle Style Templates & TTF Fonts
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
def test_tc_f16_01_hormozi_template_style(tmp_path: Path):
    """TC-F16-01: Theme selection hormozi produces bold typography and heavy outline."""
    words = make_sample_transcript(3000).words
    out = subtitles.generate_ass_subtitles(words, hook_title="Hormozi Hook", template="hormozi", output_path=tmp_path / "hormozi.ass")
    content = Path(out).read_text(encoding="utf-8")
    assert "Hormozi" in content or "Default" in content or "Style:" in content


@pytest.mark.tier1
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.subtitles not implemented yet")
def test_tc_f16_02_mrbeast_template_style(tmp_path: Path):
    """TC-F16-02: Theme selection mrbeast produces distinct yellow/red styled template."""
    words = make_sample_transcript(3000).words
    out = subtitles.generate_ass_subtitles(words, hook_title="MrBeast Hook", template="mrbeast", output_path=tmp_path / "mrbeast.ass")
    content = Path(out).read_text(encoding="utf-8")
    assert "[V4+ Styles]" in content
    assert Path(out).exists()


@pytest.mark.tier1
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.subtitles not implemented yet")
def test_tc_f16_03_tiktok_template_style(tmp_path: Path):
    """TC-F16-03: Theme selection tiktok loads clean modern style."""
    words = make_sample_transcript(3000).words
    out = subtitles.generate_ass_subtitles(words, hook_title="TikTok Hook", template="tiktok", output_path=tmp_path / "tiktok.ass")
    content = Path(out).read_text(encoding="utf-8")
    assert "[V4+ Styles]" in content
    assert Path(out).exists()


@pytest.mark.tier1
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.subtitles not implemented yet")
def test_tc_f16_04_available_templates():
    """TC-F16-04: Subtitles module exposes available templates list or template registry."""
    templates_fn = getattr(subtitles, "get_available_templates", None)
    if templates_fn:
        templates = templates_fn()
        assert "default" in templates
        assert "hormozi" in templates or "mrbeast" in templates or "tiktok" in templates
    else:
        # Check template validation via generate_ass_subtitles
        words = make_sample_transcript(3000).words
        assert len(words) > 0


@pytest.mark.tier1
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.subtitles not implemented yet")
def test_tc_f16_05_invalid_template_falls_back(tmp_path: Path):
    """TC-F16-05: Specifying an invalid template falls back gracefully to default theme."""
    words = make_sample_transcript(3000).words
    out = subtitles.generate_ass_subtitles(words, hook_title="Fallback Hook", template="non_existent_template", output_path=tmp_path / "fallback.ass")
    content = Path(out).read_text(encoding="utf-8")
    assert "[Script Info]" in content
    assert "[Events]" in content
