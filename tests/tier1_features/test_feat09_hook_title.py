"""
tests/tier1_features/test_feat09_hook_title.py
Feature 9: Hook Title Generation
Category-Partition Tests (5 cases)
"""

from unittest.mock import MagicMock, patch
import pytest

from tests.fixtures.synthetic import make_sample_transcript, make_sample_viral_segment

try:
    import clippa.analysis as analysis
    HAS_CLIPPA = True
except ImportError:
    HAS_CLIPPA = False


@pytest.mark.tier1
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.analysis not implemented yet")
def test_tc_f09_01_title_word_count():
    """TC-F09-01: Generated hook title is between 3 and 9 words."""
    seg = make_sample_viral_segment(0, 30000)
    transcript = make_sample_transcript(30000, "The biggest mistake new developers make is not writing tests early.")
    title_fn = getattr(analysis, "generate_hook_title", None)
    if title_fn:
        title = title_fn(seg, transcript)
        assert 3 <= len(title.split()) <= 9
    else:
        results = analysis.extract_viral_segments(transcript, max_clips=1, llm_model="heuristic")
        assert len(results) >= 1
        assert 3 <= len(results[0].hook_title.split()) <= 9


@pytest.mark.tier1
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.analysis not implemented yet")
def test_tc_f09_02_hook_title_character_budget():
    """TC-F09-02: Hook title character length does not exceed 64 characters."""
    seg = make_sample_viral_segment(0, 30000)
    transcript = make_sample_transcript(30000, "Why smart engineers make this huge software architecture mistake every single year.")
    title_fn = getattr(analysis, "generate_hook_title", None)
    if title_fn:
        title = title_fn(seg, transcript)
        assert len(title) <= 64
    else:
        results = analysis.extract_viral_segments(transcript, max_clips=1, llm_model="heuristic")
        assert len(results[0].hook_title) <= 64


@pytest.mark.tier1
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.analysis not implemented yet")
def test_tc_f09_03_hook_title_no_hashtags_or_quotes():
    """TC-F09-03: Title formatting excludes social hashtags (#) and surrounding quotation marks."""
    sanitize_fn = getattr(analysis, "sanitize_hook_title", None)
    if sanitize_fn:
        clean = sanitize_fn('"#Viral #Tech The Truth About Software"')
        assert "#" not in clean
        assert not (clean.startswith('"') and clean.endswith('"'))
    else:
        seg = make_sample_viral_segment(0, 30000)
        transcript = make_sample_transcript(30000)
        title_fn = getattr(analysis, "generate_hook_title", lambda s, t: analysis.extract_viral_segments(t, max_clips=1, llm_model="heuristic")[0].hook_title)
        title = title_fn(seg, transcript)
        assert "#" not in title


@pytest.mark.tier1
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.analysis not implemented yet")
def test_tc_f09_04_heuristic_hook_title_fallback():
    """TC-F09-04: Heuristic fallback generates valid title from segment transcript."""
    seg = make_sample_viral_segment(0, 30000)
    transcript = make_sample_transcript(30000, "The biggest mistake founders make is ignoring early retention numbers.")
    title_fn = getattr(analysis, "generate_hook_title", lambda s, t: analysis.extract_viral_segments(t, max_clips=1, llm_model="heuristic")[0].hook_title)
    title = title_fn(seg, transcript)
    assert len(title.split()) >= 3
    assert len(title) <= 64


@pytest.mark.tier1
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.analysis not implemented yet")
def test_tc_f09_05_hook_title_preserved_in_segment():
    """TC-F09-05: Generated hook title is preserved in segment metadata."""
    transcript = make_sample_transcript(30000, "The greatest productivity secret you never knew.")
    results = analysis.extract_viral_segments(transcript, max_clips=1, llm_model="heuristic")
    assert len(results) >= 1
    assert hasattr(results[0], "hook_title")
    assert len(results[0].hook_title.strip()) > 0
