"""
tests/tier1_features/test_title_grammar_and_semantics.py
Verification suite for video title grammar, semantic completeness, and sensible phrasing.
"""

import pytest
from pathlib import Path
from clippa.analysis import sanitize_hook_title, generate_hook_title
from clippa.subtitles import generate_ass_subtitles, wrap_hook_title
from tests.fixtures.synthetic import make_sample_transcript, make_sample_viral_segment


@pytest.mark.tier1
def test_contraction_and_apostrophe_restoration():
    """Verifies that missing apostrophes in contractions and possessives are restored."""
    t1 = sanitize_hook_title("Why are you wearing catchers gear")
    assert "Catcher's" in t1 or "catcher's" in t1.lower()

    t2 = sanitize_hook_title("why you dont want to lose")
    assert "Don't" in t2

    t3 = sanitize_hook_title("thats what they never told you")
    assert "That's" in t3

    t4 = sanitize_hook_title("cant stop winning this game")
    assert "Can't" in t4


@pytest.mark.tier1
def test_spoken_filler_removal():
    """Verifies that conversational spoken fillers and disfluencies are stripped."""
    t1 = sanitize_hook_title("Wait, let's say, how do I stop spinning")
    assert "Wait" not in t1
    assert "How Do I Stop Spinning?" in t1

    t2 = sanitize_hook_title("Wait lets say how do I stop spinning")
    assert "Wait" not in t2
    assert "How Do I Stop Spinning?" in t2

    t3 = sanitize_hook_title("You know, which burger is the best")
    assert "You know" not in t3
    assert t3.endswith("?")


@pytest.mark.tier1
def test_subordinating_conjunction_fragment_repair():
    """Verifies that subordinate clause fragments without main clauses are repaired."""
    t1 = sanitize_hook_title("Before they move on to the next round")
    assert not t1.lower().startswith("before")
    assert t1.lower() == "they move on to the next round"
    assert t1.startswith("They Move")

    t2 = sanitize_hook_title("Because they never knew about this")
    assert not t2.lower().startswith("because")
    assert "They Never Knew About This" in t2


@pytest.mark.tier1
def test_dangling_word_elimination():
    """Verifies that trailing prepositions and conjunctions are eliminated."""
    t1 = sanitize_hook_title("Can I Return a 120 MPH Serve From")
    assert not t1.rstrip("?").lower().endswith("from")
    assert t1 == "Can I Return a 120 MPH Serve?"

    t2 = sanitize_hook_title("Why we decided to go with")
    assert not t2.rstrip("?").lower().endswith("with")


@pytest.mark.tier1
def test_proper_name_truncation_protection():
    """Verifies that cutting long titles does not leave half of a proper name or dangling preposition."""
    t1 = sanitize_hook_title("Can I Return a 120 MPH Serve From Serena Williams?")
    assert not t1.rstrip("?").lower().endswith("serena")
    assert not t1.rstrip("?").lower().endswith("from")
    assert t1 == "Can I Return a 120 MPH Serve?"


@pytest.mark.tier1
def test_smart_title_casing_and_punctuation():
    """Verifies Title Case capitalization and question mark placement."""
    t1 = sanitize_hook_title("you say you want to battle Serena Williams 3")
    assert t1.startswith("You")
    assert "Serena Williams" in t1
    assert not t1.rstrip("?").endswith("3")
    assert t1.endswith("?")

    t2 = sanitize_hook_title("which burger is the best one")
    assert t2 == "Which Burger Is the Best One?"

    t3 = sanitize_hook_title("why are you wearing catchers gear")
    assert t3 == "Why Are You Wearing Catcher's Gear?"


@pytest.mark.tier1
def test_colon_context_separation():
    """Verifies run-on titles between entity context and dialogue are separated cleanly."""
    t1 = sanitize_hook_title("Beat Professional Athlete: Why are you wearing catchers gear")
    assert "Beat Professional Athlete: Why Are You Wearing Catcher's Gear?" in t1
    assert ":" in t1
    assert t1.endswith("?")


@pytest.mark.tier1
def test_subtitle_overlay_preserves_apostrophe(tmp_path):
    """Verifies that the generated ASS subtitle file retains apostrophes in the hook overlay."""
    ass_file = tmp_path / "test_hook.ass"
    words = [{"text": "Hello", "start": 0, "end": 1000}]
    generate_ass_subtitles(
        words=words,
        hook_title="Why Are You Wearing Catcher's Gear?",
        output_path=ass_file,
    )
    assert ass_file.exists()
    content = ass_file.read_text(encoding="utf-8")
    assert "Catcher's" in content
    assert "Catchers" not in content
