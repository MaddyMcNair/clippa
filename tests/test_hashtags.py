"""
tests/test_hashtags.py - Unit Tests for Hashtag & Shorts SEO Generation
"""

import pytest
from clippa.hashtags import (
    clean_hashtag,
    extract_hashtags,
    format_youtube_shorts_title,
    format_youtube_description,
)


def test_clean_hashtag():
    assert clean_hashtag("shorts") == "#Shorts"
    assert clean_hashtag("#viral") == "#Viral"
    assert clean_hashtag("Serena Williams!") == "#SerenaWilliams"
    assert clean_hashtag("120mph") == "#120mph"
    assert clean_hashtag("   ") == ""


def test_extract_hashtags_core_tags():
    tags = extract_hashtags("Can I Return a 120 MPH Serve From Serena Williams?")
    assert "#Shorts" in tags
    assert any("Serena" in t for t in tags)
    assert any("Williams" in t or "Tennis" in t or "Serve" in t for t in tags)
    assert len(tags) <= 8


def test_extract_hashtags_with_dialogue():
    title = "Testing Tennis Serves"
    dialogue = "This is an insane 120 miles per hour tennis ball serve challenge."
    tags = extract_hashtags(title=title, text=dialogue, max_tags=6)
    assert "#Shorts" in tags
    assert len(tags) <= 6


def test_format_youtube_shorts_title():
    t1 = "Can I Return a 120 MPH Serve?"
    res1 = format_youtube_shorts_title(t1)
    assert "#Shorts" in res1
    assert len(res1) <= 100

    # Already has #Shorts
    t2 = "Epic Tennis Moment #Shorts"
    res2 = format_youtube_shorts_title(t2)
    assert res2.count("#Shorts") == 1


def test_format_youtube_description():
    desc = format_youtube_description(
        hook_title="Can I Return a 120 MPH Serve?",
        source_title="Can I Return a 120 MPH Serve From Serena Williams?",
        hashtags=["#Shorts", "#Viral", "#Tennis"],
    )
    assert "Can I Return a 120 MPH Serve?" in desc
    assert "Original Video:" in desc
    assert "#Shorts" in desc
    assert "#Tennis" in desc
