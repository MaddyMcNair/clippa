"""
clippa/hashtags.py - Intelligent Hashtag & YouTube Shorts SEO Metadata Generator

Generates high-engagement, trending hashtags and optimized title/description
metadata for YouTube Shorts based on video titles, entities, and dialogue.
"""

from __future__ import annotations

import re
from typing import List, Optional, Set


# Core viral hashtags recognized by YouTube Shorts recommendation algorithms
CORE_SHORTS_TAGS = ["#Shorts", "#Viral", "#Trending", "#FYP", "#YouTubeShorts"]

# Stop words to ignore during hashtag extraction
STOP_WORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's",
    "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
    "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves", "just", "like", "get", "got", "video", "clip"
}


def clean_hashtag(tag: str) -> str:
    """Sanitizes text into a clean CamelCase hashtag."""
    tag = tag.strip().strip("#")
    # Remove all non-alphanumeric characters
    tag = re.sub(r"[^\w]", "", tag)
    if not tag:
        return ""
    # Ensure starts with uppercase if purely alphabetic
    if tag.islower():
        tag = tag.capitalize()
    return f"#{tag}"


def extract_hashtags(
    title: str = "",
    text: str = "",
    category: str = "",
    max_tags: int = 8,
) -> List[str]:
    """
    Extracts high-impact hashtags tailored for YouTube Shorts.
    Combines core viral tags with topic-specific entities from the title and transcript.
    """
    tags: List[str] = ["#Shorts"]

    # 1. Extract proper nouns / multi-word entities from title
    combined_source = f"{title} {category}"
    entities = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", combined_source)
    for ent in entities:
        words = ent.split()
        if len(words) <= 3 and not any(w.lower() in STOP_WORDS for w in words):
            clean_tag = clean_hashtag("".join(w.capitalize() for w in words))
            if clean_tag and clean_tag not in tags and len(clean_tag) > 2:
                tags.append(clean_tag)

    # 2. Extract salient topic keywords from title
    for word in re.findall(r"\b[a-zA-Z]{4,}\b", title):
        low = word.lower()
        if low not in STOP_WORDS:
            tag = clean_hashtag(word)
            if tag and tag not in tags and len(tag) > 2:
                tags.append(tag)

    # 3. Add core viral tags if space permits
    for core in CORE_SHORTS_TAGS:
        if core not in tags and len(tags) < max_tags:
            tags.append(core)

    # 4. Extract keywords from dialogue if still under max_tags
    if len(tags) < max_tags and text:
        dialogue_words = re.findall(r"\b[A-Za-z]{5,}\b", text)
        freq: dict[str, int] = {}
        for w in dialogue_words:
            low = w.lower()
            if low not in STOP_WORDS:
                freq[low] = freq.get(low, 0) + 1
        top_words = sorted(freq.keys(), key=lambda k: freq[k], reverse=True)
        for w in top_words:
            tag = clean_hashtag(w)
            if tag and tag not in tags and len(tags) < max_tags:
                tags.append(tag)

    return tags[:max_tags]


def format_youtube_shorts_title(hook_title: str, max_chars: int = 100) -> str:
    """
    Formats YouTube Shorts video title to fit within limit and include #Shorts.
    """
    try:
        from clippa.analysis import sanitize_hook_title
        t = sanitize_hook_title(hook_title)
    except Exception:
        t = (hook_title or "").strip()

    if not t:
        t = "Epic Moment #Shorts"

    if "#Shorts" not in t and "#shorts" not in t.lower():
        if len(t) + 9 <= max_chars:
            t = f"{t} #Shorts"
        elif len(t) > max_chars - 9:
            t = t[:max_chars - 10].rstrip(".,?!;:-_ ") + " #Shorts"

    return t[:max_chars]


def format_youtube_description(
    hook_title: str,
    source_title: str = "",
    hashtags: Optional[List[str]] = None,
    channel_credit: str = "",
) -> str:
    """
    Builds clean, high-CTR YouTube Shorts description with attribution and hashtags.
    """
    lines: List[str] = [hook_title.strip(), ""]

    if source_title:
        lines.append(f"Original Video: {source_title.strip()}")
    if channel_credit:
        lines.append(f"Credit: {channel_credit.strip()}")

    lines.append("Clipped and reframed to 9:16 vertical video using CLIPPA AI.")
    lines.append("")

    tags = hashtags or ["#Shorts", "#Viral", "#Trending", "#FYP"]
    tag_string = " ".join(tags)
    lines.append(tag_string)

    return "\n".join(lines)
