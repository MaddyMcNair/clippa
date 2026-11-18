"""
clippa/subtitles.py - Animated ASS Subtitle Builder with Karaoke and Hook Title Overlay

Features Covered:
- Feature 15: Animated ASS Subtitle Builder with word karaoke (\\k)
- Feature 16: Subtitle Style Templates & Local TTF Fonts
- Feature 17: Contextual Emoji Injection
- Feature 18: Burned Hook Title Overlay in Top Safe Area
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Dict, List, Optional, Union

FONTS_DIR = Path(__file__).parent.parent / "backend" / "fonts"

EMOJI_KEYWORD_MAP: Dict[str, str] = {
    # money / wealth
    "money": "💰",
    "cash": "💵",
    "dollar": "💵",
    "dollars": "💵",
    "rich": "🤑",
    "wealth": "💰",
    "wealthy": "💰",
    "millionaire": "🤑",
    "billionaire": "🤑",
    "profit": "📈",
    "revenue": "📈",
    "income": "💵",
    "invest": "📊",
    "stock": "📈",
    "crypto": "🪙",
    "bitcoin": "₿",
    "business": "💼",
    # hype / emotions
    "fire": "🔥",
    "lit": "🔥",
    "warning": "⚠️",
    "danger": "⚠️",
    "boom": "💥",
    "rocket": "🚀",
    "boost": "🚀",
    "win": "🏆",
    "winner": "🥇",
    "won": "🏆",
    "success": "🏆",
    "goal": "🎯",
    "best": "🥇",
    "power": "⚡",
    "strong": "💪",
    # ideas / mind
    "idea": "💡",
    "ideas": "💡",
    "secret": "🤫",
    "smart": "🧠",
    "brain": "🧠",
    "think": "🤔",
    "truth": "💯",
    "question": "❓",
    "love": "❤️",
}

TEMPLATE_PRESETS: Dict[str, Dict[str, Any]] = {
    "default": {
        "font_name": "THEBOLDFONT",
        "font_size": 60,
        "primary_color": "&H00FFFFFF",     # White
        "secondary_color": "&H0000FFFF",   # Yellow highlight
        "outline_color": "&H00000000",     # Black outline
        "back_color": "&H80000000",        # 50% Shadow
        "bold": -1,
        "outline": 3,
        "shadow": 0,
        "alignment": 2,                    # Bottom center
        "margin_v": 240,
        "words_per_line": 4,
        "uppercase": False,
    },
    "hormozi": {
        "font_name": "THEBOLDFONT",
        "font_size": 60,
        "primary_color": "&H00FFFFFF",     # White
        "secondary_color": "&H0066FF00",   # Bright green highlight
        "outline_color": "&H00000000",     # Black outline
        "back_color": "&H80000000",
        "bold": -1,
        "outline": 4,
        "shadow": 0,
        "alignment": 2,
        "margin_v": 240,
        "words_per_line": 3,
        "uppercase": True,
    },
    "mrbeast": {
        "font_name": "THEBOLDFONT",
        "font_size": 64,
        "primary_color": "&H0000FFFF",     # Yellow text
        "secondary_color": "&H002D2DFF",   # Red highlight
        "outline_color": "&H00000000",     # Heavy black outline
        "back_color": "&H80000000",
        "bold": -1,
        "outline": 5,
        "shadow": 0,
        "alignment": 2,
        "margin_v": 240,
        "words_per_line": 3,
        "uppercase": True,
    },
    "tiktok": {
        "font_name": "TikTokSans-Bold",
        "font_size": 56,
        "primary_color": "&H00FFFFFF",     # White text
        "secondary_color": "&H00552CFE",   # TikTok pink
        "outline_color": "&H00000000",
        "back_color": "&H80000000",
        "bold": -1,
        "outline": 3,
        "shadow": 0,
        "alignment": 2,
        "margin_v": 240,
        "words_per_line": 4,
        "uppercase": False,
    },
}


def get_available_templates() -> List[str]:
    """Returns list of supported subtitle style templates."""
    return list(TEMPLATE_PRESETS.keys())


def ms_to_ass_time(ms: int) -> str:
    """Converts milliseconds to ASS timestamp format H:MM:SS.cs."""
    ms = max(0, ms)
    cs = int((ms % 1000) / 10)
    total_seconds = ms // 1000
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours}:{minutes:02d}:{seconds:02d}.{cs:02d}"


def sanitize_ass_text(text: str) -> str:
    """Strips ASS control characters to prevent format injection."""
    text = re.sub(r"[\{\}\\]", "", str(text or ""))
    return text.replace("\n", " ").replace("\r", " ").strip()


def resolve_font_name(font_name: str) -> str:
    """Verifies local font existence in backend/fonts/ or falls back to Arial."""
    if not FONTS_DIR.exists():
        return "Arial"
    ttf_path = FONTS_DIR / f"{font_name}.ttf"
    if ttf_path.exists():
        return font_name
    return "Arial"


def wrap_hook_title(title: str, max_chars: int = 32) -> str:
    """Wraps long hook title into two balanced lines with \\N."""
    words = title.split()
    if len(title) <= max_chars or len(words) < 2:
        return title

    # If title has a colon, break right after the colon if both sides fit
    if ":" in title:
        colon_idx = title.find(":")
        left = title[:colon_idx + 1].strip()
        right = title[colon_idx + 1:].strip()
        if left and right and len(left) <= max_chars and len(right) <= max_chars:
            return f"{left}\\N{right}"

    # Find the most balanced split
    best_split = len(words) // 2
    best_diff = 999
    for i in range(1, len(words)):
        w1 = " ".join(words[:i])
        w2 = " ".join(words[i:])
        diff = abs(len(w1) - len(w2))
        # Penalize if line 1 ends with dangling preposition or conjunction
        if words[i - 1].lower().rstrip(".,?!;:") in {"a", "an", "the", "to", "of", "in", "for", "on", "at", "by", "from", "with", "and", "or"}:
            diff += 8
        if diff < best_diff:
            best_diff = diff
            best_split = i

    return " ".join(words[:best_split]) + "\\N" + " ".join(words[best_split:])


def inject_emojis(words: List[Any]) -> List[Any]:
    """
    Contextually injects emojis next to trigger words using word boundaries.
    Limits injection to max 1 emoji per line/chunk.
    """
    result: List[Any] = []
    emoji_added = False

    for w in words:
        w_copy = copy.copy(w)
        text = getattr(w_copy, "text", "")
        clean_text = sanitize_ass_text(text).lower()

        if not emoji_added:
            for kw, emoji in EMOJI_KEYWORD_MAP.items():
                if re.search(rf"\b{re.escape(kw)}\b", clean_text):
                    if hasattr(w_copy, "text"):
                        w_copy.text = f"{text} {emoji}"
                    emoji_added = True
                    break

        result.append(w_copy)

    return result


def generate_ass_subtitles(
    words: List[Any],
    hook_title: Optional[str] = None,
    template: str = "default",
    target_w: int = 1080,
    target_h: int = 1920,
    output_path: Optional[Union[str, Path]] = None,
    enable_emojis: bool = True,
    clip_duration_s: Optional[float] = None,
) -> Path:
    """
    Generates a valid .ass subtitle file with word karaoke timing tags,
    selected template styles, contextual emojis, and top hook title overlay.
    """
    style_def = TEMPLATE_PRESETS.get(template.lower(), TEMPLATE_PRESETS["default"]).copy()
    font_name = resolve_font_name(style_def["font_name"])

    # Scale font size and vertical margin based on resolution
    scale_factor = target_w / 1080.0
    font_size = max(24, int(style_def["font_size"] * scale_factor))
    margin_v = max(40, int(style_def["margin_v"] * (target_h / 1920.0)))
    outline_w = max(2, int(style_def["outline"] * scale_factor))

    # Headers & Styles
    lines: List[str] = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {target_w}",
        f"PlayResY: {target_h}",
        "ScaledBorderAndShadow: yes",
        "WrapStyle: 2",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        f"Style: Default,{font_name},{font_size},{style_def['primary_color']},{style_def['secondary_color']},{style_def['outline_color']},{style_def['back_color']},{style_def['bold']},0,0,0,100,100,0,0,1,{outline_w},{style_def['shadow']},{style_def['alignment']},40,40,{margin_v},1",
    ]

    has_hook = bool(hook_title and str(hook_title).strip())
    if has_hook:
        hook_font_size = max(32, int(72 * scale_factor))
        hook_margin_v = max(80, int(160 * (target_h / 1920.0)))
        lines.append(
            f"Style: Hook,{font_name},{hook_font_size},&H00FFFFFF,&H0000FFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,4,0,8,40,40,{hook_margin_v},1"
        )

    lines.extend([
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ])

    # Hook Title Overlay Event
    if has_hook:
        clean_title = sanitize_ass_text(hook_title).replace('"', "")
        wrapped_title = wrap_hook_title(clean_title)
        hook_dur = 3.5
        if clip_duration_s is not None and clip_duration_s > 0:
            hook_dur = min(float(clip_duration_s), 3.5)
        fade_ms = 200 if hook_dur >= 1.0 else min(200, int(hook_dur * 250))
        hook_start = "0:00:00.00"
        hook_end = ms_to_ass_time(int(hook_dur * 1000))
        lines.append(
            f"Dialogue: 1,{hook_start},{hook_end},Hook,,0,0,0,,{{\\fad({fade_ms},{fade_ms})}}{wrapped_title}"
        )

    # Normalize words into dictionary structures
    norm_words = []
    for w in words or []:
        text = getattr(w, "text", None) if not isinstance(w, dict) else w.get("text")
        start = getattr(w, "start", None) if not isinstance(w, dict) else w.get("start")
        end = getattr(w, "end", None) if not isinstance(w, dict) else w.get("end")
        if text and start is not None and end is not None:
            norm_words.append({"text": str(text), "start": int(start), "end": int(end)})

    # Chunk words into lines (3 to 5 words)
    chunk_size = style_def["words_per_line"]
    chunks: List[List[Dict[str, Any]]] = []
    current_chunk: List[Dict[str, Any]] = []

    for w in norm_words:
        if current_chunk:
            # Multi-second pause (>1.5s) splits events
            if w["start"] - current_chunk[-1]["end"] > 1500 or len(current_chunk) >= chunk_size:
                chunks.append(current_chunk)
                current_chunk = []
        current_chunk.append(w)
    if current_chunk:
        chunks.append(current_chunk)

    # Render each chunk as dialogue line
    for chunk in chunks:
        event_start = ms_to_ass_time(chunk[0]["start"])
        event_end = ms_to_ass_time(chunk[-1]["end"])

        emoji_injected = False
        k_tokens = []

        for w in chunk:
            dur_ms = max(50, w["end"] - w["start"])
            dur_cs = max(5, int(dur_ms / 10))
            word_str = sanitize_ass_text(w["text"])
            if style_def.get("uppercase"):
                word_str = word_str.upper()

            # Emoji injection (max 1 emoji per line)
            if enable_emojis and not emoji_injected:
                clean_token = re.sub(r"[^a-zA-Z0-9]+", "", word_str.lower())
                if clean_token in EMOJI_KEYWORD_MAP:
                    word_str = f"{word_str} {EMOJI_KEYWORD_MAP[clean_token]}"
                    emoji_injected = True

            k_tokens.append(f"{{\\k{dur_cs}}}{word_str}")

        line_payload = " ".join(k_tokens)
        lines.append(f"Dialogue: 0,{event_start},{event_end},Default,,0,0,0,,{line_payload}")

    content = "\n".join(lines) + "\n"

    # Write output
    if output_path is None:
        fd, tmp = tempfile.mkstemp(suffix=".ass", prefix="clippa_subs_")
        os.close(fd)
        output_path = Path(tmp)
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(content, encoding="utf-8")
    return output_path
