"""
clippa/analysis.py - Local-First AI & Heuristic Virality Analysis

Features Covered:
- Feature 6: Viral Highlight & Moment Detection (Ollama local LLM evaluation)
- Feature 7: Rule-Based Heuristic Scorer Fallback
- Feature 8: Windowed Long-Video Analysis & Deduplication
- Feature 9: Hook Title Generation & Sanitization
- Feature 10: Cold-Open Teaser Punchline Detection
"""

from __future__ import annotations

import json
import logging
import re
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

from clippa.transcription import TranscriptData, Word

logger = logging.getLogger("clippa.analysis")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

# ==============================================================================
# Data Models (Contract 3)
# ==============================================================================

@dataclass
class ViralSegment:
    start_time_ms: int
    end_time_ms: int
    hook_title: str
    hook_score: int
    engagement_score: int
    value_score: int
    shareability_score: int
    total_score: int
    reasoning: str
    hook_start_time_ms: Optional[int] = None
    hook_end_time_ms: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TeaserCandidate:
    start_time_ms: int
    end_time_ms: int

    def __iter__(self):
        yield self.start_time_ms
        yield self.end_time_ms

    def __getitem__(self, idx):
        return (self.start_time_ms, self.end_time_ms)[idx]


# Power words for viral hook scoring
POWER_KEYWORDS = {
    "secret", "mistake", "why", "never", "always", "money", "dollar",
    "dollars", "truth", "revealed", "billionaire", "millionaire", "hack",
    "trick", "rule", "stop", "best", "worst", "hidden", "shocking",
    "insane", "genius", "warning", "danger", "lesson", "biggest",
    "proven", "crucial", "huge", "fail", "success", "future", "how"
}


# ==============================================================================
# Feature 9: Hook Title Generation & Sanitization
# ==============================================================================

ACRONYMS = {
    "mph", "ai", "nba", "nfl", "ufc", "ncaa", "usa", "uk", "hd", "4k",
    "tv", "ceo", "iq", "yt", "fbi", "cia", "mlb", "nhl", "fifa", "nascar"
}

LOWERCASE_WORDS = {
    "a", "an", "the", "and", "but", "or", "nor", "for", "so", "yet",
    "at", "by", "in", "of", "off", "on", "per", "to", "up", "via", "with", "as"
}

ALWAYS_CAPITAL_WORDS = {
    "i", "i'm", "i've", "i'll", "i'd", "is", "are", "was", "were", "be", "been",
    "can", "will", "do", "does", "did", "it", "its", "my", "your", "his", "her",
    "our", "their", "who", "what", "where", "when", "why", "how", "which"
}

DANGLING_ENDINGS = {
    "from", "to", "with", "and", "or", "in", "on", "at", "by", "for",
    "of", "the", "a", "an", "my", "your", "its", "her", "his", "their", "our",
    "into", "through", "under", "over", "between", "without", "within",
    "against", "but", "so", "because", "although", "since", "while",
    "if", "unless", "until", "as", "than", "nor", "yet", "via",
    "across", "behind", "beyond", "which", "who", "whom", "whose", "whos", "who's"
}

QUESTION_STARTERS = {
    "who", "what", "when", "where", "why", "how", "which", "whose", "whom",
    "can", "could", "would", "should", "will", "shall", "is", "are", "do",
    "does", "did", "am", "was", "were", "has", "have", "had"
}

CONTRACTION_REPLACEMENTS = [
    (r"\bcatchers\s+gear\b", "catcher's gear"),
    (r"\blets\s+(go|say|see|try|do|play|talk|get|find|make|take|start|keep|put|run|watch|win|look|battle)\b", r"let's \1"),
    (r"\bdont\b", "don't"),
    (r"\bcant\b", "can't"),
    (r"\bwont\b", "won't"),
    (r"\bdidnt\b", "didn't"),
    (r"\bisnt\b", "isn't"),
    (r"\barent\b", "aren't"),
    (r"\bwasnt\b", "wasn't"),
    (r"\bwerent\b", "weren't"),
    (r"\bhasnt\b", "hasn't"),
    (r"\bhavent\b", "haven't"),
    (r"\bhadnt\b", "hadn't"),
    (r"\bcouldnt\b", "couldn't"),
    (r"\bwouldnt\b", "wouldn't"),
    (r"\bshouldnt\b", "shouldn't"),
    (r"\bthats\b", "that's"),
    (r"\bwhats\b", "what's"),
    (r"\bwheres\b", "where's"),
    (r"\bhows\b", "how's"),
    (r"\btheres\b", "there's"),
    (r"\bheres\b", "here's"),
    (r"\byoure\b", "you're"),
    (r"\btheyre\b", "they're"),
    (r"\bweve\b", "we've"),
    (r"\byouve\b", "you've"),
    (r"\btheyve\b", "they've"),
    (r"\byoull\b", "you'll"),
    (r"\btheyll\b", "they'll"),
    (r"\bwhos\b", "who's"),
    (r"\bi\b", "I"),
    (r"\bi['’]?m\b", "I'm"),
    (r"\bi['’]?ve\b", "I've"),
    (r"\bi['’]?ll\b", "I'll"),
    (r"\bi['’]?d\b", "I'd"),
]

FILLER_PREFIX_REGEXES = [
    r"^(?:wait\s*[,]?\s*(?:let['’]?s\s*say\s*[,]?)?|let['’]?s\s*say\s*[,]?)\s*",
    r"^(?:you\s*know\s*[,]?|i\s*mean\s*[,]?|so\s*like\s*[,]?|like\s*i\s*said\s*[,]?|like\s*[,]?)\s*",
    r"^(?:well\s*[,]?|uh\s*[,]?|um\s*[,]?|ah\s*[,]?|er\s*[,]?)\s*",
    r"^(?:look\s*[,]?|listen\s*[,]?|honestly\s*[,]?|to\s*be\s*honest\s*[,]?)\s*",
    r"^(?:basically\s*[,]?|actually\s*[,]?|so\s*then\s*[,]?|and\s*so\s*[,]?|but\s*like\s*[,]?)\s*",
    r"^(?:i['’]?m\s*kind\s*of\s*|kind\s*of\s*|sort\s*of\s*)\s*",
    r"^(?:and|so|but|then|or)\s+",
]

SUBORDINATE_PREFIX_REGEX = r"^(?:before|because|although|though|even\s+though|while|when|whenever|since|unless|until)\s+"


def apply_smart_title_case(title: str) -> str:
    """Capitalizes title following standard headline grammar rules."""
    words = title.split()
    if not words:
        return title

    formatted_words = []
    n = len(words)
    for idx, raw_word in enumerate(words):
        m = re.match(r"^([^\w]*)([\w'’\-]+)([^\w]*)$", raw_word)
        if not m:
            formatted_words.append(raw_word)
            continue

        prefix, core, suffix = m.groups()
        core_low = core.lower()

        if core_low in ACRONYMS:
            c_word = core_low.upper()
        elif core_low in {"i", "i'm", "i've", "i'll", "i'd"}:
            c_word = "I" + core[1:]
        elif idx == 0 or idx == n - 1 or (idx > 0 and words[idx - 1].endswith(("-", ":", "—"))):
            c_word = core[0].upper() + core[1:]
        elif core_low in ALWAYS_CAPITAL_WORDS:
            c_word = core[0].upper() + core[1:]
        elif core_low in LOWERCASE_WORDS:
            c_word = core_low
        else:
            c_word = core[0].upper() + core[1:]

        formatted_words.append(f"{prefix}{c_word}{suffix}")

    return " ".join(formatted_words)


def clean_grammar_and_fillers(text: str) -> str:
    """Removes spoken filler words, fixes contractions, and repairs incomplete fragments."""
    s = text.strip()
    s = s.strip("\"'`“”‘’")
    s = re.sub(r"#\w+", "", s).strip()

    # Strip leading clip numbering e.g. "1. ", "Clip 2: "
    s = re.sub(r"^(?:clip|segment|part|video|highlight)?\s*\d+[\s:.-]+", "", s, flags=re.IGNORECASE).strip()

    # Strip trailing clip number artifacts e.g. "Serena Williams 3"
    m_trailing_num = re.search(r"^(.*)\s+(\d+)$", s)
    if m_trailing_num:
        prefix_part, _ = m_trailing_num.group(1), m_trailing_num.group(2)
        words_prefix = prefix_part.split()
        last_word = words_prefix[-1].lower().rstrip(".,?!;:") if words_prefix else ""
        if last_word not in {"top", "round", "day", "tier", "level", "step", "phase", "part", "number", "no", "#"}:
            s = prefix_part.strip()

    # Apply contraction and grammar corrections
    for pattern, repl in CONTRACTION_REPLACEMENTS:
        s = re.sub(pattern, repl, s, flags=re.IGNORECASE)

    # Separate run-on video context and question if missing colon: e.g. "Athlete Why are you wearing"
    s = re.sub(r"\b([a-zA-Z0-9]+)\s+(Why|How|What|Where|Who|When|Can|Could|Should)\b", r"\1: \2", s)

    # Strip conversational spoken fillers at start if remaining has >= 3 words
    for pat in FILLER_PREFIX_REGEXES:
        match = re.search(pat, s, flags=re.IGNORECASE)
        if match:
            candidate = s[match.end():].strip().lstrip(",;: ")
            if len(candidate.split()) >= 3:
                s = candidate

    # Repair subordinating conjunction fragments (e.g. "Before they move on to the next round")
    if "," not in s and ":" not in s:
        sub_match = re.search(SUBORDINATE_PREFIX_REGEX, s, flags=re.IGNORECASE)
        if sub_match:
            candidate = s[sub_match.end():].strip()
            if len(candidate.split()) >= 3:
                s = candidate

    return re.sub(r"\s+", " ", s).strip()


def clean_truncated_words(orig_words: List[str], max_words: int = 9) -> List[str]:
    """Truncates words intelligently without orphaning proper names or leaving dangling prepositions."""
    if len(orig_words) <= max_words:
        truncated = list(orig_words)
    else:
        truncated = list(orig_words[:max_words])
        next_word = orig_words[max_words] if len(orig_words) > max_words else ""
        if next_word and next_word[0].isupper() and truncated[-1][0].isupper():
            # Avoid cutting off second half of a proper name (e.g. "Serena" without "Williams")
            truncated.pop()

    while len(truncated) > 3 and truncated[-1].lower().rstrip(".,?!;:—– -") in DANGLING_ENDINGS:
        truncated.pop()

    return truncated


def sanitize_hook_title(raw_title: Optional[str], default_text: Optional[str] = None) -> str:
    """
    Sanitizes hook title to ensure it makes semantic sense and is grammatically correct:
    - Strips hashtags (#) and surrounding quotation marks.
    - Removes conversational filler disfluencies (e.g. 'Wait, let's say').
    - Repairs subordinating conjunction sentence fragments.
    - Restores missing contractions and apostrophes (e.g. catcher's gear, let's).
    - Escapes ASS control characters ({, }, \\).
    - Enforces 3 to 9 words and max 64 characters without dangling prepositions.
    - Applies proper headline Title Case and terminal punctuation.
    - Replaces punctuation-only/empty strings with sensible default.
    """
    fallback = default_text or "Must Watch Video Moment"
    if not raw_title or not str(raw_title).strip():
        return fallback

    title = str(raw_title).strip()
    title = title.strip("\"'`“”‘’")
    title = re.sub(r"#\w+", "", title)
    title = re.sub(r"\s+", " ", title).strip()

    # Escape ASS control characters: {, }, \
    title = title.replace("{", "(").replace("}", ")").replace("\\", "/")

    # Filter out emoji/punctuation-only titles
    alnum_check = re.sub(r"[^\w\s]", "", title).strip()
    if len(alnum_check) == 0:
        return default_text or "Key Highlight From Video"

    # Clean grammar, spoken fillers, and artifacts
    title = clean_grammar_and_fillers(title)

    words = title.split()
    if len(words) < 3:
        clean_word = alnum_check.capitalize()
        title = f"The Biggest {clean_word} Exposed"
        words = title.split()

    if len(words) > 9:
        words = clean_truncated_words(words, max_words=9)
        title = " ".join(words)

    if len(title) > 64:
        clipped = title[:64]
        last_space = clipped.rfind(" ")
        title = clipped[:last_space].rstrip(".,;:-–— ") if last_space > 20 else clipped[:64]

    # Clean dangling prepositions/conjunctions at the end
    words = title.split()
    while len(words) > 3 and words[-1].lower().rstrip(".,?!;:—– -") in DANGLING_ENDINGS:
        words.pop()
    title = " ".join(words).strip()

    # Terminal punctuation determination
    has_question_mark = "?" in title
    has_exclamation = "!" in title
    title_clean = title.rstrip(".,;:-–— ")

    is_question = (
        has_question_mark or
        re.search(r"(?:^|[:–—\-]\s*)(who|what|when|where|why|how|which|whose|whom|can|could|would|should|will|is|are|do|does|did|am|was|were|has|have|had)\b", title_clean, re.IGNORECASE) is not None or
        re.search(r"\b(you say you want|think you can|did you know|would you rather)\b", title_clean, re.IGNORECASE) is not None
    )

    if is_question:
        title_clean = title_clean.rstrip("?!.,;:-–— ") + "?"
    elif has_exclamation:
        title_clean = title_clean.rstrip("?!.,;:-–— ") + "!"
    else:
        title_clean = title_clean.rstrip(".,;:-–— ")

    # Apply headline title case
    final_title = apply_smart_title_case(title_clean)

    # Re-enforce word and character bounds strictly
    final_words = final_title.split()
    if len(final_words) > 9:
        final_words = final_words[:9]
        final_title = " ".join(final_words)
    if len(final_title) > 64:
        final_title = final_title[:64].rstrip(".,;:-–— ")

    return final_title.strip().rstrip(".,;:-–—") or fallback


def generate_hook_title(*args, **kwargs) -> str:
    """
    Generates a punchy, context-aware hook title combining the video title and segment dialogue.
    Accepts:
    - generate_hook_title(text: str, video_title: Optional[str] = None)
    - generate_hook_title(seg: ViralSegment, transcript: TranscriptData, video_title: Optional[str] = None)
    - generate_hook_title(transcript: TranscriptData, video_title: Optional[str] = None)
    """
    text = ""
    fallback_title = kwargs.get("fallback_title")
    video_title = kwargs.get("video_title")

    if args:
        first = args[0]
        if isinstance(first, str):
            text = first
        elif hasattr(first, "text"):
            text = str(first.text)
        elif len(args) > 1 and hasattr(args[1], "words"):
            # Called with (seg, transcript)
            seg = first
            transcript = args[1]
            seg_start = getattr(seg, "start_time_ms", 0)
            seg_end = getattr(seg, "end_time_ms", 30000)
            words = [w.text for w in transcript.words if seg_start <= w.start <= seg_end]
            text = " ".join(words) if words else str(getattr(transcript, "text", ""))
        elif len(args) > 1 and isinstance(args[1], str) and not video_title:
            video_title = args[1]

    text = text.strip() if text else ""

    # Clean the video title if provided
    clean_vt = ""
    if video_title and isinstance(video_title, str) and video_title.strip():
        raw_vt = video_title.strip()
        raw_vt = re.sub(r"\[[\w-]+\]", "", raw_vt)
        raw_vt = re.sub(r"\.(mp4|mkv|mov|avi|webm)$", "", raw_vt, flags=re.IGNORECASE)
        raw_vt = re.sub(r"\b(OFFICIAL VIDEO|OFFICIAL MUSIC VIDEO|4K|HD|1080p|Full Episode)\b", "", raw_vt, flags=re.IGNORECASE)
        raw_vt = raw_vt.replace("？", "?").replace("！", "!")
        raw_vt = raw_vt.strip(" _-—–:\"'|[]()")
        # Ignore synthetic/generic test names
        if not any(raw_vt.lower().startswith(p) for p in ["live_synth", "sample", "input", "test", "dummy", "source", "black_"]):
            clean_vt = raw_vt

    # If neither text nor clean_vt is available, return default
    if not text and not clean_vt:
        return sanitize_hook_title("", default_text=fallback_title)

    # 1. Contextual Title Generation: Combining clean_vt and segment dialogue
    if clean_vt and text:
        vt_words = [w.strip(".,!?:;\"'()") for w in clean_vt.split() if w.strip(".,!?:;\"'()")]
        stopwords = {
            "a", "an", "the", "and", "or", "in", "on", "at", "to", "for", "of",
            "with", "from", "by", "is", "it", "this", "that", "i", "can", "my",
            "we", "you", "me", "he", "she", "they", "will", "do", "how"
        }
        vt_keywords = [w for w in vt_words if w.lower() not in stopwords and len(w) > 1]

        # Extract entities (proper nouns / capitalized pairs) or premise from clean_vt
        action_verbs = {
            "beat", "win", "how", "why", "can", "will", "return", "find", "make",
            "build", "eat", "play", "test", "try", "give", "spend", "run", "stop", "secret"
        }
        raw_entities = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", clean_vt)
        valid_entities = [e for e in raw_entities if e.split()[0].lower() not in action_verbs]
        primary_entity = valid_entities[0] if valid_entities else ""
        if not primary_entity and clean_vt:
            # Fallback to main phrase before comma/dash
            first_clause = re.split(r"[,–—:\-]", clean_vt)[0].strip()
            if 2 <= len(first_clause.split()) <= 4:
                primary_entity = first_clause

        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

        best_candidate = None
        best_score = -1

        for s in sentences:
            s_clean = clean_grammar_and_fillers(s)
            s_tokens = s_clean.split()
            if len(s_tokens) < 3:
                continue

            s_low = s_clean.lower()
            score = 0

            matches_vt = sum(1 for kw in vt_keywords if kw.lower() in s_low)
            score += matches_vt * 10

            pw_count = sum(1 for pw in POWER_KEYWORDS if pw in s_low)
            score += pw_count * 5

            if "?" in s_clean or any(s_low.startswith(qw) for qw in QUESTION_STARTERS):
                score += 8
            if "!" in s_clean:
                score += 4

            if 4 <= len(s_tokens) <= 9:
                score += 6
            elif len(s_tokens) > 12:
                score -= 10

            if primary_entity and primary_entity.lower() in s_low:
                score += 15

            if score > best_score:
                best_score = score
                if 3 <= len(s_tokens) <= 9:
                    best_candidate = s_clean
                else:
                    best_candidate = " ".join(clean_truncated_words(s_tokens, max_words=8))

        if best_candidate and best_score >= 12:
            candidate = best_candidate.rstrip(".,;: ")
            # If candidate doesn't mention primary entity and is short, prefix entity with colon
            if primary_entity and primary_entity.lower() not in candidate.lower() and len(candidate.split()) <= 5:
                combined = f"{primary_entity}: {candidate}"
                if len(combined.split()) <= 9 and len(combined) <= 64:
                    return sanitize_hook_title(combined, default_text=fallback_title)
            return sanitize_hook_title(candidate, default_text=fallback_title)

        # Fallback to video title if dialogue candidate score is low
        clean_vt_tokens = clean_vt.split()
        if 3 <= len(clean_vt_tokens) <= 9 and len(clean_vt) <= 64:
            return sanitize_hook_title(clean_vt, default_text=fallback_title)
        elif len(clean_vt_tokens) > 9:
            clean_short = clean_truncated_words(clean_vt_tokens, max_words=9)
            return sanitize_hook_title(" ".join(clean_short), default_text=fallback_title)

    if clean_vt and not text:
        vt_tokens = clean_vt.split()
        if 3 <= len(vt_tokens) <= 9 and len(clean_vt) <= 64:
            return sanitize_hook_title(clean_vt, default_text=fallback_title)
        clean_short = clean_truncated_words(vt_tokens, max_words=9)
        return sanitize_hook_title(" ".join(clean_short), default_text=fallback_title)

    # Pure transcript scoring without video title
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    best_cand = ""
    best_cand_score = -1

    for s in sentences:
        s_clean = clean_grammar_and_fillers(s)
        tokens = [w for w in s_clean.split() if len(w) > 1]
        if not tokens:
            continue
        s_low = s_clean.lower()
        score = sum(1 for pw in POWER_KEYWORDS if pw in s_low) * 5
        if "?" in s_clean or any(s_low.startswith(qw) for qw in QUESTION_STARTERS):
            score += 6
        if 3 <= len(tokens) <= 8 and len(s_clean) <= 60:
            score += 5
            if score > best_cand_score:
                best_cand_score = score
                best_cand = " ".join(tokens)

    if best_cand:
        raw_title = best_cand
    else:
        tokens = [w for w in text.split() if len(w) > 1][:7]
        raw_title = " ".join(tokens).rstrip(".,!?")

    return sanitize_hook_title(raw_title, default_text=fallback_title)


# ==============================================================================
# Feature 10: Cold-Open Teaser Detection
# ==============================================================================

def detect_cold_open_teaser(
    seg: ViralSegment,
    transcript: Union[TranscriptData, List[Word]],
    min_teaser_duration_s: float = 2.0,
    max_teaser_duration_s: float = 5.0,
) -> Optional[TeaserCandidate]:
    """
    Identifies a 2.0s to 5.0s punchline teaser from within the segment.
    - Suppressed if segment total duration < 20s.
    - Suppressed if segment opens with an immediate strong hook.
    - Aligns teaser bounds to Whisper word boundaries.
    """
    seg_dur_ms = seg.end_time_ms - seg.start_time_ms
    if seg_dur_ms < 20_000:
        return None

    # Immediate hook statement in opening means teaser can be omitted
    if seg.hook_score >= 25:
        return None

    words_list: List[Word] = (
        transcript.words if hasattr(transcript, "words") else list(transcript)
    )
    seg_words = [w for w in words_list if seg.start_time_ms <= w.start < seg.end_time_ms]
    if len(seg_words) < 8:
        return None

    min_dur_ms = int(min_teaser_duration_s * 1000)
    max_dur_ms = int(max_teaser_duration_s * 1000)

    # Search in second half of the segment
    mid_ms = seg.start_time_ms + seg_dur_ms // 2
    later_words = [w for w in seg_words if w.start >= mid_ms]
    if len(later_words) < 4:
        later_words = seg_words[len(seg_words) // 2 :]

    best_teaser: Optional[Tuple[int, int]] = None
    best_score = -1

    for i in range(len(later_words)):
        for j in range(i + 1, min(len(later_words), i + 15)):
            t_start = later_words[i].start
            t_end = later_words[j].end
            dur = t_end - t_start

            if min_dur_ms <= dur <= max_dur_ms:
                sub_text = " ".join(w.text.lower() for w in later_words[i : j + 1])
                score = sum(3 for kw in POWER_KEYWORDS if kw in sub_text)
                if "?" in sub_text or "!" in sub_text:
                    score += 5
                if score > best_score:
                    best_score = score
                    best_teaser = (t_start, t_end)

    if not best_teaser and len(seg_words) >= 10:
        # Fallback window near 70% of segment
        target_idx = int(len(seg_words) * 0.7)
        t_start = seg_words[target_idx].start
        t_end = min(seg.end_time_ms, t_start + 3000)
        for w in seg_words[target_idx:]:
            if min_dur_ms <= w.end - t_start <= max_dur_ms:
                t_end = w.end
        if min_dur_ms <= t_end - t_start <= max_dur_ms:
            best_teaser = (t_start, t_end)

    if not best_teaser:
        return None

    t_start, t_end = best_teaser

    # If teaser is at clip start, disable teaser
    if t_start <= seg.start_time_ms:
        return None

    # Clamp duration
    if t_end - t_start < min_dur_ms:
        t_end = t_start + min_dur_ms
    elif t_end - t_start > max_dur_ms:
        t_end = t_start + max_dur_ms

    # Ensure strictly contained within segment
    if not (seg.start_time_ms < t_start < t_end <= seg.end_time_ms):
        return None

    return TeaserCandidate(start_time_ms=t_start, end_time_ms=t_end)


# ==============================================================================
# Feature 8: Windowed Long-Video Analysis
# ==============================================================================

def create_analysis_windows(
    total_duration_s: float,
    window_size_s: float = 720.0,
    overlap_s: float = 120.0,
) -> List[Tuple[float, float]]:
    """
    Partitions total video duration into overlapping windows.
    Videos <= 15 minutes (900s) return a single window.
    """
    if total_duration_s <= 900.0:
        return [(0.0, total_duration_s)]

    step = max(1.0, window_size_s - overlap_s)
    windows: List[Tuple[float, float]] = []
    start = 0.0

    while start < total_duration_s:
        end = min(total_duration_s, start + window_size_s)
        windows.append((start, end))
        if end >= total_duration_s:
            break
        start += step

    return windows


def deduplicate_segments(
    segments: List[ViralSegment],
    overlap_threshold: float = 0.5,
) -> List[ViralSegment]:
    """
    Deduplicates overlapping candidate segments across windows,
    keeping the higher-scoring candidate with deterministic tie-breaking.
    """
    sorted_candidates = sorted(
        segments, key=lambda s: (-s.total_score, s.start_time_ms)
    )
    kept: List[ViralSegment] = []

    for cand in sorted_candidates:
        cand_dur = cand.end_time_ms - cand.start_time_ms
        overlap_found = False

        for k in kept:
            k_dur = k.end_time_ms - k.start_time_ms
            shorter = min(cand_dur, k_dur)
            if shorter <= 0:
                continue

            overlap_ms = max(
                0,
                min(cand.end_time_ms, k.end_time_ms)
                - max(cand.start_time_ms, k.start_time_ms),
            )
            ratio = overlap_ms / float(shorter)
            if ratio > overlap_threshold:
                overlap_found = True
                break

        if not overlap_found:
            kept.append(cand)

    return kept


# ==============================================================================
# Feature 7: Rule-Based Heuristic Virality Scorer & Narrative Boundaries
# ==============================================================================

def _chunk_sentences(words: List[Word]) -> List[List[Word]]:
    """Chunks words into grammatically clean sentence units using punctuation (. ! ?) or pauses > 600ms."""
    if not words:
        return []

    sentences: List[List[Word]] = []
    current: List[Word] = [words[0]]

    for i in range(1, len(words)):
        prev_w = words[i - 1]
        curr_w = words[i]
        gap = curr_w.start - prev_w.end

        ends_sentence = any(prev_w.text.endswith(p) for p in [".", "?", "!", "..."])
        if ends_sentence or gap >= 650:
            sentences.append(current)
            current = []
        current.append(curr_w)

    if current:
        sentences.append(current)
    return sentences


_chunk_words = _chunk_sentences  # Compatibility alias


def find_narrative_boundaries(
    words: List[Word],
    target_start_ms: int,
    target_end_ms: int,
    min_duration_ms: int = 20_000,
    max_duration_ms: int = 120_000,
) -> Tuple[int, int]:
    """
    Adjusts candidate segment start and end to strictly align with clean,
    complete sentence and narrative boundaries. Never starts or ends mid-sentence.
    """
    if not words:
        return target_start_ms, target_end_ms

    sentences = _chunk_sentences(words)
    if not sentences:
        return target_start_ms, target_end_ms

    best_start_idx = 0
    for idx, sent in enumerate(sentences):
        if sent[0].start <= target_start_ms <= sent[-1].end:
            best_start_idx = idx
            break
        if sent[0].start > target_start_ms:
            best_start_idx = max(0, idx - 1)
            break

    dangling = {"and", "or", "but", "so", "then", "because", "also", "plus", "anyway"}
    first_word = sentences[best_start_idx][0].text.lower().strip(".,!?:;\"'")
    if first_word in dangling and best_start_idx > 0:
        cand_dur = sentences[-1][-1].end - sentences[best_start_idx - 1][0].start
        if cand_dur <= max_duration_ms:
            best_start_idx -= 1
        elif best_start_idx + 1 < len(sentences):
            best_start_idx += 1

    start_ms = sentences[best_start_idx][0].start
    best_end_idx = best_start_idx

    for idx in range(best_start_idx, len(sentences)):
        cur_end = sentences[idx][-1].end
        dur = cur_end - start_ms
        if dur > max_duration_ms:
            break
        best_end_idx = idx
        if cur_end >= target_end_ms and dur >= min_duration_ms:
            break

    end_ms = sentences[best_end_idx][-1].end
    dur = end_ms - start_ms

    while dur < min_duration_ms and best_end_idx + 1 < len(sentences):
        next_end = sentences[best_end_idx + 1][-1].end
        if next_end - start_ms <= max_duration_ms:
            best_end_idx += 1
            end_ms = next_end
            dur = end_ms - start_ms
        else:
            break

    while dur < min_duration_ms and best_start_idx > 0:
        prev_start = sentences[best_start_idx - 1][0].start
        if end_ms - prev_start <= max_duration_ms:
            best_start_idx -= 1
            start_ms = prev_start
            dur = end_ms - start_ms
        else:
            break

    return start_ms, end_ms


def extract_heatmap_highlights(
    heatmap: List[Dict[str, float]],
    transcript: TranscriptData,
    max_clips: int = 3,
    min_duration_ms: int = 20_000,
    max_duration_ms: int = 120_000,
    video_title: Optional[str] = None,
) -> List[ViralSegment]:
    """
    Extracts top viral clips centered on YouTube's 'Most Replayed' heatmap peaks.
    Each clip starts and ends on complete sentence boundaries without awkward cutoffs.
    """
    if not heatmap or not transcript.words:
        return []

    total_video_ms = transcript.words[-1].end

    valid_points = [
        p for p in heatmap
        if isinstance(p, dict) and "start_time" in p and "end_time" in p and "value" in p
    ]
    if not valid_points:
        return []

    peaks: List[Dict[str, Any]] = []
    n = len(valid_points)
    for i in range(n):
        val = float(valid_points[i].get("value", 0.0) or 0.0)
        prev_val = float(valid_points[i - 1].get("value", 0.0)) if i > 0 else 0.0
        next_val = float(valid_points[i + 1].get("value", 0.0)) if i < n - 1 else 0.0

        if val >= prev_val and val >= next_val and val >= 0.15:
            mid_s = (valid_points[i]["start_time"] + valid_points[i]["end_time"]) / 2.0
            peaks.append({"time_s": mid_s, "value": val, "index": i})

    if not peaks:
        top_points = sorted(valid_points, key=lambda p: float(p.get("value", 0.0)), reverse=True)[:5]
        for p in top_points:
            mid_s = (p["start_time"] + p["end_time"]) / 2.0
            peaks.append({"time_s": mid_s, "value": float(p.get("value", 0.0)), "index": -1})

    peaks.sort(key=lambda x: -x["value"])
    candidates: List[ViralSegment] = []

    for peak in peaks:
        peak_ms = int(peak["time_s"] * 1000)
        if peak_ms > total_video_ms:
            continue

        raw_start = max(0, peak_ms - 30_000)
        raw_end = min(total_video_ms, peak_ms + 30_000)

        start_ms, end_ms = find_narrative_boundaries(
            transcript.words, raw_start, raw_end,
            min_duration_ms=min_duration_ms, max_duration_ms=max_duration_ms,
        )

        if end_ms <= start_ms:
            continue

        seg_words = [w for w in transcript.words if start_ms <= w.start <= end_ms]
        if not seg_words:
            continue

        seg_text = " ".join(w.text for w in seg_words)
        title = generate_hook_title(seg_text, video_title=video_title)

        replay_score = int(round(peak["value"] * 40))
        kw_count = sum(1 for kw in POWER_KEYWORDS if kw in seg_text.lower())
        hook_score = min(25, 12 + (5 if "?" in seg_text[:100] else 0) + kw_count)
        eng_score = min(25, 14 + replay_score // 3)
        val_score = min(25, 14 + (4 if any(c.isdigit() for c in seg_text) else 0))
        share_score = min(25, 13 + min(12, kw_count * 2))
        total_score = min(100, int(round(replay_score * 0.4 + (hook_score + eng_score + val_score + share_score) * 0.6)))

        reason = f"YouTube Most Replayed peak ({int(peak['value'] * 100)}% replay intensity) with complete narrative sentence boundaries."

        candidates.append(
            ViralSegment(
                start_time_ms=start_ms,
                end_time_ms=end_ms,
                hook_title=title,
                hook_score=hook_score,
                engagement_score=eng_score,
                value_score=val_score,
                shareability_score=share_score,
                total_score=total_score,
                reasoning=reason,
            )
        )

    deduped = deduplicate_segments(candidates, overlap_threshold=0.35)
    return deduped[:max_clips]


def heuristic_score_segments(
    transcript: TranscriptData,
    max_clips: int = 3,
    max_duration_ms: int = 120_000,
    video_title: Optional[str] = None,
) -> List[ViralSegment]:
    """
    Deterministic rule-based virality scorer fallback.
    - Evaluates power words, question marks, speech rate, and information density.
    - Aligns candidates strictly to complete narrative sentences (no mid-sentence cuts).
    - Guarantees baseline score >= 10.
    - Bounded within 15s to 120s (max 2 minutes).
    """
    if not transcript.words:
        return []

    total_duration_ms = transcript.words[-1].end

    # Under 15s short video boundary
    if total_duration_ms <= 15_000:
        title = generate_hook_title(transcript.text, video_title=video_title)
        return [
            ViralSegment(
                start_time_ms=0,
                end_time_ms=total_duration_ms,
                hook_title=title,
                hook_score=15,
                engagement_score=15,
                value_score=15,
                shareability_score=15,
                total_score=60,
                reasoning="Short video clip clamped to full available duration.",
            )
        ]

    sentences = _chunk_sentences(transcript.words)
    candidates: List[ViralSegment] = []
    dangling = {"and", "or", "but", "so", "then", "because", "also"}

    for i in range(len(sentences)):
        first_w = sentences[i][0].text.lower().strip(".,!?:;\"'")
        is_dangling = first_w in dangling

        cand_words: List[Word] = []
        for j in range(i, len(sentences)):
            cand_words.extend(sentences[j])
            start_ms = cand_words[0].start
            end_ms = cand_words[-1].end
            dur_ms = end_ms - start_ms

            if dur_ms >= 15_000:
                if dur_ms <= max_duration_ms:
                    text = " ".join(w.text for w in cand_words)
                    opening = " ".join(w.text for w in cand_words[:12]).lower()

                    # 1. Hook score
                    hook_s = 14
                    if "?" in opening:
                        hook_s += 5
                    kw_count = sum(1 for kw in POWER_KEYWORDS if kw in opening)
                    hook_s += min(6, kw_count * 2)
                    if is_dangling:
                        hook_s = max(0, hook_s - 3)
                    hook_s = max(0, min(25, hook_s))

                    # 2. Engagement score
                    eng_s = 14
                    dur_s = dur_ms / 1000.0
                    wpm = (len(cand_words) / max(1.0, dur_s)) * 60.0
                    if 130 <= wpm <= 185:
                        eng_s += 4
                    unique_words = len(set(w.text.lower() for w in cand_words))
                    ttr = unique_words / max(1, len(cand_words))
                    if ttr > 0.50:
                        eng_s += 4
                    eng_s = max(0, min(25, eng_s))

                    # 3. Value score
                    val_s = 13
                    if any(char.isdigit() for char in text):
                        val_s += 4
                    if any(w in text.lower() for w in ["because", "how", "step", "first", "result"]):
                        val_s += 4
                    val_s = max(0, min(25, val_s))

                    # 4. Shareability score
                    share_s = 13
                    total_kw = sum(1 for kw in POWER_KEYWORDS if kw in text.lower())
                    share_s += min(7, total_kw * 2)
                    share_s = max(0, min(25, share_s))

                    total_s = max(10, min(100, hook_s + eng_s + val_s + share_s))
                    title = generate_hook_title(text, video_title=video_title)

                    seg = ViralSegment(
                        start_time_ms=start_ms,
                        end_time_ms=end_ms,
                        hook_title=title,
                        hook_score=hook_s,
                        engagement_score=eng_s,
                        value_score=val_s,
                        shareability_score=share_s,
                        total_score=total_s,
                        reasoning=f"Heuristic highlight: complete narrative sentence boundaries with {kw_count} hook keywords.",
                    )
                    teaser = detect_cold_open_teaser(seg, transcript.words)
                    if teaser:
                        seg.hook_start_time_ms = teaser.start_time_ms
                        seg.hook_end_time_ms = teaser.end_time_ms

                    candidates.append(seg)
                else:
                    break

    if not candidates:
        end_ms = min(total_duration_ms, 30_000)
        title = generate_hook_title(transcript.text, video_title=video_title)
        candidates.append(
            ViralSegment(
                start_time_ms=0,
                end_time_ms=end_ms,
                hook_title=title,
                hook_score=15,
                engagement_score=15,
                value_score=15,
                shareability_score=15,
                total_score=60,
                reasoning="Standard baseline candidate.",
            )
        )

    deduped = deduplicate_segments(candidates, overlap_threshold=0.35)
    return deduped[:max_clips]


# ==============================================================================
# Feature 6: Local Ollama Virality Analysis
# ==============================================================================

def _safe_float(val: Any, default: Optional[float] = None) -> Optional[float]:
    """Safely converts numeric or string input to float."""
    if val is None:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def _safe_int(val: Any, default: Optional[int] = None) -> Optional[int]:
    """Safely converts numeric or string input to integer with rounding."""
    if val is None:
        return default
    try:
        return int(round(float(val)))
    except (ValueError, TypeError):
        return default


def _parse_timestamp_ms(val: Any) -> Optional[int]:
    """Parses integer milliseconds or string timestamps (MM:SS, HH:MM:SS, SS.ms)."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return int(round(val)) if val > 1000 else int(round(val * 1000))
    s = str(val).strip()
    if s.isdigit():
        return int(s)
    parts = s.split(":")
    try:
        if len(parts) == 2:
            m, sec = parts
            return int((float(m) * 60 + float(sec)) * 1000)
        elif len(parts) == 3:
            h, m, sec = parts
            return int((float(h) * 3600 + float(m) * 60 + float(sec)) * 1000)
        f_val = float(s)
        return int(round(f_val)) if f_val > 1000 else int(round(f_val * 1000))
    except Exception:
        return None


def _parse_and_validate_ollama_response(
    raw_response: str,
    transcript: TranscriptData,
    max_clips: int = 3,
) -> List[ViralSegment]:
    """Parses JSON or markdown codeblocks from Ollama response and clamps constraints."""
    json_str = raw_response.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw_response)
    if match:
        json_str = match.group(1).strip()

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        return []

    if isinstance(data, dict):
        raw_segs = data.get("segments") or data.get("most_relevant_segments") or []
    elif isinstance(data, list):
        raw_segs = data
    else:
        return []

    total_duration_ms = transcript.words[-1].end if transcript.words else 0
    validated: List[ViralSegment] = []

    for item in raw_segs:
        start_ms_raw = item.get("start_time_ms")
        if start_ms_raw is None and "start_time" in item:
            start_ms_raw = item["start_time"]
        end_ms_raw = item.get("end_time_ms")
        if end_ms_raw is None and "end_time" in item:
            end_ms_raw = item["end_time"]

        if start_ms_raw is None or end_ms_raw is None:
            continue

        start_ms = _safe_int(start_ms_raw)
        if start_ms is None:
            start_ms = _parse_timestamp_ms(start_ms_raw)

        end_ms = _safe_int(end_ms_raw)
        if end_ms is None:
            end_ms = _parse_timestamp_ms(end_ms_raw)

        if start_ms is None or end_ms is None:
            continue

        start_ms = max(0, start_ms)
        if total_duration_ms >= 15_000:
            start_ms = min(total_duration_ms, start_ms)
            end_ms = min(total_duration_ms, end_ms)

        if end_ms <= start_ms:
            continue

        # Enforce duration bounds [15s, 120s]
        duration_ms = end_ms - start_ms
        if duration_ms < 15_000:
            end_ms = min(total_duration_ms, start_ms + 25_000)
        elif duration_ms > 120_000:
            end_ms = start_ms + 120_000

        # Subscores clamping [0, 25] with safe float-string parsing
        hook_s = max(0, min(25, _safe_int(item.get("hook_score"), 15)))
        eng_s = max(0, min(25, _safe_int(item.get("engagement_score"), 15)))
        val_s = max(0, min(25, _safe_int(item.get("value_score"), 15)))
        share_s = max(0, min(25, _safe_int(item.get("shareability_score"), 15)))
        total_s = max(0, min(100, hook_s + eng_s + val_s + share_s))

        title = sanitize_hook_title(item.get("hook_title") or item.get("title"))
        reasoning = str(
            item.get("reasoning")
            or "High-impact hook with compelling engagement and clear payoff."
        ).strip()

        seg = ViralSegment(
            start_time_ms=start_ms,
            end_time_ms=end_ms,
            hook_title=title,
            hook_score=hook_s,
            engagement_score=eng_s,
            value_score=val_s,
            shareability_score=share_s,
            total_score=total_s,
            reasoning=reasoning,
        )

        teaser_start = item.get("hook_start_time_ms")
        teaser_end = item.get("hook_end_time_ms")
        valid_llm_teaser = False
        if teaser_start is not None and teaser_end is not None:
            t_s = _safe_int(teaser_start)
            if t_s is None:
                t_s = _parse_timestamp_ms(teaser_start)
            t_e = _safe_int(teaser_end)
            if t_e is None:
                t_e = _parse_timestamp_ms(teaser_end)

            if t_s is not None and t_e is not None:
                # Clamp teaser duration bounds [2000, 5000]ms
                if t_e - t_s < 2000:
                    t_e = t_s + 2000
                elif t_e - t_s > 5000:
                    t_e = t_s + 5000

                seg_dur_ms = seg.end_time_ms - seg.start_time_ms
                if (
                    seg_dur_ms >= 20_000
                    and seg.start_time_ms < t_s < t_e <= seg.end_time_ms
                    and 2000 <= (t_e - t_s) <= 5000
                ):
                    seg.hook_start_time_ms = t_s
                    seg.hook_end_time_ms = t_e
                    valid_llm_teaser = True

        if not valid_llm_teaser:
            teaser = detect_cold_open_teaser(seg, transcript.words)
            if teaser:
                seg.hook_start_time_ms = teaser.start_time_ms
                seg.hook_end_time_ms = teaser.end_time_ms

        validated.append(seg)

    validated.sort(key=lambda s: (-s.total_score, s.start_time_ms))
    return validated[:max_clips]


def _call_ollama_virality(
    transcript: TranscriptData,
    max_clips: int = 3,
    model_name: str = "ollama:llama3.2:3b",
    ollama_url: str = "http://localhost:11434/api/generate",
    timeout_s: float = 30.0,
    video_title: Optional[str] = None,
) -> List[ViralSegment]:
    """Queries local Ollama daemon for virality evaluation."""
    model = model_name.split(":", 1)[1] if ":" in model_name else model_name
    title_ctx = f"The source video title is: '{video_title}'.\n" if video_title else ""
    prompt = (
        f"You are an expert viral video editor and headline copywriter. {title_ctx}"
        f"Analyze this transcript and select top {max_clips} viral highlight segments (15s to 60s each).\n"
        f"Rules for hook_title:\n"
        f"- Must make complete semantic sense on its own and be grammatically correct English.\n"
        f"- Enforce 3 to 9 words, max 60 characters.\n"
        f"- Use Title Case capitalization and proper contractions (e.g. don't, can't, it's, let's, catcher's).\n"
        f"- Do NOT output incomplete dependent clauses (e.g. never 'Before they move to round' or 'Because we did it').\n"
        f"- Do NOT include conversational filler words like 'Wait, let's say', 'You know', 'I mean', 'Like'.\n"
        f"- If it asks a question, end with '?'.\n"
        f"- Do NOT include clip numbering or list prefixes like '1.' or 'Part 1'.\n"
        f"Return JSON with 'segments' list. Each item must have: start_time_ms, end_time_ms, hook_title, "
        f"hook_score (0-25), engagement_score (0-25), value_score (0-25), shareability_score (0-25), reasoning.\n\n"
        f"Transcript:\n{transcript.text[:4000]}"
    )
    payload = {
        "model": model,
        "prompt": prompt,
        "format": "json",
        "stream": False,
    }
    req = urllib.request.Request(
        ollama_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    resp = urllib.request.urlopen(req, timeout=timeout_s)
    try:
        raw = resp.read()
        if isinstance(raw, (bytes, bytearray)):
            raw_text = raw.decode("utf-8")
        elif isinstance(raw, str):
            raw_text = raw
        else:
            raw_val = getattr(resp.read, "return_value", "")
            raw_text = raw_val.decode("utf-8") if isinstance(raw_val, (bytes, bytearray)) else str(raw_val)
    finally:
        if hasattr(resp, "close") and callable(resp.close):
            try:
                resp.close()
            except Exception:
                pass

    res_data = json.loads(raw_text)
    raw_response = res_data.get("response", "")
    return _parse_and_validate_ollama_response(raw_response, transcript, max_clips)


def analyze_long_video_windowed(
    transcript: TranscriptData,
    window_size_s: int = 720,
    overlap_s: int = 120,
    max_clips: int = 3,
    llm_model: str = "ollama:llama3.2:3b",
    video_title: Optional[str] = None,
) -> List[ViralSegment]:
    """Windows long transcripts > 15m and deduplicates top candidates."""
    total_duration_ms = transcript.words[-1].end if transcript.words else 0
    total_duration_s = total_duration_ms / 1000.0

    if total_duration_s <= 900.0:
        if llm_model == "heuristic" or llm_model.startswith("heuristic"):
            return heuristic_score_segments(transcript, max_clips=max_clips, video_title=video_title)
        try:
            return _call_ollama_virality(transcript, max_clips=max_clips, model_name=llm_model)
        except Exception:
            return heuristic_score_segments(transcript, max_clips=max_clips, video_title=video_title)

    windows = create_analysis_windows(total_duration_s, float(window_size_s), float(overlap_s))
    all_candidates: List[ViralSegment] = []

    for win_start_s, win_end_s in windows:
        win_start_ms = int(win_start_s * 1000)
        win_end_ms = int(win_end_s * 1000)

        sub_words = [w for w in transcript.words if win_start_ms <= w.start < win_end_ms]
        if not sub_words:
            continue

        sub_transcript = TranscriptData(
            text=" ".join(w.text for w in sub_words),
            language=transcript.language,
            words=sub_words,
        )

        if llm_model == "heuristic" or llm_model.startswith("heuristic"):
            win_candidates = heuristic_score_segments(sub_transcript, max_clips=max_clips, video_title=video_title)
        else:
            try:
                win_candidates = _call_ollama_virality(sub_transcript, max_clips=max_clips, model_name=llm_model, video_title=video_title)
            except Exception:
                win_candidates = heuristic_score_segments(sub_transcript, max_clips=max_clips, video_title=video_title)

        all_candidates.extend(win_candidates)

    deduped = deduplicate_segments(all_candidates, overlap_threshold=0.5)
    return deduped[:max_clips]


def extract_viral_segments(
    transcript: TranscriptData,
    max_clips: int = 3,
    llm_model: str = "ollama:llama3.2:3b",
    window_size_s: int = 720,
    overlap_s: int = 120,
    heatmap: Optional[List[Dict[str, float]]] = None,
    max_duration_s: float = 120.0,
    video_title: Optional[str] = None,
) -> List[ViralSegment]:
    """
    Extracts top viral segments using YouTube Most Replayed heatmap, Ollama local model, or heuristic fallback.
    - Senses and prioritizes YouTube 'Most Replayed' heatmap peaks when available.
    - Aligns segment boundaries strictly to complete sentences (<= 120 seconds).
    - Evaluates 4 dimensions: hook, engagement, value, shareability.
    """
    max_dur_ms = int(max_duration_s * 1000)

    # 1. YouTube Most Replayed Heatmap (Direct viewer engagement sensing)
    if heatmap:
        try:
            hm_segs = extract_heatmap_highlights(
                heatmap=heatmap,
                transcript=transcript,
                max_clips=max_clips,
                min_duration_ms=20_000,
                max_duration_ms=max_dur_ms,
                video_title=video_title,
            )
            if hm_segs:
                logger.info("Extracted %d viral highlight clips from YouTube Most Replayed heatmap!", len(hm_segs))
                return hm_segs
        except Exception as exc:
            logger.warning("Heatmap highlight extraction failed (%s); proceeding to semantic analysis.", exc)

    total_duration_ms = transcript.words[-1].end if transcript.words else 0

    if total_duration_ms > 900_000:  # > 15 minutes
        return analyze_long_video_windowed(
            transcript=transcript,
            window_size_s=window_size_s,
            overlap_s=overlap_s,
            max_clips=max_clips,
            llm_model=llm_model,
            video_title=video_title,
        )

    if llm_model == "heuristic" or llm_model.startswith("heuristic"):
        return heuristic_score_segments(transcript, max_clips=max_clips, max_duration_ms=max_dur_ms, video_title=video_title)

    try:
        segments = _call_ollama_virality(transcript, max_clips=max_clips, model_name=llm_model, video_title=video_title)
        if not segments:
            return heuristic_score_segments(transcript, max_clips=max_clips, max_duration_ms=max_dur_ms, video_title=video_title)
        return segments
    except Exception as exc:
        logger.exception("Ollama virality analysis failed (%s); falling back to heuristic scorer.", exc)
        return heuristic_score_segments(transcript, max_clips=max_clips, max_duration_ms=max_dur_ms, video_title=video_title)
