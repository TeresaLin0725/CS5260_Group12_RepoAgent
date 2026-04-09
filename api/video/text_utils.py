"""Pure text/keyword utility functions shared across video pipeline modules."""

import re
from typing import TYPE_CHECKING, Any, List, Optional

from api.video.constants import (
    MAX_BULLET_CHARS,
    MAX_BULLETS,
    MAX_KEYWORDS,
    MAX_NODE_DESC_CHARS,
    MAX_SUBTITLE_CHARS,
    NARRATION_MAX_CHARS,
)

if TYPE_CHECKING:
    from api.content_analyzer import AnalyzedContent


def _truncate_narration(text: str, max_chars: int = NARRATION_MAX_CHARS) -> str:
    """Truncate narration at a sentence boundary to fit TTS duration target."""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    last_period = truncated.rfind(". ")
    if last_period > max_chars // 3:
        return truncated[: last_period + 1]
    last_space = truncated.rfind(" ")
    if last_space > 0:
        return truncated[:last_space].rstrip(",.;:") + "."
    return truncated + "."


def _clean_keyword(text: str, max_chars: Optional[int] = MAX_BULLET_CHARS) -> str:
    text = re.sub(r"[`*_]", "", text)
    text = text.replace("_", " ").replace("/", " / ")
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = re.sub(r"\s+", " ", text).strip(" .,:;-\n\t")
    if max_chars is not None and len(text) > max_chars:
        text = text[: max_chars - 3].rstrip() + "..."
    return text


def _clean_entity_label(text: str) -> str:
    return _clean_keyword(text, max_chars=None)


def _keyword_phrases(text: str, limit: int = MAX_KEYWORDS) -> List[str]:
    if not text:
        return []
    parts = re.split(r"[.;:()\n]|\band\b|\bthat\b|\bwhich\b|\bwith\b", text, flags=re.IGNORECASE)
    keywords: list[str] = []
    seen: set[str] = set()
    for part in parts:
        cleaned = _clean_keyword(part)
        if not cleaned:
            continue
        words = cleaned.split()
        if len(words) > 6:
            cleaned = " ".join(words[:6])
        lowered = cleaned.lower()
        if lowered not in seen:
            keywords.append(cleaned)
            seen.add(lowered)
        if len(keywords) >= limit:
            break
    return keywords


def _short_desc(text: str, max_chars: int = MAX_NODE_DESC_CHARS) -> str:
    """Extract the shortest meaningful keyword phrase from text for on-screen display."""
    if not text:
        return ""
    phrases = _keyword_phrases(text, limit=1)
    if phrases:
        return _clean_keyword(phrases[0], max_chars)
    return _clean_keyword(text, max_chars)


def _bubble_caption(text: str, max_words: int = 3, max_chars: int = 22) -> str:
    """Extract a very short caption (2-3 key words) for speech bubbles."""
    if not text:
        return ""
    filler = {"is", "an", "a", "the", "it", "this", "that", "of", "for", "to",
              "and", "or", "in", "on", "by", "via", "with", "from", "who", "want",
              "repo", "helper", "repository", "so", "what", "how", "where"}
    words = _clean_keyword(text, None).split()
    key_words = [w for w in words if w.lower() not in filler and len(w) > 1]
    if not key_words:
        key_words = words[:max_words]
    caption = " ".join(key_words[:max_words])
    if len(caption) > max_chars:
        truncated = caption[:max_chars - 2].rsplit(" ", 1)[0]
        return truncated.rstrip(",.;:") if len(truncated) > 3 else caption[:max_chars]
    return caption


def _chunk_list(items: List[Any], chunk_size: int) -> List[List[Any]]:
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def _module_lookup(analyzed: "AnalyzedContent") -> dict[str, Any]:
    return {m.name: m for m in analyzed.module_progression}


def _segment_narration(narration: str, entities: List[dict]) -> List[dict]:
    """Split narration into 2-4 segments, each mapped to entities to highlight."""
    if not narration or not narration.strip():
        return [{"text": "", "highlight_labels": [], "duration_fraction": 1.0}]

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", narration.strip()) if s.strip()]
    if not sentences:
        return [{"text": "", "highlight_labels": [], "duration_fraction": 1.0}]

    n_segments = min(max(2, len(sentences)), 4)
    segments: List[List[str]] = [[] for _ in range(n_segments)]
    for i, sentence in enumerate(sentences):
        segments[i % n_segments].append(sentence)

    entity_labels = [ent.get("label", "").lower() for ent in entities if ent.get("label")]

    result = []
    total_chars = sum(len(" ".join(seg)) for seg in segments) or 1
    for seg_sentences in segments:
        if not seg_sentences:
            continue
        text = " ".join(seg_sentences)
        display_text = text[:MAX_SUBTITLE_CHARS].rstrip() + ("..." if len(text) > MAX_SUBTITLE_CHARS else "")
        text_lower = text.lower()
        highlight = [ent.get("label", "") for ent in entities
                     if ent.get("label") and ent["label"].lower() in text_lower]
        if not highlight and entities:
            seg_idx = len(result)
            if seg_idx < len(entities):
                highlight = [entities[seg_idx].get("label", "")]
        result.append({
            "text": display_text,
            "highlight_labels": highlight,
            "duration_fraction": len(text) / total_chars,
        })

    return result if result else [{"text": "", "highlight_labels": [], "duration_fraction": 1.0}]


def _segment_narration_sequential(narration: str, panel_labels: List[str]) -> List[dict]:
    """Split narration into N segments matching panel count, highlight panels sequentially."""
    if not narration or not narration.strip():
        return [{"text": "", "highlight_labels": [], "duration_fraction": 1.0}]

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", narration.strip()) if s.strip()]
    if not sentences:
        return [{"text": "", "highlight_labels": [], "duration_fraction": 1.0}]

    n_panels = len(panel_labels) or 3
    segments: List[List[str]] = [[] for _ in range(n_panels)]
    for i, sentence in enumerate(sentences):
        segments[i % n_panels].append(sentence)

    result = []
    total_chars = sum(len(" ".join(seg)) for seg in segments) or 1
    for i, seg_sentences in enumerate(segments):
        if not seg_sentences:
            continue
        text = " ".join(seg_sentences)
        display_text = text[:MAX_SUBTITLE_CHARS].rstrip() + ("..." if len(text) > MAX_SUBTITLE_CHARS else "")
        highlight = [panel_labels[i]] if i < len(panel_labels) else []
        result.append({
            "text": display_text,
            "highlight_labels": highlight,
            "duration_fraction": len(text) / total_chars,
        })

    return result if result else [{"text": "", "highlight_labels": [], "duration_fraction": 1.0}]


def _split_narration_to_bullets(narration: str) -> List[str]:
    """Convert narration into a short bullet list for the baseline card layout."""
    if not narration:
        return []

    sentences = [
        part.strip(" -\n\t")
        for part in re.split(r"(?<=[.!?])\s+", narration.strip())
        if part.strip(" -\n\t")
    ]

    bullets: list[str] = []
    for sentence in sentences[:MAX_BULLETS]:
        compact = re.sub(r"\s+", " ", sentence).strip()
        if len(compact) > MAX_BULLET_CHARS:
            compact = compact[: MAX_BULLET_CHARS - 3].rstrip() + "..."
        bullets.append(compact)

    if bullets:
        return bullets

    compact = re.sub(r"\s+", " ", narration).strip()
    if len(compact) > MAX_BULLET_CHARS:
        compact = compact[: MAX_BULLET_CHARS - 3].rstrip() + "..."
    return [compact]
