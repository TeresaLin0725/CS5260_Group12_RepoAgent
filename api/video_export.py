"""
Video Export Module - Video Renderer.

Converts structured AnalyzedContent into a baseline video presentation.

Phase 2b-video uses an LLM call to rewrite structured analysis into a
conversational narration script with scene titles and duration hints.
Phase 3 renders the narration script into a slide-style MP4 walkthrough.
"""

import json
import logging
import os
import re
import tempfile
import time
import textwrap
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional

if TYPE_CHECKING:
    from api.content_analyzer import AnalyzedContent

logger = logging.getLogger(__name__)

VIDEO_SIZE = (1280, 720)
SCENE_DURATION_DEFAULT = 7
SCENE_DURATION_MIN = 4
SCENE_DURATION_MAX = 30
AUDIO_PADDING_SECONDS = 1.0
MAX_SCENES = 6
MAX_BULLETS = 3
MAX_BULLET_CHARS = 48
MAX_EXPANSION_SCENES = max(2, MAX_SCENES - 3)
MAX_KEYWORDS = 4
TRANSITION_SECONDS = 0.35
VIDEO_FPS = 24
NARRATION_MAX_CHARS = 280
MAX_NODE_DESC_CHARS = 30


# ---------------------------------------------------------------------------
# Phase 2b: structured analysis -> narration script
# ---------------------------------------------------------------------------

def _analysis_to_prompt_json(analyzed: "AnalyzedContent") -> str:
    """Serialize structured analysis into compact JSON for the prompt."""
    payload = {
        "repo_name": analyzed.repo_name,
        "repo_type_hint": analyzed.repo_type_hint,
        "project_overview": analyzed.project_overview,
        "architecture": analyzed.architecture,
        "tech_stack": {
            "languages": analyzed.tech_stack.languages,
            "frameworks": analyzed.tech_stack.frameworks,
            "key_libraries": analyzed.tech_stack.key_libraries,
            "infrastructure": analyzed.tech_stack.infrastructure,
        },
        "key_modules": [{"name": m.name, "responsibility": m.responsibility} for m in analyzed.key_modules],
        "data_flow": analyzed.data_flow,
        "api_points": analyzed.api_points,
        "target_users": analyzed.target_users,
        "module_progression": [
            {
                "name": m.name,
                "stage": m.stage,
                "role": m.role,
                "solves": m.solves,
                "position": m.position,
            }
            for m in analyzed.module_progression
        ],
    }
    if analyzed.deployment_info:
        payload["deployment_info"] = analyzed.deployment_info
    if analyzed.component_hierarchy:
        payload["component_hierarchy"] = analyzed.component_hierarchy
    if analyzed.data_schemas:
        payload["data_schemas"] = analyzed.data_schemas

    return json.dumps(payload, ensure_ascii=False, indent=2)


async def generate_narration_script(
    analyzed: "AnalyzedContent",
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> List[dict]:
    """
    Call the LLM with VIDEO_NARRATION_PROMPT to produce a narration script.

    Returns a list of scene dicts:
        [{"title": str, "narration": str, "duration_seconds": int}, ...]
    """
    from api.content_analyzer import _get_language_name
    from api.prompts import VIDEO_NARRATION_PROMPT

    start = time.perf_counter()
    language_name = _get_language_name(analyzed.language)
    analysis_json = _analysis_to_prompt_json(analyzed)
    prompt = VIDEO_NARRATION_PROMPT.format(
        language_name=language_name,
        analysis_json=analysis_json,
    )
    logger.info("Timing - video prompt assembly completed in %.2fs", time.perf_counter() - start)

    llm_start = time.perf_counter()
    raw_text = await _call_llm_raw(
        prompt=prompt,
        provider=provider or "openai",
        model=model,
    )
    logger.info("Timing - narration LLM call completed in %.2fs", time.perf_counter() - llm_start)

    parse_start = time.perf_counter()
    scenes = _parse_scene_array(raw_text)
    logger.info("Timing - narration parsing completed in %.2fs", time.perf_counter() - parse_start)
    if not scenes:
        fallback_start = time.perf_counter()
        scenes = _fallback_narration_script(analyzed)
        logger.info("Timing - narration fallback script built in %.2fs", time.perf_counter() - fallback_start)

    logger.info("Narration script generated - %d scenes (total %.2fs)", len(scenes), time.perf_counter() - start)
    return scenes


async def _call_llm_raw(prompt: str, provider: str, model: Optional[str]) -> str:
    """Dispatch a non-streaming LLM call and return the raw text response."""
    from adalflow.core.types import ModelType
    from api.config import get_model_config
    from api.content_analyzer import _extract_text_from_response

    config = get_model_config(provider, model)
    model_kwargs = config["model_kwargs"]

    if provider == "ollama":
        from adalflow.components.model_client.ollama_client import OllamaClient

        client = OllamaClient()
        kwargs = {
            "model": model_kwargs["model"],
            "stream": False,
            "options": {k: v for k, v in model_kwargs.items() if k != "model"},
        }
        api_kwargs = client.convert_inputs_to_api_kwargs(
            input=prompt,
            model_kwargs=kwargs,
            model_type=ModelType.LLM,
        )
        response = await client.acall(api_kwargs=api_kwargs, model_type=ModelType.LLM)
    elif provider == "openrouter":
        from api.openrouter_client import OpenRouterClient

        client = OpenRouterClient()
        kwargs = {
            "model": model_kwargs["model"],
            "stream": False,
            "temperature": model_kwargs.get("temperature", 0.5),
        }
        api_kwargs = client.convert_inputs_to_api_kwargs(
            input=prompt,
            model_kwargs=kwargs,
            model_type=ModelType.LLM,
        )
        response = await client.acall(api_kwargs=api_kwargs, model_type=ModelType.LLM)
    elif provider == "openai":
        from api.openai_client import OpenAIClient

        client = OpenAIClient()
        kwargs = {
            "model": model_kwargs["model"],
            "stream": False,
            "temperature": model_kwargs.get("temperature", 0.5),
        }
        api_kwargs = client.convert_inputs_to_api_kwargs(
            input=prompt,
            model_kwargs=kwargs,
            model_type=ModelType.LLM,
        )
        response = await client.acall(api_kwargs=api_kwargs, model_type=ModelType.LLM)
    elif provider == "bedrock":
        from api.bedrock_client import BedrockClient

        client = BedrockClient()
        kwargs = {
            "model": model_kwargs["model"],
            "temperature": model_kwargs.get("temperature", 0.5),
            "top_p": model_kwargs.get("top_p", 0.9),
        }
        api_kwargs = client.convert_inputs_to_api_kwargs(
            input=prompt,
            model_kwargs=kwargs,
            model_type=ModelType.LLM,
        )
        response = await client.acall(api_kwargs=api_kwargs, model_type=ModelType.LLM)
    elif provider == "azure":
        from api.azureai_client import AzureAIClient

        client = AzureAIClient()
        kwargs = {
            "model": model_kwargs["model"],
            "stream": False,
            "temperature": model_kwargs.get("temperature", 0.5),
            "top_p": model_kwargs.get("top_p", 0.9),
        }
        api_kwargs = client.convert_inputs_to_api_kwargs(
            input=prompt,
            model_kwargs=kwargs,
            model_type=ModelType.LLM,
        )
        response = await client.acall(api_kwargs=api_kwargs, model_type=ModelType.LLM)
    elif provider == "dashscope":
        from api.dashscope_client import DashscopeClient

        client = DashscopeClient()
        kwargs = {
            "model": model_kwargs["model"],
            "stream": False,
            "temperature": model_kwargs.get("temperature", 0.5),
            "top_p": model_kwargs.get("top_p", 0.9),
        }
        api_kwargs = client.convert_inputs_to_api_kwargs(
            input=prompt,
            model_kwargs=kwargs,
            model_type=ModelType.LLM,
        )
        response = await client.acall(api_kwargs=api_kwargs, model_type=ModelType.LLM)
    else:
        import google.generativeai as genai

        gen_model = genai.GenerativeModel(
            model_name=model_kwargs.get("model", "gemini-2.0-flash"),
            generation_config={
                "temperature": model_kwargs.get("temperature", 0.5),
                "top_p": model_kwargs.get("top_p", 0.9),
                "top_k": model_kwargs.get("top_k", 40),
            },
        )
        response = await gen_model.generate_content_async(prompt)
        return response.text if hasattr(response, "text") else str(response)

    return _extract_text_from_response(response, provider)


# ---------------------------------------------------------------------------
# Phase 2b helpers: narration-script parsing and fallback construction
# ---------------------------------------------------------------------------

def _parse_scene_array(raw_text: str) -> List[dict]:
    """Parse a JSON array of scene objects from raw LLM text."""
    cleaned = re.sub(r"```(?:json)?\s*", "", raw_text)
    cleaned = re.sub(r"```\s*$", "", cleaned)

    try:
        data = json.loads(cleaned)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass

    match = re.search(r"\[", cleaned)
    if match:
        depth = 0
        start = match.start()
        for i in range(start, len(cleaned)):
            if cleaned[i] == "[":
                depth += 1
            elif cleaned[i] == "]":
                depth -= 1
                if depth == 0:
                    try:
                        data = json.loads(cleaned[start : i + 1])
                        if isinstance(data, list):
                            return data
                    except json.JSONDecodeError:
                        break

    return []


def _truncate_narration(text: str, max_chars: int = NARRATION_MAX_CHARS) -> str:
    """Truncate narration at a sentence boundary to fit TTS duration target."""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    # Try to cut at a sentence boundary
    truncated = text[:max_chars]
    last_period = truncated.rfind(". ")
    if last_period > max_chars // 3:
        return truncated[: last_period + 1]
    # Fall back to word boundary
    last_space = truncated.rfind(" ")
    if last_space > 0:
        return truncated[:last_space].rstrip(",.;:") + "."
    return truncated + "."


MAX_SUBTITLE_CHARS = 160


def _segment_narration(narration: str, entities: List[dict]) -> List[dict]:
    """Split narration into 2-4 segments, each mapped to entities to highlight.

    Returns list of dicts with keys: text, highlight_labels, duration_fraction.
    """
    if not narration or not narration.strip():
        return [{"text": "", "highlight_labels": [], "duration_fraction": 1.0}]

    # Split into sentences
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", narration.strip()) if s.strip()]
    if not sentences:
        return [{"text": "", "highlight_labels": [], "duration_fraction": 1.0}]

    # Group sentences into 2-4 segments
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
        # Truncate subtitle display text
        display_text = text[:MAX_SUBTITLE_CHARS].rstrip() + ("..." if len(text) > MAX_SUBTITLE_CHARS else "")

        # Match entity labels mentioned in this segment
        text_lower = text.lower()
        highlight = [ent.get("label", "") for ent in entities
                     if ent.get("label") and ent["label"].lower() in text_lower]

        # If no match found, highlight by index position
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
    """Split narration into N segments matching panel count, highlight panels sequentially.

    Used for comic-style scenes where each narration segment maps 1:1 to a panel.
    """
    if not narration or not narration.strip():
        return [{"text": "", "highlight_labels": [], "duration_fraction": 1.0}]

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", narration.strip()) if s.strip()]
    if not sentences:
        return [{"text": "", "highlight_labels": [], "duration_fraction": 1.0}]

    n_panels = len(panel_labels) or 3
    # Distribute sentences evenly across panels
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

        # Sequential: highlight the panel at this index
        highlight = [panel_labels[i]] if i < len(panel_labels) else []

        result.append({
            "text": display_text,
            "highlight_labels": highlight,
            "duration_fraction": len(text) / total_chars,
        })

    return result if result else [{"text": "", "highlight_labels": [], "duration_fraction": 1.0}]


def _chunk_list(items: List[Any], chunk_size: int) -> List[List[Any]]:
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


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
    """Extract a very short caption (2-3 key words) for speech bubbles.

    More aggressive than _short_desc — strips filler words and takes only
    the most meaningful nouns/adjectives.
    """
    if not text:
        return ""
    # Remove common filler words
    filler = {"is", "an", "a", "the", "it", "this", "that", "of", "for", "to",
              "and", "or", "in", "on", "by", "via", "with", "from", "who", "want",
              "repo", "helper", "repository", "so", "what", "how", "where"}
    words = _clean_keyword(text, None).split()
    key_words = [w for w in words if w.lower() not in filler and len(w) > 1]
    if not key_words:
        key_words = words[:max_words]
    caption = " ".join(key_words[:max_words])
    if len(caption) > max_chars:
        # Truncate at word boundary
        truncated = caption[:max_chars - 2].rsplit(" ", 1)[0]
        return truncated.rstrip(",.;:") if len(truncated) > 3 else caption[:max_chars]
    return caption


def _module_lookup(analyzed: "AnalyzedContent") -> dict[str, Any]:
    return {m.name: m for m in analyzed.module_progression}


def _measure_lines(draw, text: str, font, max_width: int, max_lines: Optional[int] = None) -> List[str]:
    words = _clean_keyword(text, max_chars=None).split()
    if not words:
        return []
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        trial = f"{current} {word}"
        if draw.textbbox((0, 0), trial, font=font)[2] <= max_width:
            current = trial
        else:
            lines.append(current)
            current = word
            if max_lines and len(lines) >= max_lines:
                return lines
    lines.append(current)
    if max_lines:
        return lines[:max_lines]
    return lines


def _font_height(draw, font) -> int:
    bbox = draw.textbbox((0, 0), "Ag", font=font)
    return bbox[3] - bbox[1]


def _fit_text_block(draw, text: str, max_width: int, max_height: int, preferred_font, min_size: int = 14, max_lines: int = 3):
    from PIL import ImageFont

    if not text:
        return preferred_font, [], 0

    font = preferred_font
    font_path = getattr(preferred_font, 'path', None)
    current_size = getattr(preferred_font, 'size', min_size)
    while current_size >= min_size:
        lines = _measure_lines(draw, text, font, max_width)
        if lines and len(lines) <= max_lines:
            line_height = _font_height(draw, font)
            total_height = len(lines) * line_height + max(0, len(lines) - 1) * 8
            widest = max(draw.textbbox((0, 0), line, font=font)[2] for line in lines)
            if widest <= max_width and total_height <= max_height:
                return font, lines, line_height
        current_size -= 2
        if font_path:
            font = ImageFont.truetype(font_path, current_size)
        else:
            break

    lines = _measure_lines(draw, text, font, max_width, max_lines=max_lines) or [text]
    line_height = _font_height(draw, font)
    return font, lines[:max_lines], line_height


def _fit_text_lines(draw, text: str, font, max_width: int, max_lines: int) -> List[str]:
    _, lines, _ = _fit_text_block(draw, text, max_width, 10_000, font, min_size=max(12, getattr(font, 'size', 18) - 14), max_lines=max_lines)
    return lines


def _rect_with_padding(rect: tuple[int, int, int, int], padding: int = 8) -> tuple[int, int, int, int]:
    return (rect[0] - padding, rect[1] - padding, rect[2] + padding, rect[3] + padding)


def _rects_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] <= b[0] or a[0] >= b[2] or a[3] <= b[1] or a[1] >= b[3])


def _find_clear_label_rect(label_size: tuple[int, int], anchor: tuple[int, int], blocked: List[tuple[int, int, int, int]], canvas: tuple[int, int], padding: int = 8) -> tuple[int, int, int, int]:
    width, height = label_size
    ax, ay = anchor
    candidates = [
        (ax - width // 2, ay - height - 18),
        (ax - width // 2, ay + 18),
        (ax + 18, ay - height // 2),
        (ax - width - 18, ay - height // 2),
        (ax - width // 2, ay - height // 2),
    ]
    canvas_w, canvas_h = canvas
    for x, y in candidates:
        x = max(24, min(x, canvas_w - width - 24))
        y = max(110, min(y, canvas_h - height - 24))
        rect = (x, y, x + width, y + height)
        padded = _rect_with_padding(rect, padding)
        if not any(_rects_overlap(padded, _rect_with_padding(other, padding)) for other in blocked):
            return rect
    x = max(24, min(ax - width // 2, canvas_w - width - 24))
    y = max(110, min(ay + 18, canvas_h - height - 24))
    return (x, y, x + width, y + height)


def _draw_fitted_text(draw, text: str, rect: tuple[int, int, int, int], font, fill, align: str = 'left', valign: str = 'middle', max_lines: int = 3) -> tuple[int, int, int, int]:
    max_width = max(20, rect[2] - rect[0])
    max_height = max(20, rect[3] - rect[1])
    fitted_font, lines, line_height = _fit_text_block(draw, text, max_width, max_height, font, max_lines=max_lines)
    total_height = len(lines) * line_height + max(0, len(lines) - 1) * 8
    if valign == 'top':
        y = rect[1]
    elif valign == 'bottom':
        y = rect[3] - total_height
    else:
        y = rect[1] + max(0, (max_height - total_height) // 2)
    used_left = rect[2]
    used_right = rect[0]
    for line in lines:
        line_width = draw.textbbox((0, 0), line, font=fitted_font)[2]
        if align == 'center':
            x = rect[0] + max(0, (max_width - line_width) // 2)
        elif align == 'right':
            x = rect[2] - line_width
        else:
            x = rect[0]
        draw.text((x, y), line, font=fitted_font, fill=fill)
        used_left = min(used_left, x)
        used_right = max(used_right, x + line_width)
        y += line_height + 8
    return (used_left, rect[1], used_right, rect[1] + total_height)


def _draw_boxed_keywords(draw, title: str, items: List[str], rect: tuple[int, int, int, int], title_font, item_font, colors: tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]) -> None:
    fill, outline, text_fill = colors
    draw.rounded_rectangle(rect, radius=18, fill=fill, outline=outline, width=2)
    draw.text((rect[0] + 18, rect[1] + 14), title, font=title_font, fill=text_fill)
    y = rect[1] + 50
    max_width = rect[2] - rect[0] - 58
    for item in items[:3]:
        lines = _fit_text_lines(draw, item, item_font, max_width, 1)
        if not lines:
            continue
        draw.ellipse((rect[0] + 22, y + 12, rect[0] + 30, y + 20), fill=text_fill)
        draw.text((rect[0] + 42, y + 7), lines[0], font=item_font, fill=(235, 241, 248))
        y += 46
        if y > rect[3] - 44:
            break


def _build_storyline_scenes(analyzed: "AnalyzedContent", raw_scenes: List[dict]) -> List[dict]:
    """Force a stable walkthrough structure: overview -> core -> expansion* -> summary."""

    def _pick_scene(section: str) -> Optional[dict]:
        for scene in raw_scenes:
            if str(scene.get("section") or "").strip().lower() == section:
                return scene
        return None

    overview_scene = _pick_scene("overview")
    core_scene = _pick_scene("core")
    summary_scene = _pick_scene("summary")
    expansion_seed_scenes = [
        scene for scene in raw_scenes if str(scene.get("section") or "").strip().lower() == "expansion"
    ]

    core_modules = [m for m in analyzed.module_progression if getattr(m, "stage", "") == "core"]
    expansion_modules = [m for m in analyzed.module_progression if getattr(m, "stage", "") == "expansion"]

    scenes: list[dict] = []

    tech_anchor: list[str] = []
    tech_anchor.extend(analyzed.tech_stack.languages[:2])
    tech_anchor.extend(analyzed.tech_stack.frameworks[:2])
    overview_fallback = (
        f"So what is {analyzed.repo_name or 'this project'}? "
        f"{analyzed.project_overview.strip()} "
        f"{'It is built with ' + ', '.join(tech_anchor[:4]) + '.' if tech_anchor else ''}"
    ).strip()
    overview_narration = (
        str((overview_scene or {}).get("narration") or "").strip()
        or overview_fallback
    )
    scenes.append(
        {
            "title": str((overview_scene or {}).get("title") or (analyzed.repo_name or "Repository Overview")).strip(),
            "section": "overview",
            "visual_type": "overview_map",
            "visual_motif": str((overview_scene or {}).get("visual_motif") or "diagram").strip().lower(),
            "entities": (overview_scene or {}).get("entities") or [
                {"label": _clean_entity_label(analyzed.repo_name or "Repository"), "kind": "file"},
                {"label": "Core path", "kind": "concept"},
                {"label": "Users", "kind": "user"},
                {"label": "Outputs", "kind": "concept"},
            ],
            "relations": (overview_scene or {}).get("relations") or [
                {"from": _clean_entity_label(analyzed.repo_name or "Repository"), "to": "Core path", "type": "feeds"},
                {"from": "Core path", "to": "Outputs", "type": "extends"},
                {"from": "Outputs", "to": "Users", "type": "helps"},
            ],
            "narration": _truncate_narration(overview_narration),
            "duration_seconds": (overview_scene or {}).get("duration_seconds", 6),
            "focus_modules": [],
        }
    )

    core_focus = core_modules[:3]
    if core_focus:
        core_names = " and ".join(m.name for m in core_focus)
        core_roles = " ".join(
            f"{m.name} {m.role.rstrip('.')}." for m in core_focus[:2]
        )
        core_default = (
            f"The minimum viable system is built on {core_names}. "
            f"{core_roles} "
            f"With just these pieces, users can already get the core value out of {analyzed.repo_name or 'the project'}."
        )
    else:
        core_default = (
            f"Let's look at the foundation. The smallest useful version of {analyzed.repo_name or 'this project'} "
            f"needs just a few key modules working together to deliver its core value."
        )
    scenes.append(
        {
            "title": str((core_scene or {}).get("title") or "The Core Backbone").strip(),
            "section": "core",
            "visual_type": "core_diagram",
            "visual_motif": str((core_scene or {}).get("visual_motif") or "relay").strip().lower(),
            "entities": (core_scene or {}).get("entities") or [
                {"label": _clean_entity_label(m.name), "kind": "file"} for m in core_focus[:3]
            ],
            "relations": (core_scene or {}).get("relations") or [
                {"from": _clean_entity_label(core_focus[i].name), "to": _clean_entity_label(core_focus[i + 1].name), "type": "calls"}
                for i in range(max(0, len(core_focus) - 1))
            ],
            "narration": _truncate_narration(str((core_scene or {}).get("narration") or "").strip() or core_default),
            "duration_seconds": (core_scene or {}).get("duration_seconds", 7),
            "focus_modules": [m.name for m in core_focus],
        }
    )

    expansion_groups = _chunk_list(expansion_modules[:MAX_EXPANSION_SCENES], 1)
    for index, group in enumerate(expansion_groups, start=1):
        seed = expansion_seed_scenes[index - 1] if index - 1 < len(expansion_seed_scenes) else {}
        module = group[0]
        solves_text = module.solves.rstrip('.') if module.solves else "a gap in the system"
        role_text = module.role.rstrip('.') if module.role else "extends the core"
        default_narration = (
            f"At this point the core works, but there is a problem: {solves_text}. "
            f"{module.name} addresses this. It {role_text}. "
            f"With this in place, the system becomes more capable."
        )
        scenes.append(
            {
                "title": str(seed.get("title") or f"Expansion Layer {index}").strip(),
                "section": "expansion",
                "visual_type": "expansion_ladder",
                "visual_motif": str(seed.get("visual_motif") or "dialogue").strip().lower(),
                "entities": seed.get("entities") or [
                    {"label": _clean_entity_label(group[0].name), "kind": "file"},
                    {"label": "Core path", "kind": "concept"},
                    {"label": _clean_entity_label(_keyword_phrases(group[0].solves, 1)[0] if _keyword_phrases(group[0].solves, 1) else 'Capability'), "kind": "concept"},
                ],
                "relations": seed.get("relations") or [
                    {"from": "Core path", "to": _clean_entity_label(group[0].name), "type": "extends"},
                    {"from": _clean_entity_label(group[0].name), "to": _clean_entity_label(_keyword_phrases(group[0].solves, 1)[0] if _keyword_phrases(group[0].solves, 1) else 'Capability'), "type": "helps"},
                ],
                "narration": _truncate_narration(str(seed.get("narration") or "").strip() or default_narration),
                "duration_seconds": seed.get("duration_seconds", 6),
                "focus_modules": [m.name for m in group],
            }
        )

    user_story = analyzed.target_users[:320] if analyzed.target_users else ""
    if user_story:
        summary_default = (
            f"Now let's put it all together. {user_story} "
            f"That is what {analyzed.repo_name or 'this project'} enables end to end."
        )
    else:
        summary_default = (
            f"With all these pieces in place, {analyzed.repo_name or 'this project'} "
            f"goes from a collection of files to a working system that users can rely on for their day to day workflow."
        )
    scenes.append(
        {
            "title": str((summary_scene or {}).get("title") or "Complete System and Use Cases").strip(),
            "section": "summary",
            "visual_type": "summary_usecases",
            "visual_motif": str((summary_scene or {}).get("visual_motif") or "usecases").strip().lower(),
            "entities": (summary_scene or {}).get("entities") or [
                {"label": "Users", "kind": "user"},
                {"label": "Workflow", "kind": "concept"},
                {"label": "Outcome", "kind": "concept"},
            ],
            "relations": (summary_scene or {}).get("relations") or [
                {"from": "Users", "to": "Workflow", "type": "calls"},
                {"from": "Workflow", "to": "Outcome", "type": "helps"},
            ],
            "narration": _truncate_narration(str((summary_scene or {}).get("narration") or "").strip() or summary_default),
            "duration_seconds": (summary_scene or {}).get("duration_seconds", 6),
            "focus_modules": [],
        }
    )
    return scenes[:MAX_SCENES]


def _fallback_narration_script(analyzed: "AnalyzedContent") -> List[dict]:
    """Build a deterministic storyline-first script directly from structured data."""
    return _build_storyline_scenes(analyzed, raw_scenes=[])


# ---------------------------------------------------------------------------
# Phase 3 helpers: normalize narration scenes for rendering
# ---------------------------------------------------------------------------

def _normalize_scenes(raw_scenes: List[dict], repo_name: str) -> List[dict]:
    """Normalize raw scene data into a renderer-friendly structure."""
    normalized: list[dict] = []

    for index, raw_scene in enumerate(raw_scenes[:MAX_SCENES], start=1):
        if not isinstance(raw_scene, dict):
            continue

        title = str(raw_scene.get("title") or f"Scene {index}").strip() or f"Scene {index}"
        narration = str(raw_scene.get("narration") or "").strip()
        if not narration:
            continue

        duration = raw_scene.get("duration_seconds", SCENE_DURATION_DEFAULT)
        try:
            duration = int(duration)
        except (TypeError, ValueError):
            duration = SCENE_DURATION_DEFAULT
        duration = max(SCENE_DURATION_MIN, min(duration, SCENE_DURATION_MAX))

        normalized.append(
            {
                "title": title[:80],
                "section": str(raw_scene.get("section") or "").strip().lower(),
                "visual_type": str(raw_scene.get("visual_type") or "").strip().lower(),
                "visual_motif": str(raw_scene.get("visual_motif") or "").strip().lower(),
                "focus_modules": [str(item).strip() for item in raw_scene.get("focus_modules", []) if str(item).strip()],
                "entities": [item for item in raw_scene.get("entities", []) if isinstance(item, dict)],
                "relations": [item for item in raw_scene.get("relations", []) if isinstance(item, dict)],
                "narration": _truncate_narration(narration),
                "duration_seconds": duration,
            }
        )

    if normalized:
        return normalized

    fallback_title = repo_name or "Repository Overview"
    return [
        {
            "title": fallback_title,
            "section": "overview",
            "visual_type": "overview_map",
            "visual_motif": "diagram",
            "focus_modules": [],
            "entities": [],
            "relations": [],
            "narration": f"This video provides a quick overview of {fallback_title}.",
            "duration_seconds": SCENE_DURATION_DEFAULT,
        }
    ]


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


def _scene_to_card_content(scene: dict, analyzed: "AnalyzedContent", index: int, total: int) -> dict:
    """Create a visual scene plan with keyword-first metadata for the renderer."""
    section = scene.get("section") or "overview"
    visual_type = scene.get("visual_type") or {
        "overview": "overview_map",
        "core": "core_diagram",
        "expansion": "expansion_ladder",
        "summary": "summary_usecases",
    }.get(section, "overview_map")

    focus_modules = scene.get("focus_modules") or []
    modules = _module_lookup(analyzed)
    scene_entities = [item for item in scene.get("entities", []) if isinstance(item, dict)]
    scene_relations = [item for item in scene.get("relations", []) if isinstance(item, dict)]
    tech_chips = [_clean_keyword(item, 24) for item in (analyzed.tech_stack.languages + analyzed.tech_stack.frameworks)[:4]]

    screen_keywords: list[str] = []
    for module_name in focus_modules:
        module = modules.get(module_name)
        if module:
            screen_keywords.append(_clean_keyword(module.name, 26))
            screen_keywords.extend(_keyword_phrases(module.solves, limit=2))
            screen_keywords.extend(_keyword_phrases(module.role, limit=1))
        else:
            screen_keywords.append(_clean_keyword(module_name, 26))

    if scene_entities:
        screen_keywords = [_clean_keyword(item.get("label", ""), 28) for item in scene_entities[:MAX_KEYWORDS]]
    elif section == "overview":
        screen_keywords = _keyword_phrases(analyzed.project_overview, limit=3) + tech_chips[:2]
    elif section == "summary":
        screen_keywords = _keyword_phrases(analyzed.target_users or scene["narration"], limit=4)

    deduped_keywords: list[str] = []
    for item in screen_keywords:
        cleaned = _clean_keyword(item, 32)
        if cleaned and cleaned.lower() not in {x.lower() for x in deduped_keywords}:
            deduped_keywords.append(cleaned)

    use_cases = [_clean_keyword(item, 34) for item in _keyword_phrases(analyzed.target_users or scene["narration"], limit=3)]
    microcopy = [_clean_keyword(item, 42) for item in _keyword_phrases(scene["narration"], limit=3)]

    # Build rich module details for expansion scenes
    module_details: list[dict] = []
    for module_name in focus_modules:
        module = modules.get(module_name)
        if module:
            module_details.append({
                "name": module.name,
                "role": _clean_keyword(module.role or "", 40),
                "solves": _clean_keyword(module.solves or "", 40),
                "stage": getattr(module, "stage", ""),
                "position": _clean_keyword(getattr(module, "position", ""), 40),
            })

    # For overview, build keyword-only node descriptions (full text goes to audio narration)
    overview_descriptions: list[str] = []
    if section == "overview":
        overview_descriptions = [
            _short_desc(analyzed.project_overview) if analyzed.project_overview else "",
            ", ".join(analyzed.tech_stack.frameworks[:2] + analyzed.tech_stack.languages[:1]),
            _short_desc(analyzed.target_users) if analyzed.target_users else "",
            ", ".join(_short_desc(f.responsibility) for f in analyzed.key_modules[:2]) if analyzed.key_modules else "",
        ]

    # For core, build keyword descriptions from module roles (full text goes to audio)
    core_descriptions: list[str] = []
    if section == "core":
        for m_name in focus_modules:
            m = modules.get(m_name)
            if m:
                core_descriptions.append(_short_desc(m.role, 40))

    # Build personas for comic-style overview and summary scenes
    # Labels: single word. Captions: one short phrase for the speech bubble.
    personas: list[dict] = []
    if section == "overview":
        # Extract a single-word user role from target_users
        user_role = "Developer"
        if analyzed.target_users:
            first_word = analyzed.target_users.split()[0:2]
            user_role = " ".join(first_word).rstrip("s,.")  # e.g. "Developer"
            if len(user_role) > 12:
                user_role = "Developer"
        # Compose short purposeful captions from structured data
        tech_names = analyzed.tech_stack.frameworks[:2]
        tech_caption = " + ".join(t.split()[0] for t in tech_names) if tech_names else "Code"
        # What this repo produces (from key_libraries or overview keywords)
        output_types = []
        for kw in ["doc", "diagram", "video", "chat", "pdf", "ppt", "export"]:
            if kw in (analyzed.project_overview or "").lower():
                output_types.append(kw.capitalize())
        output_caption = " & ".join(output_types[:2]) if output_types else "Documentation"
        personas = [
            {"svg": "person_thinking", "label": user_role, "caption": "What is this repo?"},
            {"svg": "person_at_desk", "label": "Analyze", "caption": tech_caption},
            {"svg": "person_happy", "label": "Understand", "caption": output_caption},
        ]
    elif section == "summary":
        # User journey: fixed purposeful captions (who → action → result)
        personas = [
            {"svg": "person_at_desk", "label": "User", "caption": "Submit repo URL"},
            {"svg": "process_gear", "label": "Process", "caption": "AI analysis"},
            {"svg": "person_happy", "label": "Result", "caption": "Get walkthrough"},
        ]

    built_entities = [{"label": _clean_entity_label(item.get("label", "")), "kind": item.get("kind", "concept")} for item in scene_entities[:4]]
    narration_text = scene.get("narration", "")

    # For comic scenes, use sequential highlighting (panel 1→2→3) instead of text matching
    if section in ("overview", "summary") and personas:
        narration_segments = _segment_narration_sequential(
            narration_text, [p["label"] for p in personas]
        )
    else:
        narration_segments = _segment_narration(narration_text, built_entities)

    return {
        "title": _clean_keyword(scene["title"], 50),
        "subtitle": _clean_keyword(analyzed.repo_name or "Repository Walkthrough", 36),
        "section": section,
        "visual_type": visual_type,
        "visual_motif": str(scene.get("visual_motif") or "").strip().lower(),
        "focus_modules": [_clean_keyword(item, 26) for item in focus_modules[:3]],
        "entities": built_entities,
        "relations": [
            {
                "from": _clean_entity_label(item.get("from", "")),
                "to": _clean_entity_label(item.get("to", "")),
                "type": _clean_keyword(item.get("type", ""), 18),
            }
            for item in scene_relations[:4]
        ],
        "tech_chips": tech_chips,
        "keywords": deduped_keywords[:MAX_KEYWORDS],
        "microcopy": microcopy[:3],
        "use_cases": use_cases[:3],
        "module_details": module_details,
        "overview_descriptions": overview_descriptions,
        "core_descriptions": core_descriptions,
        "personas": personas,
        "narration_segments": narration_segments,
        "footer": f"{analyzed.repo_name or 'Repo'} | {index}/{total}",
    }


# ---------------------------------------------------------------------------
# Phase 3 helpers: visual asset rendering
# ---------------------------------------------------------------------------

def _load_fonts():
    """Load available fonts for the card renderer, falling back safely."""
    from PIL import ImageFont

    candidate_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]

    for font_path in candidate_paths:
        if os.path.exists(font_path):
            try:
                return (
                    ImageFont.truetype(font_path, 52),
                    ImageFont.truetype(font_path, 30),
                    ImageFont.truetype(font_path, 22),
                )
            except OSError:
                continue

    fallback = ImageFont.load_default()
    return fallback, fallback, fallback


def _draw_wrapped_text(draw, text: str, font, x: int, y: int, fill, line_spacing: int = 10) -> int:
    """Draw wrapped text and return the next y position."""
    lines = textwrap.wrap(text, width=46) or [text]
    for line in lines:
        draw.text((x, y), line, font=font, fill=fill)
        bbox = draw.textbbox((x, y), line, font=font)
        y += (bbox[3] - bbox[1]) + line_spacing
    return y


def _render_scene_card_image(card: dict, output_path: str, width: int = VIDEO_SIZE[0], height: int = VIDEO_SIZE[1]) -> None:
    """Render a single scene card to a PNG image with measured, conflict-aware layouts."""
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise ImportError("Pillow is required for video export. Install pillow.") from exc

    image = Image.new("RGB", (width, height), color=(10, 16, 28))
    draw = ImageDraw.Draw(image)
    title_font, body_font, small_font = _load_fonts()
    visual_type = card.get("visual_type", "overview_map")
    visual_motif = card.get("visual_motif", "diagram")
    entities = card.get("entities") or []
    relations = card.get("relations") or []
    blocked_rects: list[tuple[int, int, int, int]] = []

    def register(rect: tuple[int, int, int, int], padding: int = 10) -> None:
        blocked_rects.append(_rect_with_padding(rect, padding))

    def draw_node(rect: tuple[int, int, int, int], label: str, fill, outline, text_fill, max_lines: int = 3) -> tuple[int, int, int, int]:
        draw.rounded_rectangle(rect, radius=20, fill=fill, outline=outline, width=3)
        _draw_fitted_text(draw, label, (rect[0] + 18, rect[1] + 16, rect[2] - 18, rect[3] - 16), body_font, text_fill, align='center', valign='middle', max_lines=max_lines)
        register(rect)
        return rect

    def draw_relation_label(anchor: tuple[int, int], text_label: str, fill_color, text_color) -> None:
        label = _clean_keyword(text_label, 20)
        if not label:
            return
        font, lines, line_height = _fit_text_block(draw, label, 180, 72, small_font, min_size=14, max_lines=2)
        if not lines:
            return
        label_width = max(draw.textbbox((0, 0), line, font=font)[2] for line in lines) + 24
        label_height = len(lines) * line_height + max(0, len(lines) - 1) * 8 + 18
        label_rect = _find_clear_label_rect((label_width, label_height), anchor, blocked_rects, (width, height), padding=10)
        draw.rounded_rectangle(label_rect, radius=14, fill=fill_color, outline=text_color, width=2)
        _draw_fitted_text(draw, label, (label_rect[0] + 12, label_rect[1] + 9, label_rect[2] - 12, label_rect[3] - 9), font, text_color, align='center', valign='middle', max_lines=2)
        register(label_rect, padding=6)

    def connect(rect1: tuple[int, int, int, int], rect2: tuple[int, int, int, int], color, label: str) -> None:
        x1, y1 = rect1[2], (rect1[1] + rect1[3]) // 2
        x2, y2 = rect2[0], (rect2[1] + rect2[3]) // 2
        draw.line((x1, y1, x2, y2), fill=color, width=5)
        draw.polygon([(x2, y2), (x2 - 16, y2 - 10), (x2 - 16, y2 + 10)], fill=color)
        draw_relation_label(((x1 + x2) // 2, (y1 + y2) // 2), label, (18, 32, 52), color)

    draw.rectangle((0, 0, width, height), fill=(10, 16, 28))
    draw.rectangle((0, 0, width, 112), fill=(14, 24, 40))
    draw.text((56, 28), card["subtitle"], font=small_font, fill=(165, 186, 214))
    title_lines = _fit_text_lines(draw, card["title"], title_font, width - 112, 2)
    title_y = 66
    for line in title_lines:
        draw.text((56, title_y), line, font=title_font, fill=(244, 248, 255))
        title_y += 54
    register((40, 20, width - 40, 130), padding=4)

    if visual_type == "overview_map":
        labels = [item.get("label", "Node") for item in entities[:4]] or ["Repo", "Core path", "Users", "Outputs"]
        rects = [
            (96, 210, 332, 338),
            (392, 168, 700, 312),
            (392, 356, 700, 500),
            (884, 240, 1188, 384),
        ]
        node_map = {}
        palette = [((24, 40, 68), (93, 155, 255), (238, 244, 255)), ((20, 48, 54), (94, 214, 160), (240, 246, 255)), ((60, 41, 25), (255, 168, 76), (252, 241, 228)), ((54, 41, 82), (201, 141, 255), (244, 237, 255))]
        for idx, label in enumerate(labels[:len(rects)]):
            rect = draw_node(rects[idx], label, *palette[idx], max_lines=3)
            node_map[label] = rect
        for relation in relations[:3]:
            if relation.get("from") in node_map and relation.get("to") in node_map:
                connect(node_map[relation["from"]], node_map[relation["to"]], (93, 155, 255), relation.get("type", "flows"))
        _draw_boxed_keywords(draw, "Keywords", card.get("keywords", []), (760, 430, 1192, 644), small_font, small_font, ((19, 35, 57), (82, 122, 172), (120, 223, 191)))
        _draw_boxed_keywords(draw, "Tech", card.get("tech_chips", []), (84, 520, 520, 650), small_font, small_font, ((20, 36, 60), (93, 155, 255), (176, 204, 236)))

    elif visual_type == "core_diagram":
        labels = [item.get("label", "Core") for item in entities[:3]] or card.get("focus_modules")[:3] or ["Core A", "Core B", "Core C"]
        box_w = 300
        gap = 54
        left = 78
        top = 270
        node_rects = []
        for i, label in enumerate(labels[:3]):
            x1 = left + i * (box_w + gap)
            rect = (x1, top, x1 + box_w, top + 128)
            node_rects.append(draw_node(rect, label, (21, 53, 66), (94, 214, 160), (240, 246, 255), max_lines=3))
        node_map = {labels[i]: node_rects[i] for i in range(min(len(labels), len(node_rects)))}
        for relation in relations[:3]:
            if relation.get("from") in node_map and relation.get("to") in node_map:
                connect(node_map[relation["from"]], node_map[relation["to"]], (94, 214, 160), relation.get("type", "calls"))
        _draw_boxed_keywords(draw, "Core jobs", card.get("keywords", []), (84, 454, 500, 648), small_font, small_font, ((20, 44, 54), (94, 214, 160), (166, 225, 206)))
        _draw_boxed_keywords(draw, "Signals", card.get("microcopy", []), (540, 454, 1194, 648), small_font, small_font, ((20, 44, 54), (94, 214, 160), (166, 225, 206)))

    elif visual_type == "expansion_ladder" and visual_motif == "dialogue":
        speaker_a = entities[0].get("label", "Core path") if entities else "Core path"
        speaker_b = entities[1].get("label", "Expansion") if len(entities) > 1 else "Expansion"
        draw.ellipse((120, 292, 284, 456), fill=(24, 54, 72), outline=(93, 155, 255), width=3)
        draw.ellipse((992, 292, 1156, 456), fill=(72, 47, 27), outline=(255, 168, 76), width=3)
        register((120, 292, 284, 456))
        register((992, 292, 1156, 456))
        _draw_fitted_text(draw, speaker_a, (86, 470, 320, 550), body_font, (235, 241, 248), align='center', valign='top', max_lines=2)
        _draw_fitted_text(draw, speaker_b, (958, 470, 1190, 550), body_font, (252, 241, 228), align='center', valign='top', max_lines=2)
        left_bubble = (286, 224, 620, 372)
        right_bubble = (660, 224, 994, 372)
        draw.rounded_rectangle(left_bubble, radius=22, fill=(26, 40, 67), outline=(93, 155, 255), width=3)
        draw.rounded_rectangle(right_bubble, radius=22, fill=(60, 41, 25), outline=(255, 168, 76), width=3)
        register(left_bubble)
        register(right_bubble)
        left_text = relations[0].get("type", "hands off") if relations else (card.get("keywords") or ["core handoff"])[0]
        right_text = (card.get("keywords") or ["new capability"])[1] if len(card.get("keywords") or []) > 1 else (card.get("keywords") or ["new capability"])[0]
        _draw_fitted_text(draw, left_text, (left_bubble[0] + 22, left_bubble[1] + 20, left_bubble[2] - 22, left_bubble[3] - 20), body_font, (238, 244, 255), align='center', valign='middle', max_lines=3)
        _draw_fitted_text(draw, right_text, (right_bubble[0] + 22, right_bubble[1] + 20, right_bubble[2] - 22, right_bubble[3] - 20), body_font, (252, 241, 228), align='center', valign='middle', max_lines=3)
        _draw_boxed_keywords(draw, "Adds", card.get("keywords", []), (388, 478, 892, 652), small_font, small_font, ((60, 41, 25), (255, 168, 76), (255, 214, 173)))

    elif visual_type == "expansion_ladder" and visual_motif == "analogy":
        analogy_labels = [item.get("label", "Capability") for item in entities[:3]] or ["Backbone", "Extension", "Outcome"]
        rects = [(90, 286, 350, 482), (500, 286, 780, 482), (890, 286, 1170, 482)]
        colors = ((72, 47, 27), (255, 168, 76), (252, 241, 228))
        for rect, label in zip(rects, analogy_labels[:3]):
            draw_node(rect, label, *colors, max_lines=4)
        connect(rects[0], rects[1], (255, 168, 76), relations[0].get('type', 'extends') if relations else 'extends')
        connect(rects[1], rects[2], (255, 168, 76), relations[1].get('type', 'enables') if len(relations) > 1 else 'enables')
        _draw_boxed_keywords(draw, "Analogy", card.get("microcopy", []), (220, 520, 1040, 650), small_font, small_font, ((60, 41, 25), (255, 168, 76), (255, 214, 173)))

    elif visual_type == "expansion_ladder":
        core_rect = draw_node((86, 256, 340, 520), entities[1].get('label', 'Core path') if len(entities) > 1 else 'Core path', (24, 54, 72), (93, 155, 255), (235, 241, 248), max_lines=3)
        module_label = entities[0].get('label', 'Expansion') if entities else (card.get('focus_modules') or ['Expansion Module'])[0]
        expansion_rect = draw_node((514, 250, 1160, 408), module_label, (72, 47, 27), (255, 168, 76), (252, 241, 228), max_lines=3)
        connect(core_rect, expansion_rect, (255, 168, 76), relations[0].get('type', 'extends') if relations else 'extends')
        _draw_boxed_keywords(draw, "Adds", card.get("keywords", []), (514, 434, 842, 652), small_font, small_font, ((60, 41, 25), (255, 168, 76), (255, 214, 173)))
        _draw_boxed_keywords(draw, "Why it matters", card.get("microcopy", []), (870, 434, 1192, 652), small_font, small_font, ((60, 41, 25), (255, 168, 76), (255, 214, 173)))

    else:
        draw.text((88, 244), "Complete system", font=body_font, fill=(240, 231, 252))
        use_cases = card.get("use_cases") or [item.get("label", "Use case") for item in entities[:3]] or ["Primary audience", "Main workflow", "Expected value"]
        card_rects = [(86, 320, 386, 544), (490, 320, 790, 544), (894, 320, 1194, 544)]
        for rect, label in zip(card_rects, use_cases[:3]):
            draw_node(rect, label, (54, 41, 82), (201, 141, 255), (244, 237, 255), max_lines=4)
        _draw_boxed_keywords(draw, "Takeaway", card.get("keywords", []), (86, 574, 780, 652), small_font, small_font, ((54, 41, 82), (201, 141, 255), (229, 204, 255)))

    draw.text((56, height - 48), card["footer"], font=small_font, fill=(140, 157, 184))
    image.save(output_path, format="PNG")


# ---------------------------------------------------------------------------
# Phase 3 helpers: video composition
# ---------------------------------------------------------------------------

def _build_scene_clip(image_path: str, duration: float, audio_path: Optional[str] = None):
    """Create a moviepy clip from a rendered image card, optionally with TTS audio."""
    try:
        from moviepy import ImageClip, AudioFileClip
    except ImportError:
        try:
            from moviepy.editor import ImageClip, AudioFileClip
        except ImportError as exc:
            raise ImportError("moviepy is required for video export. Install moviepy.") from exc

    clip = ImageClip(image_path)
    if hasattr(clip, "with_duration"):
        clip = clip.with_duration(duration)
    else:
        clip = clip.set_duration(duration)

    if audio_path and os.path.exists(audio_path):
        try:
            audio_clip = AudioFileClip(audio_path)
            if hasattr(clip, "with_audio"):
                clip = clip.with_audio(audio_clip)
            else:
                clip = clip.set_audio(audio_clip)
            logger.info("Audio attached to scene clip: %s (%.2fs)", audio_path, audio_clip.duration)
        except Exception as e:
            logger.warning("Failed to attach audio to scene clip: %s", e)

    if hasattr(clip, "fadein"):
        clip = clip.fadein(TRANSITION_SECONDS).fadeout(TRANSITION_SECONDS)
    return clip


def _compose_final_video(clips: List[Any], output_path: str, has_audio: bool = False) -> None:
    """Concatenate scene clips and write the final MP4 to disk."""
    try:
        from moviepy import concatenate_videoclips
    except ImportError:
        try:
            from moviepy.editor import concatenate_videoclips
        except ImportError as exc:
            raise ImportError("moviepy is required for video export. Install moviepy.") from exc

    try:
        final_clip = concatenate_videoclips(clips, method="compose", padding=-TRANSITION_SECONDS)
    except TypeError:
        final_clip = concatenate_videoclips(clips, method="compose")
    try:
        write_kwargs = {
            "fps": VIDEO_FPS,
            "codec": "libx264",
            "preset": "ultrafast",
            "bitrate": "700k",
            "logger": None,
        }
        if has_audio:
            write_kwargs["audio"] = True
            write_kwargs["audio_codec"] = "aac"
            write_kwargs["audio_bitrate"] = "128k"
        else:
            write_kwargs["audio"] = False
        final_clip.write_videofile(output_path, **write_kwargs)
    except Exception as exc:
        raise RuntimeError(
            "Failed to render MP4 video. Ensure ffmpeg is installed and available to moviepy."
        ) from exc
    finally:
        try:
            final_clip.close()
        except Exception:
            pass
        for clip in clips:
            try:
                clip.close()
            except Exception:
                pass


def _read_file_bytes(path: str) -> bytes:
    with open(path, "rb") as fh:
        return fh.read()


# ---------------------------------------------------------------------------
# Phase 3 entry point: narration scenes -> MP4 bytes
# ---------------------------------------------------------------------------

async def render_video_from_analyzed(  # noqa: C901
    analyzed: "AnalyzedContent",
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> bytes:
    """
    Phase 2b-video plus Phase 3: AnalyzedContent to MP4 bytes.

    Renderer with TTS narration: generates scene-specific visuals and
    TTS audio, then composes them into an MP4 walkthrough.
    Falls back to silent video if TTS is unavailable.
    """
    overall_start = time.perf_counter()
    logger.info("Video export requested for %s", analyzed.repo_name)

    narration_start = time.perf_counter()
    raw_scenes = await generate_narration_script(analyzed, provider=provider, model=model)
    normalized_raw = _normalize_scenes(raw_scenes, analyzed.repo_name)
    scenes = _build_storyline_scenes(analyzed, normalized_raw)
    scenes = _normalize_scenes(scenes, analyzed.repo_name)
    logger.info("Timing - video narration + storyline planning completed in %.2fs", time.perf_counter() - narration_start)
    total = len(scenes)

    if total == 0:
        raise ValueError("No valid narration scenes were generated for video export.")

    with tempfile.TemporaryDirectory(prefix="repohelper_video_") as tmpdir:
        tmp_path = Path(tmpdir)

        # --- TTS audio generation ---
        tts_start = time.perf_counter()
        has_audio = False
        try:
            from api.tts_service import generate_all_scene_audio
            audio_paths = await generate_all_scene_audio(
                scenes, str(tmp_path), language=analyzed.language,
            )
            has_audio = any(p is not None for p in audio_paths)
            logger.info(
                "Timing - TTS generation completed in %.2fs (%d/%d scenes with audio)",
                time.perf_counter() - tts_start,
                sum(1 for p in audio_paths if p),
                total,
            )
        except Exception as e:
            logger.warning("TTS generation failed, falling back to silent video: %s", e)
            audio_paths = [None] * total

        # --- Update scene durations based on audio length ---
        for scene in scenes:
            audio_duration = scene.get("audio_duration")
            if audio_duration and audio_duration > 0:
                scene["duration_seconds"] = max(
                    SCENE_DURATION_MIN,
                    min(audio_duration + AUDIO_PADDING_SECONDS, SCENE_DURATION_MAX),
                )

        # --- Render scene images + build clips (multi-frame with subtitles) ---
        clips = []
        expansion_counter = 0
        use_playwright = True
        try:
            from api.scene_renderer import render_scene_to_png, close_browser
        except ImportError:
            use_playwright = False
            logger.warning("Playwright renderer not available, falling back to Pillow")

        try:
            from moviepy import ImageClip, AudioFileClip, concatenate_videoclips as _concat
        except ImportError:
            from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips as _concat

        for index, scene in enumerate(scenes, start=1):
            scene_start = time.perf_counter()
            logger.info("Rendering video scene %d/%d: %s", index, total, scene.get("title", f"Scene {index}"))
            card = _scene_to_card_content(scene, analyzed, index, total)
            card["narration"] = scene.get("narration", "")

            if scene.get("section") == "expansion":
                expansion_counter += 1

            scene_duration = scene["duration_seconds"]
            audio_path = audio_paths[index - 1] if index - 1 < len(audio_paths) else None
            segments = card.get("narration_segments") or [{"text": "", "highlight_labels": [], "duration_fraction": 1.0}]

            if use_playwright and len(segments) > 1:
                # Multi-frame path: one PNG per narration segment
                sub_clips = []
                for seg_idx, seg in enumerate(segments):
                    frame_path = tmp_path / f"scene_{index:02d}_f{seg_idx:02d}.png"
                    seg_duration = max(0.5, scene_duration * seg["duration_fraction"])
                    try:
                        await render_scene_to_png(
                            card, str(frame_path),
                            expansion_index=expansion_counter or 1,
                            subtitle_text=seg["text"],
                            highlight_labels=seg["highlight_labels"],
                        )
                    except Exception as render_err:
                        logger.warning("Playwright render failed for scene %d frame %d: %s", index, seg_idx, render_err)
                        _render_scene_card_image(card, str(frame_path))

                    sub_clip = ImageClip(str(frame_path))
                    if hasattr(sub_clip, "with_duration"):
                        sub_clip = sub_clip.with_duration(seg_duration)
                    else:
                        sub_clip = sub_clip.set_duration(seg_duration)
                    sub_clips.append(sub_clip)

                # Concatenate sub-frames (no fade within a scene)
                scene_clip = _concat(sub_clips, method="compose")

                # Attach audio to the scene-level clip
                if audio_path and os.path.exists(audio_path):
                    try:
                        audio_clip = AudioFileClip(audio_path)
                        if hasattr(scene_clip, "with_audio"):
                            scene_clip = scene_clip.with_audio(audio_clip)
                        else:
                            scene_clip = scene_clip.set_audio(audio_clip)
                    except Exception as e:
                        logger.warning("Failed to attach audio to scene %d: %s", index, e)

                # Apply fade transitions between scenes
                if hasattr(scene_clip, "fadein"):
                    scene_clip = scene_clip.fadein(TRANSITION_SECONDS).fadeout(TRANSITION_SECONDS)
                clips.append(scene_clip)
            else:
                # Single-frame path (Pillow fallback or single segment)
                image_path = tmp_path / f"scene_{index:02d}.png"
                subtitle_text = segments[0]["text"] if segments else ""
                highlight_labels = segments[0]["highlight_labels"] if segments else []

                if use_playwright:
                    try:
                        await render_scene_to_png(
                            card, str(image_path),
                            expansion_index=expansion_counter or 1,
                            subtitle_text=subtitle_text,
                            highlight_labels=highlight_labels,
                        )
                    except Exception as render_err:
                        logger.warning("Playwright render failed for scene %d, falling back to Pillow: %s", index, render_err)
                        _render_scene_card_image(card, str(image_path))
                else:
                    _render_scene_card_image(card, str(image_path))

                clips.append(_build_scene_clip(str(image_path), scene_duration, audio_path=audio_path))

            logger.info(
                "Timing - scene %d/%d rendered in %.2fs (%d frames, audio=%s)",
                index, total, time.perf_counter() - scene_start,
                len(segments), "yes" if audio_path else "no",
            )

        # Clean up Playwright browser
        if use_playwright:
            try:
                await close_browser()
            except Exception:
                pass

        output_path = tmp_path / "repo_overview.mp4"
        compose_start = time.perf_counter()
        logger.info("Composing final MP4 for %s with %d scenes (audio=%s)", analyzed.repo_name, total, has_audio)
        _compose_final_video(clips, str(output_path), has_audio=has_audio)
        logger.info("Timing - final MP4 composition completed in %.2fs", time.perf_counter() - compose_start)

        read_start = time.perf_counter()
        payload = _read_file_bytes(str(output_path))
        logger.info(
            "Final MP4 composed for %s (%d bytes, audio=%s, readback %.2fs, total %.2fs)",
            analyzed.repo_name,
            len(payload),
            has_audio,
            time.perf_counter() - read_start,
            time.perf_counter() - overall_start,
        )
        return payload


# ---------------------------------------------------------------------------
# Legacy compatibility wrapper
# ---------------------------------------------------------------------------

def render_video(summary_text: str, repo_name: str) -> bytes:
    """Backward-compatible wrapper for sync callers."""
    import asyncio
    from api.content_analyzer import AnalyzedContent

    analyzed = AnalyzedContent(repo_name=repo_name, project_overview=summary_text)
    return asyncio.run(render_video_from_analyzed(analyzed))

