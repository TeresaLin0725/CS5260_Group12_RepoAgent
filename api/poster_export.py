"""
Poster Export Module — Gemini Image Generation (NanoBanana)

Converts structured AnalyzedContent into an illustrated poster
by calling Google's Gemini image generation API.

Phase 2b-Poster: Uses an LLM call (POSTER_LAYOUT_PROMPT) to rewrite the
    structured analysis into a poster layout specification with
    sections, highlights, and visual hints.
Phase 3: Builds an image generation prompt from the layout and calls
    Gemini's gemini-3.1-flash-image-preview model to produce a PNG poster.
"""

import base64
import json
import logging
import os
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from api.content_analyzer import AnalyzedContent

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GEMINI_IMAGE_MODEL = os.environ.get(
    "GEMINI_IMAGE_MODEL", "gemini-3.1-flash-image-preview"
)
GEMINI_IMAGE_TIMEOUT = int(os.environ.get("GEMINI_IMAGE_TIMEOUT", "300"))


# ---------------------------------------------------------------------------
# Phase 2b-Poster: structured AnalyzedContent → poster layout (LLM call)
# ---------------------------------------------------------------------------

def _analysis_to_poster_payload(analyzed: "AnalyzedContent") -> Dict[str, Any]:
    """Serialize the structured analysis into a poster-friendly payload."""
    payload: Dict[str, Any] = {
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
        "key_modules": [
            {"name": m.name, "responsibility": m.responsibility}
            for m in analyzed.key_modules
        ],
        "data_flow": analyzed.data_flow,
        "api_points": analyzed.api_points,
        "target_users": analyzed.target_users,
    }
    if analyzed.deployment_info:
        payload["deployment_info"] = analyzed.deployment_info
    if analyzed.component_hierarchy:
        payload["component_hierarchy"] = analyzed.component_hierarchy
    return payload


async def generate_poster_layout(
    analyzed: "AnalyzedContent",
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Phase 2b-Poster: call the LLM with POSTER_LAYOUT_PROMPT to produce a
    poster layout specification.

    Returns a list of section dicts:
        [{"title": str, "content": str, "visual_hint": str}, ...]
    """
    from api.content_analyzer import _get_language_name, _extract_json_from_llm, _call_llm

    language_name = _get_language_name(analyzed.language)
    analysis_json = json.dumps(
        _analysis_to_poster_payload(analyzed), ensure_ascii=False, indent=2
    )

    from api.prompts import POSTER_LAYOUT_PROMPT

    prompt = POSTER_LAYOUT_PROMPT.format(
        language=language_name,
        analysis_json=analysis_json,
    )

    raw = await _call_llm(prompt, provider=provider, model=model)
    sections = _extract_json_from_llm(raw)

    if not isinstance(sections, list):
        logger.warning("Poster layout LLM returned non-list; wrapping.")
        sections = [sections] if sections else []

    return sections


# ---------------------------------------------------------------------------
# Phase 3: Gemini image generation
# ---------------------------------------------------------------------------

def _build_image_prompt(
    repo_name: str,
    language: str,
    sections: List[Dict[str, Any]],
) -> str:
    """Build a detailed image generation prompt from the poster layout."""
    lang_instruction = (
        "海报中的所有文字必须使用中文。"
        if (language or "").startswith("zh")
        else "All text in the poster must be in English."
    )

    sections_text = "\n".join(
        f"- **{s.get('title', '')}**: {s.get('content', '')} "
        f"(Visual: {s.get('visual_hint', '')})"
        for s in sections
    )

    return (
        f'Generate a beautiful, professional infographic poster for the '
        f'software project "{repo_name}". {lang_instruction}\n\n'
        f'The poster should be a vertical, portrait-oriented technical '
        f'infographic with a modern, clean design. Use a cohesive color '
        f'palette with clear section divisions.\n\n'
        f'Include these sections:\n{sections_text}\n\n'
        f'Design requirements:\n'
        f'- Professional typography with clear hierarchy\n'
        f'- Icons or small illustrations for each section\n'
        f'- A banner/header area with the project name prominently displayed\n'
        f'- Visual connectors or flow lines between related sections\n'
        f'- A clean, modern tech aesthetic with high contrast for readability\n'
        f'- No placeholder text — all content must come from the sections above'
    )


async def _call_gemini_image(prompt: str) -> bytes:
    """
    Call Gemini image generation API and return PNG image bytes.

    Uses the gemini-3.1-flash-image-preview model via Google's
    generativelanguage REST API.
    """
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY is not set — required for poster generation"
        )

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_IMAGE_MODEL}:generateContent"
    )

    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key,
    }

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "responseModalities": ["image", "text"],
        },
    }

    logger.info("Calling Gemini image model %s …", GEMINI_IMAGE_MODEL)

    async with httpx.AsyncClient(timeout=GEMINI_IMAGE_TIMEOUT) as client:
        response = await client.post(url, json=payload, headers=headers)

    if response.status_code != 200:
        detail = response.text[:500]
        logger.error(
            "Gemini image API returned %s: %s", response.status_code, detail
        )
        raise RuntimeError(
            f"Gemini image generation failed "
            f"(HTTP {response.status_code}): {detail}"
        )

    result = response.json()

    # Extract image from response:
    # candidates[0].content.parts[] → find part with inlineData
    candidates = result.get("candidates", [])
    if not candidates:
        raise RuntimeError("Gemini returned no candidates for image generation")

    parts = candidates[0].get("content", {}).get("parts", [])
    for part in parts:
        inline_data = part.get("inlineData")
        if inline_data and inline_data.get("mimeType", "").startswith("image/"):
            image_bytes = base64.b64decode(inline_data["data"])
            logger.info(
                "Poster image generated: %d bytes, mime=%s",
                len(image_bytes),
                inline_data["mimeType"],
            )
            return image_bytes

    raise RuntimeError(
        "No image found in Gemini response — model may have returned text only"
    )


async def render_poster_from_analyzed(analyzed: "AnalyzedContent") -> bytes:
    """
    Async entry point: AnalyzedContent → poster image bytes.

    Runs the LLM layout generation + Gemini image render.
    """
    sections = await generate_poster_layout(analyzed)
    prompt = _build_image_prompt(
        repo_name=analyzed.repo_name,
        language=analyzed.language,
        sections=sections,
    )
    return await _call_gemini_image(prompt)
