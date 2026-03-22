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
import textwrap
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional

if TYPE_CHECKING:
    from api.content_analyzer import AnalyzedContent

logger = logging.getLogger(__name__)

VIDEO_SIZE = (1280, 720)
SCENE_DURATION_DEFAULT = 12
SCENE_DURATION_MIN = 6
SCENE_DURATION_MAX = 24
MAX_SCENES = 8
MAX_BULLETS = 4
MAX_BULLET_CHARS = 140


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

    language_name = _get_language_name(analyzed.language)
    analysis_json = _analysis_to_prompt_json(analyzed)
    prompt = VIDEO_NARRATION_PROMPT.format(
        language_name=language_name,
        analysis_json=analysis_json,
    )

    raw_text = await _call_llm_raw(
        prompt=prompt,
        provider=provider or "openai",
        model=model,
    )

    scenes = _parse_scene_array(raw_text)
    if not scenes:
        scenes = _fallback_narration_script(analyzed)

    logger.info("Narration script generated - %d scenes", len(scenes))
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


def _fallback_narration_script(analyzed: "AnalyzedContent") -> List[dict]:
    """Build a simple narration script directly from the structured data."""
    scenes: list[dict] = []

    scenes.append(
        {
            "title": analyzed.repo_name,
            "narration": (
                f"Welcome. Let's walk through {analyzed.repo_name}. "
                f"{analyzed.project_overview[:200] if analyzed.project_overview else ''}"
            ).strip(),
            "duration_seconds": 15,
        }
    )

    if analyzed.architecture:
        narration = "Here is the high-level architecture. " + " ".join(analyzed.architecture[:3])
        scenes.append({"title": "Architecture", "narration": narration[:400], "duration_seconds": 20})

    if analyzed.tech_stack and (analyzed.tech_stack.languages or analyzed.tech_stack.frameworks):
        items = analyzed.tech_stack.languages + analyzed.tech_stack.frameworks
        narration = "The project uses: " + ", ".join(items[:6]) + "."
        scenes.append({"title": "Tech Stack", "narration": narration, "duration_seconds": 15})

    if analyzed.key_modules:
        mods = ", ".join(m.name for m in analyzed.key_modules[:5])
        narration = f"The key modules are: {mods}."
        scenes.append({"title": "Key Modules", "narration": narration, "duration_seconds": 15})

    if analyzed.data_flow:
        narration = "Let's trace the data flow. " + " Then, ".join(analyzed.data_flow[:4]) + "."
        scenes.append({"title": "Data Flow", "narration": narration[:400], "duration_seconds": 20})

    scenes.append(
        {
            "title": "Thank You",
            "narration": f"That is a quick overview of {analyzed.repo_name}. Thanks for watching!",
            "duration_seconds": 10,
        }
    )

    return scenes


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
                "narration": narration[:1200],
                "duration_seconds": duration,
            }
        )

    if normalized:
        return normalized

    # Keep the renderer robust even if the LLM returns unusable data.
    # A single generic scene is better than failing the whole export path.
    fallback_title = repo_name or "Repository Overview"
    return [
        {
            "title": fallback_title,
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


def _scene_to_card_content(scene: dict, repo_name: str, index: int, total: int) -> dict:
    """Create a simple visual card representation for one scene."""
    # Baseline v1 uses a deterministic card layout:
    # title + subtitle + short bullets + footer progress indicator.
    return {
        "title": scene["title"],
        "subtitle": repo_name or "Repository Walkthrough",
        "bullets": _split_narration_to_bullets(scene["narration"]),
        "footer": f"{repo_name or 'Repo'}  |  Scene {index}/{total}",
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
    """Render a single scene card to a PNG image."""
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise ImportError("Pillow is required for video export. Install pillow.") from exc

    image = Image.new("RGB", (width, height), color=(11, 18, 32))
    draw = ImageDraw.Draw(image)
    title_font, body_font, small_font = _load_fonts()

    # The baseline visual language is intentionally simple:
    # one framed card, one accent bar, and text-heavy content for clarity.
    draw.rounded_rectangle(
        (50, 50, width - 50, height - 50),
        radius=32,
        fill=(19, 32, 56),
        outline=(64, 98, 142),
        width=3,
    )
    draw.rectangle((90, 120, width - 90, 126), fill=(62, 207, 142))

    y = 165
    draw.text((90, y), card["subtitle"], font=small_font, fill=(155, 193, 229))
    y += 48
    y = _draw_wrapped_text(draw, card["title"], title_font, 90, y, (242, 247, 255), line_spacing=14)
    y += 18

    for bullet in card["bullets"]:
        draw.ellipse((96, y + 10, 112, y + 26), fill=(62, 207, 142))
        y = _draw_wrapped_text(draw, bullet, body_font, 132, y, (222, 230, 242), line_spacing=12)
        y += 12

    draw.text((90, height - 95), card["footer"], font=small_font, fill=(140, 157, 184))
    image.save(output_path, format="PNG")


# ---------------------------------------------------------------------------
# Phase 3 helpers: video composition
# ---------------------------------------------------------------------------

def _build_scene_clip(image_path: str, duration: int):
    """Create a moviepy clip from a rendered image card."""
    try:
        from moviepy import ImageClip
    except ImportError:
        try:
            from moviepy.editor import ImageClip
        except ImportError as exc:
            raise ImportError("moviepy is required for video export. Install moviepy.") from exc

    clip = ImageClip(image_path)
    if hasattr(clip, "with_duration"):
        clip = clip.with_duration(duration)
    else:
        clip = clip.set_duration(duration)

    if hasattr(clip, "fadein"):
        clip = clip.fadein(0.25).fadeout(0.25)
    return clip


def _compose_final_video(clips: List[Any], output_path: str) -> None:
    """Concatenate scene clips and write the final MP4 to disk."""
    try:
        from moviepy import concatenate_videoclips
    except ImportError:
        try:
            from moviepy.editor import concatenate_videoclips
        except ImportError as exc:
            raise ImportError("moviepy is required for video export. Install moviepy.") from exc

    final_clip = concatenate_videoclips(clips, method="compose")
    try:
        final_clip.write_videofile(output_path, fps=24, codec="libx264", audio=False, logger=None)
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

async def render_video_from_analyzed(
    analyzed: "AnalyzedContent",
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> bytes:
    """
    Phase 2b-video plus Phase 3: AnalyzedContent to MP4 bytes.

    Baseline v1 renderer: create slide-style cards from the narration script
    and compose them into an MP4 walkthrough.
    """
    logger.info("Video export requested for %s - baseline renderer", analyzed.repo_name)

    raw_scenes = await generate_narration_script(analyzed, provider=provider, model=model)
    scenes = _normalize_scenes(raw_scenes, analyzed.repo_name)
    total = len(scenes)

    if total == 0:
        raise ValueError("No valid narration scenes were generated for video export.")

    with tempfile.TemporaryDirectory(prefix="repohelper_video_") as tmpdir:
        tmp_path = Path(tmpdir)
        clips = []

        # Render each narration scene into a static card image, then wrap that
        # image in a timed movie clip. Baseline v1 stays deterministic on
        # purpose so the export always reflects the actual repo analysis.
        for index, scene in enumerate(scenes, start=1):
            card = _scene_to_card_content(scene, analyzed.repo_name, index, total)
            image_path = tmp_path / f"scene_{index:02d}.png"
            _render_scene_card_image(card, str(image_path))
            clips.append(_build_scene_clip(str(image_path), scene["duration_seconds"]))

        output_path = tmp_path / "repo_overview.mp4"
        _compose_final_video(clips, str(output_path))
        return _read_file_bytes(str(output_path))


# ---------------------------------------------------------------------------
# Legacy compatibility wrapper
# ---------------------------------------------------------------------------

def render_video(summary_text: str, repo_name: str) -> bytes:
    """Backward-compatible wrapper for sync callers."""
    import asyncio
    from api.content_analyzer import AnalyzedContent

    analyzed = AnalyzedContent(repo_name=repo_name, project_overview=summary_text)
    return asyncio.run(render_video_from_analyzed(analyzed))

