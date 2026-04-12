"""
External video generation API renderer.

Generates a short (10s) AI video via fal.ai using shot-by-shot
script format — concrete UI/code visuals, fast-paced cuts.
"""

import asyncio
import logging
import os
import time
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from api.content_analyzer import AnalyzedContent

logger = logging.getLogger(__name__)

_MODELS = {
    "wan": "fal-ai/wan-25-preview/text-to-video",
    "kling": "fal-ai/kling-video/v3/standard/text-to-video",
}
_DEFAULT_MODEL = os.getenv("VIDEO_API_MODEL", "kling")


# ---------------------------------------------------------------------------
# Shot-by-shot prompt builder
# ---------------------------------------------------------------------------

def _build_video_prompt(analyzed: "AnalyzedContent", scenes: List[dict]) -> str:
    """Build a shot-by-shot video prompt from analyzed content.

    Converts narration scenes into concrete visual shots that AI video
    models can render well: UI screens, code views, architecture diagrams,
    abstract data flows. Avoids on-screen text (models render it as gibberish).
    """
    name = analyzed.repo_name or "software project"
    overview = (analyzed.project_overview or "")[:200]

    # Gather concrete details
    tech = ", ".join(
        (analyzed.tech_stack.languages + analyzed.tech_stack.frameworks)[:3]
    ) if analyzed.tech_stack else ""

    modules = []
    for m in (analyzed.key_modules or [])[:4]:
        modules.append(m.name if hasattr(m, "name") else str(m))

    # Build shots from scenes (max 5 shots for 10s video)
    shots = []

    # Shot 1: Hook
    shots.append(
        "Shot 1 (3s): Close-up of a browser search bar, a GitHub URL is typed in. "
        "Dark modern UI, cursor blinking. Shallow depth of field."
    )

    # Shot 2-4: Core content
    shot_templates = [
        lambda s, mods: (
            f"Shot {{n}} (3s): CUT TO dark-themed dashboard showing "
            f"{', '.join(mods[:3]) if mods else 'analysis panels'} as interactive cards. "
            f"Animated data flows between panels with glowing lines."
        ),
        lambda s, mods: (
            f"Shot {{n}} (3s): CUT TO split-screen — left shows scrolling source code, "
            f"right shows architecture diagram with {tech or 'tech stack'} icons "
            f"connected by animated arrows. Camera pans left to right."
        ),
        lambda s, mods: (
            f"Shot {{n}} (3s): CUT TO AI chat interface with real-time analysis "
            f"results streaming in. Charts and module cards animate into place."
        ),
    ]

    for i, tmpl in enumerate(shot_templates):
        scene = scenes[i + 1] if i + 1 < len(scenes) else {}
        shot_text = tmpl(scene, modules)
        shots.append(shot_text.format(n=i + 2))

    # Shot 5: Closure
    shots.append(
        "Shot 5 (3s): CUT TO wide shot — full application as holographic display "
        "floating in dark space. All panels glow softly. Lens flare, fade to dark."
    )

    prompt = (
        f"A 15-second tech product video with 5 distinct shots, hard cuts between each. Introducing {name}, {overview}. "
        + " ".join(shots)
        + " Each shot is a distinct scene with hard cuts between them. "
        + "Style: modern dark UI, blue and teal accents, professional motion graphics. No readable text on screen."
    )

    if len(prompt) > 1500:
        prompt = prompt[:1497] + "..."

    logger.info("Video API prompt (%d chars): %s...", len(prompt), prompt[:120])
    return prompt


# ---------------------------------------------------------------------------
# fal.ai API call
# ---------------------------------------------------------------------------

async def _call_fal(prompt: str, model_key: str, api_key: str, duration: str = "15") -> bytes:
    """Call fal.ai for text-to-video generation."""
    import httpx

    os.environ["FAL_KEY"] = api_key

    try:
        import fal_client
    except ImportError:
        raise ImportError("fal-client required. Run: pip install fal-client")

    model_id = _MODELS.get(model_key, _MODELS["wan"])
    logger.info("fal.ai: model=%s, duration=%ss", model_id, duration)
    start = time.perf_counter()

    if "wan" in model_id:
        arguments = {
            "prompt": prompt,
            "duration": duration,
            "aspect_ratio": "16:9",
            "resolution": "1080p",
            "negative_prompt": "readable text, words, letters, titles, watermark, blurry, low quality",
            "enable_prompt_expansion": True,
        }
    else:
        arguments = {
            "prompt": prompt,
            "duration": duration,
            "aspect_ratio": "16:9",
            "generate_audio": False,
            "negative_prompt": "readable text, words, letters, titles, watermark, blurry, low quality",
        }

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: fal_client.subscribe(model_id, arguments=arguments, with_logs=True),
    )

    video_url = result["video"]["url"]
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.get(video_url)
        resp.raise_for_status()

    elapsed = time.perf_counter() - start
    logger.info("fal.ai video: %d bytes in %.1fs", len(resp.content), elapsed)
    return resp.content


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def render_via_api(
    analyzed: "AnalyzedContent",
    scenes: List[dict],
    model: Optional[str] = None,
) -> bytes:
    """Generate video via fal.ai. Returns raw MP4 bytes (no audio — BGM added by orchestrator)."""
    api_key = os.environ.get("FAL_KEY") or os.environ.get("FAL_API_KEY", "")
    if not api_key:
        raise ValueError("FAL_API_KEY not set in .env")

    model_key = model or _DEFAULT_MODEL
    prompt = _build_video_prompt(analyzed, scenes)
    return await _call_fal(prompt, model_key, api_key)