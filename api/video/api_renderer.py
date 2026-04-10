"""
External video generation API renderer.

Generates video by calling external text-to-video APIs instead of the
baseline Playwright/Pillow + TTS + MoviePy pipeline.

Supported providers:
  - "fal": fal.ai platform (Kling v3, Seedance, etc.)

Set FAL_API_KEY in .env to enable.
"""

import asyncio
import logging
import os
import time
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from api.content_analyzer import AnalyzedContent

logger = logging.getLogger(__name__)

# Provider → default model
_DEFAULT_MODELS = {
    "fal": "fal-ai/kling-video/v3/standard/text-to-video",
}


def _build_video_prompt(analyzed: "AnalyzedContent", scenes: List[dict]) -> str:
    """Build a concise text prompt for the video generation API."""
    parts = []

    overview = analyzed.project_overview or analyzed.repo_name or "software project"
    parts.append(f"Create a short explainer video about {analyzed.repo_name}: {overview}")

    for i, scene in enumerate(scenes[:6], 1):
        narration = scene.get("narration", "")
        title = scene.get("title", f"Scene {i}")
        if narration:
            parts.append(f"Scene {i} ({title}): {narration[:120]}")

    parts.append(
        "Style: professional tech explainer, clean motion graphics, "
        "dark background with colored highlights, smooth transitions."
    )

    prompt = "\n".join(parts)
    if len(prompt) > 1500:
        prompt = prompt[:1497] + "..."
    return prompt


# ---------------------------------------------------------------------------
# fal.ai provider
# ---------------------------------------------------------------------------

async def _render_fal(prompt: str, model: str, api_key: str) -> bytes:
    """Call fal.ai API for text-to-video."""
    import httpx

    os.environ["FAL_KEY"] = api_key

    try:
        import fal_client
    except ImportError:
        raise ImportError("fal-client required. Run: pip install fal-client")

    logger.info("fal.ai text-to-video: model=%s, prompt=%d chars", model, len(prompt))
    start = time.perf_counter()

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: fal_client.subscribe(
            model,
            arguments={
                "prompt": prompt,
                "duration": "5",
                "aspect_ratio": "16:9",
                "generate_audio": False,
                "negative_prompt": "blur, distort, low quality, text overlay",
            },
            with_logs=True,
        ),
    )

    video_url = result["video"]["url"]
    logger.info("fal.ai video ready at %s, downloading...", video_url)

    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.get(video_url)
        resp.raise_for_status()

    elapsed = time.perf_counter() - start
    logger.info("fal.ai video: %d bytes in %.1fs", len(resp.content), elapsed)
    return resp.content


# ---------------------------------------------------------------------------
# Provider registry
# ---------------------------------------------------------------------------

_PROVIDERS = {
    "fal": _render_fal,
}


async def render_via_api(
    analyzed: "AnalyzedContent",
    scenes: List[dict],
    provider: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> bytes:
    """Generate video via external API.

    Args:
        analyzed: Structured repo analysis.
        scenes: Narration scenes (used to build the video prompt).
        provider: "fal" (more providers can be added later).
        model: Model ID override. Uses provider default if not set.
        api_key: API key override. Read from env if not set.

    Returns:
        Raw MP4 bytes.
    """
    # Auto-detect provider
    if not provider:
        if os.environ.get("FAL_KEY") or os.environ.get("FAL_API_KEY"):
            provider = "fal"
        else:
            raise ValueError(
                "No video API provider configured. Set FAL_API_KEY in .env"
            )

    # Resolve API key
    if not api_key:
        if provider == "fal":
            api_key = os.environ.get("FAL_KEY") or os.environ.get("FAL_API_KEY", "")

    if not api_key:
        raise ValueError(f"No API key found for provider '{provider}'")

    model = model or _DEFAULT_MODELS.get(provider, "")
    if not model:
        raise ValueError(f"No default model for provider '{provider}'")

    prompt = _build_video_prompt(analyzed, scenes)
    logger.info("API video render: provider=%s, model=%s", provider, model)

    render_fn = _PROVIDERS.get(provider)
    if not render_fn:
        raise ValueError(f"Unknown provider '{provider}'. Supported: {list(_PROVIDERS.keys())}")

    return await render_fn(prompt, model, api_key)
