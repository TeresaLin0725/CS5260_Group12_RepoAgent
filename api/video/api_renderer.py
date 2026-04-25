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
    """Build a concise text prompt for the technical-explainer style video.

    This is the original API-mode prompt: shows software UI, dashboards,
    code, with narration-driven scenes. Suited for users who already
    understand what the project is.
    """
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


def _build_onboard_video_prompt(analyzed: "AnalyzedContent") -> str:
    """Build a beginner-friendly "show the OUTCOME" video prompt.

    Strategy: the most powerful hook for a beginner is showing what the
    tool *produces*, not its architecture. We describe a 5-shot story:
    URL pasted in → AI thinking → final artifacts (PDF/video/poster)
    appearing → happy user.

    Reads analyzed.onboard if present (one_liner, mental_model_3_boxes)
    to ground the visuals in the real project. Falls back to generic
    "code → docs" framing when onboard is missing.
    """
    onboard = getattr(analyzed, "onboard", None)
    one_liner = onboard.one_liner if (onboard and onboard.one_liner) else (
        analyzed.project_overview[:100] if analyzed.project_overview else "a software project"
    )
    boxes = (onboard.mental_model_3_boxes if (onboard and onboard.mental_model_3_boxes)
             else ["input", "AI processes", "output"])
    box_str = " → ".join(boxes[:3])

    parts = [
        f"A 15-second beginner-friendly product demo video for {analyzed.repo_name}, "
        f"a tool that {one_liner.lower().rstrip('.')}. Mental model: {box_str}. "
        f"The video has 5 quick shots showing the OUTCOME, not technical details:",
        "",
        "Shot 1 (3s): Close-up of hands typing a GitHub URL into a search bar on a clean modern web app. "
        "Soft lighting, shallow depth of field, hopeful mood.",
        "",
        "Shot 2 (3s): The screen shows a friendly progress animation — abstract glowing dots flowing "
        "through a stylized neural-network shape, suggesting 'AI is thinking'. No readable text.",
        "",
        "Shot 3 (3s): A beautifully designed PDF document fades in on the right, "
        "next to a short video player on the left, both arriving with smooth motion. "
        "The user smiles in the corner.",
        "",
        "Shot 4 (3s): Camera pulls back showing all three artifacts — PDF, video, infographic poster — "
        "arranged like a portfolio on a clean desk surface. Warm sunlight.",
        "",
        "Shot 5 (3s): Final hero shot — a relaxed person holding a tablet showing the PDF, "
        "smiling, in a coffee shop or library. Soft bokeh background.",
        "",
        "Style: cinematic, warm and inviting, soft pastel palette with gentle teal and amber accents, "
        "professional product-demo polish. NO readable text or UI labels (AI cannot render text well). "
        "Each shot has a clear hard cut to the next. The story is INPUT → MAGIC → OUTPUT → JOY.",
    ]
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
    mode: str = "default",
) -> bytes:
    """Generate video via external API.

    Args:
        analyzed: Structured repo analysis.
        scenes: Narration scenes (used to build the default-mode prompt).
        provider: "fal" (more providers can be added later).
        model: Model ID override. Uses provider default if not set.
        api_key: API key override. Read from env if not set.
        mode: "default" for the technical-explainer prompt (uses scenes),
              "onboard" for the beginner-friendly outcome-first prompt
              (uses analyzed.onboard).

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

    if mode == "onboard":
        prompt = _build_onboard_video_prompt(analyzed)
    else:
        prompt = _build_video_prompt(analyzed, scenes)
    logger.info("API video render: provider=%s, model=%s", provider, model)

    render_fn = _PROVIDERS.get(provider)
    if not render_fn:
        raise ValueError(f"Unknown provider '{provider}'. Supported: {list(_PROVIDERS.keys())}")

    return await render_fn(prompt, model, api_key)
