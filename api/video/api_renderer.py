"""
External video generation API renderer (stub).

Future: accepts scenes/analyzed content, calls an external video generation
API (e.g. Runway, Kling, Sora), and returns MP4 bytes directly — bypassing
the Playwright/Pillow + TTS + MoviePy baseline pipeline.
"""

from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from api.content_analyzer import AnalyzedContent


async def render_via_api(
    analyzed: "AnalyzedContent",
    scenes: List[dict],
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
) -> bytes:
    """Generate video via external API. Not yet implemented."""
    raise NotImplementedError(
        "External video API rendering is not yet implemented. "
        "Use the baseline Playwright/Pillow pipeline instead."
    )
