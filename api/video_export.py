"""
Video Export Module — Legacy Compatibility Wrapper

Delegates to the new api.video pipeline for actual video generation.

The original stub is replaced with a thin wrapper that calls
api.video.orchestrator.render_video_from_analyzed (async) and
api.video.narration.generate_narration_script for Phase 2b.
"""

import asyncio
import json
import logging
from typing import Any, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from api.content_analyzer import AnalyzedContent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Phase 2b-Video: Delegated to api.video.narration
# ---------------------------------------------------------------------------

def _analysis_to_prompt_json(analyzed: "AnalyzedContent") -> str:
    """Serialize the structured analysis into a compact JSON string for the prompt."""
    from api.video.narration import _analysis_to_prompt_json as _impl
    return _impl(analyzed)


async def generate_narration_script(
    analyzed: "AnalyzedContent",
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> List[dict]:
    """Phase 2b-Video: delegate to api.video.narration."""
    from api.video.narration import generate_narration_script as _impl
    return await _impl(analyzed, provider=provider, model=model)


def _parse_scene_array(raw_text: str) -> List[dict]:
    """Parse a JSON array of scene objects from raw LLM text."""
    from api.video.narration import _parse_scene_array as _impl
    return _impl(raw_text)


def _fallback_narration_script(analyzed: "AnalyzedContent") -> List[dict]:
    """Build a simple narration script directly from the structured data."""
    from api.video.storyline import _fallback_narration_script as _impl
    return _impl(analyzed)


# ---------------------------------------------------------------------------
# Phase 3: Video render — delegates to new video orchestrator
# ---------------------------------------------------------------------------

def render_video_from_analyzed(analyzed: "AnalyzedContent") -> bytes:
    """
    Phase 2b-Video + Phase 3: AnalyzedContent → MP4 bytes.

    Synchronous wrapper around the async video orchestrator pipeline.
    """
    from api.video.orchestrator import render_video_from_analyzed as _async_render
    return asyncio.run(_async_render(analyzed))


def render_video(summary_text: str, repo_name: str) -> bytes:
    """Backward-compatible entry point."""
    from api.content_analyzer import AnalyzedContent
    analyzed = AnalyzedContent(repo_name=repo_name, project_overview=summary_text)
    return render_video_from_analyzed(analyzed)

