"""
Export Service — Unified Orchestration Layer

Routes analyzed content to the appropriate renderer (PDF, PPT, Video, Poster).
PPT always uses the Gamma API for professionally designed presentations.

Architecture:
    Request → ContentAnalyzer (Phase 2a: structured JSON)
            → Format Adapter (Phase 2b: format-specific transform)
            → Renderer (Phase 3: binary output)
"""

import logging
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from api.content_analyzer import (
    AnalyzedContent,
    RepoAnalysisRequest,
    analyze_repo_content,
)

logger = logging.getLogger(__name__)


class ExportFormat(str, Enum):
    """Supported export formats."""
    PDF = "pdf"
    PPT = "ppt"
    VIDEO = "video"
    POSTER = "poster"


class ExportResult(BaseModel):
    """The result of an export operation."""
    content_bytes: bytes = Field(..., description="The rendered binary content")
    filename: str = Field(..., description="Suggested filename for download")
    media_type: str = Field(..., description="MIME type for the response")


# ---------------------------------------------------------------------------
# Phase 2b adapters + Phase 3 renderer dispatch
# ---------------------------------------------------------------------------

def _render_pdf(analyzed: AnalyzedContent) -> ExportResult:
    """Phase 2b-PDF (template assembly) + Phase 3 (fpdf2 render)."""
    from api.pdf_export import render_pdf_from_analyzed
    from datetime import datetime

    pdf_bytes = render_pdf_from_analyzed(analyzed)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_name = analyzed.repo_name.replace("/", "_")
    filename = f"{short_name}_summary_{timestamp}.pdf"

    return ExportResult(
        content_bytes=pdf_bytes,
        filename=filename,
        media_type="application/pdf",
    )


async def _render_ppt(analyzed: AnalyzedContent) -> ExportResult:
    """Phase 2b-PPT (outline) + Phase 3 (Gamma API -> PPTX)."""
    from api.gamma_ppt_export import render_gamma_ppt_from_analyzed
    from datetime import datetime

    pptx_bytes = await render_gamma_ppt_from_analyzed(analyzed)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_name = analyzed.repo_name.replace("/", "_")
    filename = f"{short_name}_slides_{timestamp}.pptx"

    return ExportResult(
        content_bytes=pptx_bytes,
        filename=filename,
        media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
    )


def _render_video(analyzed: AnalyzedContent) -> ExportResult:
    """Phase 2b-Video (LLM narration script) + Phase 3 (mp4 render) — stub, see async version."""
    raise NotImplementedError("Use _render_video_async for video export")


async def _render_video_async(analyzed: AnalyzedContent) -> ExportResult:
    """Phase 2b-Video (narration + storyline) + Phase 3 (MP4 composition)."""
    from api.video.orchestrator import render_video_from_analyzed
    from datetime import datetime

    video_bytes = await render_video_from_analyzed(analyzed)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_name = analyzed.repo_name.replace("/", "_")
    filename = f"{short_name}_overview_{timestamp}.mp4"

    return ExportResult(
        content_bytes=video_bytes,
        filename=filename,
        media_type="video/mp4",
    )


async def _render_poster(analyzed: AnalyzedContent) -> ExportResult:
    """Phase 2b-Poster (layout spec) + Phase 3 (NanoBanana render)."""
    from api.poster_export import render_poster_from_analyzed
    from datetime import datetime

    poster_bytes = await render_poster_from_analyzed(analyzed)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_name = analyzed.repo_name.replace("/", "_")
    filename = f"{short_name}_poster_{timestamp}.png"

    return ExportResult(
        content_bytes=poster_bytes,
        filename=filename,
        media_type="image/png",
    )


_RENDERERS = {
    ExportFormat.PDF: _render_pdf,
    ExportFormat.PPT: _render_ppt,
    ExportFormat.VIDEO: _render_video,
    ExportFormat.POSTER: _render_poster,
}


def _print_analyzed_content(analyzed: AnalyzedContent, fmt: ExportFormat):
    """Pretty-print the AnalyzedContent structured fields to console & log."""
    import json

    data = {
        "repo_name": analyzed.repo_name,
        "repo_url": analyzed.repo_url,
        "language": analyzed.language,
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
        "deployment_info": analyzed.deployment_info,
        "component_hierarchy": analyzed.component_hierarchy,
        "data_schemas": analyzed.data_schemas,
    }

    json_str = json.dumps(data, ensure_ascii=False, indent=2)
    header = f"\n{'='*60}\nAnalyzedContent (input to {fmt.value.upper()} renderer)\n{'='*60}"
    print(header)
    print(json_str)
    print('='*60 + '\n')
    logger.info("AnalyzedContent for %s export:\n%s", fmt.value, json_str)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def export_repo(
    request: RepoAnalysisRequest,
    fmt: ExportFormat = ExportFormat.PDF,
) -> ExportResult:
    """
    Full pipeline: repo embeddings -> analysis -> render to the requested format.
    """
    logger.info("Export repo as %s for %s", fmt.value, request.repo_name or request.repo_url)

    analyzed = await analyze_repo_content(request)
    _print_analyzed_content(analyzed, fmt)

    if fmt == ExportFormat.VIDEO:
        result = await _render_video_async(analyzed)
    else:
        renderer = _RENDERERS[fmt]
        import inspect
        result = renderer(analyzed)
        if inspect.isawaitable(result):
            result = await result

    logger.info("Export complete — format=%s, %d bytes", fmt.value, len(result.content_bytes))
    return result
