"""
Export Service — Unified Orchestration Layer

Routes analyzed content to the appropriate renderer (PDF, PPT, Video).
This is the single entry point for all export operations.

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
    WikiAnalysisRequest,
    RepoAnalysisRequest,
    WikiPageInput,
    analyze_wiki_content,
    analyze_repo_content,
)

logger = logging.getLogger(__name__)


class ExportFormat(str, Enum):
    """Supported export formats."""
    PDF = "pdf"
    PPT = "ppt"
    VIDEO = "video"


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


def _render_ppt(analyzed: AnalyzedContent) -> ExportResult:
    """Phase 2b-PPT (structured → slide outline) + Phase 3 (pptx render)."""
    from api.ppt_export import render_ppt_from_analyzed
    from datetime import datetime

    ppt_bytes = render_ppt_from_analyzed(analyzed)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_name = analyzed.repo_name.replace("/", "_")
    filename = f"{short_name}_slides_{timestamp}.pptx"

    return ExportResult(
        content_bytes=ppt_bytes,
        filename=filename,
        media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
    )


def _render_video(analyzed: AnalyzedContent) -> ExportResult:
    """Phase 2b-Video (LLM narration script) + Phase 3 (mp4 render)."""
    from api.video_export import render_video_from_analyzed
    from datetime import datetime

    video_bytes = render_video_from_analyzed(analyzed)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_name = analyzed.repo_name.replace("/", "_")
    filename = f"{short_name}_overview_{timestamp}.mp4"

    return ExportResult(
        content_bytes=video_bytes,
        filename=filename,
        media_type="video/mp4",
    )


_RENDERERS = {
    ExportFormat.PDF: _render_pdf,
    ExportFormat.PPT: _render_ppt,
    ExportFormat.VIDEO: _render_video,
}


def _print_analyzed_content(analyzed: AnalyzedContent, fmt: ExportFormat):
    """Pretty-print the AnalyzedContent structured fields to console & log."""
    import json

    # Build a dict of the structured fields (exclude raw_llm_text & summary_text)
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

async def export_wiki(
    request: WikiAnalysisRequest,
    fmt: ExportFormat = ExportFormat.PDF,
) -> ExportResult:
    """
    Full pipeline: wiki pages → analysis → render to the requested format.
    """
    logger.info("Export wiki as %s for %s", fmt.value, request.repo_name or request.repo_url)

    analyzed = await analyze_wiki_content(request)

    # ── Print AnalyzedContent (LLM structured output) ──
    _print_analyzed_content(analyzed, fmt)

    renderer = _RENDERERS[fmt]
    result = renderer(analyzed)

    logger.info("Export complete — format=%s, %d bytes", fmt.value, len(result.content_bytes))
    return result


async def export_repo(
    request: RepoAnalysisRequest,
    fmt: ExportFormat = ExportFormat.PDF,
) -> ExportResult:
    """
    Full pipeline: repo embeddings → analysis → render to the requested format.
    """
    logger.info("Export repo as %s for %s", fmt.value, request.repo_name or request.repo_url)

    analyzed = await analyze_repo_content(request)

    # ── Print AnalyzedContent (LLM structured output) ──
    _print_analyzed_content(analyzed, fmt)

    renderer = _RENDERERS[fmt]
    result = renderer(analyzed)

    logger.info("Export complete — format=%s, %d bytes", fmt.value, len(result.content_bytes))
    return result
