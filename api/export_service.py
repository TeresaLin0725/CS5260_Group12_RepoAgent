"""
Export Service - Unified Orchestration Layer.

Routes analyzed content to the appropriate renderer (PDF, PPT, Video).
This is the single entry point for all export operations.
"""

import inspect
import logging
import time
from enum import Enum

from pydantic import BaseModel, Field

from api.content_analyzer import AnalyzedContent, RepoAnalysisRequest, analyze_repo_content

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


def _render_pdf(analyzed: AnalyzedContent) -> ExportResult:
    """Phase 2b-PDF plus Phase 3 PDF render."""
    from datetime import datetime
    from api.pdf_export import render_pdf_from_analyzed

    pdf_bytes = render_pdf_from_analyzed(analyzed)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_name = analyzed.repo_name.replace("/", "_")
    filename = f"{short_name}_summary_{timestamp}.pdf"
    return ExportResult(content_bytes=pdf_bytes, filename=filename, media_type="application/pdf")


def _render_ppt(analyzed: AnalyzedContent) -> ExportResult:
    """Phase 2b-PPT plus Phase 3 PPT render."""
    from datetime import datetime
    from api.ppt_export import render_ppt_from_analyzed

    ppt_bytes = render_ppt_from_analyzed(analyzed)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_name = analyzed.repo_name.replace("/", "_")
    filename = f"{short_name}_slides_{timestamp}.pptx"
    return ExportResult(
        content_bytes=ppt_bytes,
        filename=filename,
        media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
    )


async def _render_video(analyzed: AnalyzedContent, job_id: str | None = None) -> ExportResult:
    """Phase 2b-video plus Phase 3 MP4 render."""
    from datetime import datetime
    from api.video_export import render_video_from_analyzed

    video_bytes = await render_video_from_analyzed(analyzed, job_id=job_id)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_name = analyzed.repo_name.replace("/", "_")
    filename = f"{short_name}_overview_{timestamp}.mp4"
    return ExportResult(content_bytes=video_bytes, filename=filename, media_type="video/mp4")


_RENDERERS = {
    ExportFormat.PDF: _render_pdf,
    ExportFormat.PPT: _render_ppt,
    ExportFormat.VIDEO: _render_video,
}


def _print_analyzed_content(analyzed: AnalyzedContent, fmt: ExportFormat) -> None:
    """Pretty-print the AnalyzedContent structured fields to console and log."""
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
        "deployment_info": analyzed.deployment_info,
        "component_hierarchy": analyzed.component_hierarchy,
        "data_schemas": analyzed.data_schemas,
    }

    json_str = json.dumps(data, ensure_ascii=False, indent=2)
    header = f"\n{'=' * 60}\nAnalyzedContent (input to {fmt.value.upper()} renderer)\n{'=' * 60}"
    print(header)
    print(json_str)
    print("=" * 60 + "\n")
    logger.info("AnalyzedContent for %s export:\n%s", fmt.value, json_str)


async def export_repo(
    request: RepoAnalysisRequest,
    fmt: ExportFormat = ExportFormat.PDF,
    job_id: str | None = None,
) -> ExportResult:
    """Full pipeline: repo embeddings to analysis to render."""
    overall_start = time.perf_counter()
    logger.info("Export repo as %s for %s", fmt.value, request.repo_name or request.repo_url)

    analysis_start = time.perf_counter()
    analyzed = await analyze_repo_content(request)
    analysis_elapsed = time.perf_counter() - analysis_start
    logger.info("Timing - export analysis completed in %.2fs", analysis_elapsed)
    _print_analyzed_content(analyzed, fmt)

    renderer = _RENDERERS[fmt]
    render_start = time.perf_counter()
    # Pass job_id to video renderer for progress tracking
    if fmt == ExportFormat.VIDEO and job_id:
        result = renderer(analyzed, job_id=job_id)
    else:
        result = renderer(analyzed)
    if inspect.isawaitable(result):
        result = await result
    render_elapsed = time.perf_counter() - render_start

    total_elapsed = time.perf_counter() - overall_start
    logger.info(
        "Export complete - format=%s, %d bytes, render=%.2fs, total=%.2fs",
        fmt.value,
        len(result.content_bytes),
        render_elapsed,
        total_elapsed,
    )
    return result
