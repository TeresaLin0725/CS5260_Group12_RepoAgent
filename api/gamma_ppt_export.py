"""
Gamma PPT Export Module — Gamma.app API Integration

Converts structured AnalyzedContent into a professional PPTX presentation
by calling Gamma's public generation API (https://public-api.gamma.app).

Phase 2b-Gamma: Converts AnalyzedContent directly into a structured text
    outline (no LLM call needed — the analysis is already structured).
Phase 3: Sends the outline to Gamma's /generations endpoint with
    exportAs=pptx, polls until completion, and downloads the PPTX file.
"""

import asyncio
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

GAMMA_API_BASE = "https://public-api.gamma.app/v1.0"
GAMMA_API_KEY = os.environ.get("GAMMA_API_KEY", "")
GAMMA_POLL_INTERVAL = int(os.environ.get("GAMMA_POLL_INTERVAL", "5"))
GAMMA_TIMEOUT = int(os.environ.get("GAMMA_TIMEOUT", "300"))
GAMMA_NUM_CARDS = int(os.environ.get("GAMMA_NUM_CARDS", "10"))


# ---------------------------------------------------------------------------
# Phase 2b-Gamma: structured AnalyzedContent → plain-text outline for Gamma
# ---------------------------------------------------------------------------

def build_gamma_outline(analyzed: "AnalyzedContent") -> str:
    """
    Phase 2b-Gamma: deterministically convert AnalyzedContent into a
    structured text outline suitable for Gamma's inputText field.

    No LLM call required — we already have the structured analysis from
    Phase 1+2a. Gamma will expand this outline into polished slides via
    textMode='condense'.

    Uses '---' as the slide separator that Gamma recognises.
    """
    is_zh = (analyzed.language or "").startswith("zh")
    sep = "\n---\n"
    slides: List[str] = []

    # ── Slide 1: Title ──────────────────────────────────────────────────────
    slides.append(analyzed.repo_name)

    # ── Slide 2: Project Overview ───────────────────────────────────────────
    if analyzed.project_overview:
        title = "项目概述" if is_zh else "Project Overview"
        slides.append(f"{title}\n{analyzed.project_overview}")

    # ── Slide 3: Architecture ───────────────────────────────────────────────
    if analyzed.architecture:
        title = "架构设计" if is_zh else "Architecture & Design"
        bullets = "\n".join(f"- {item}" for item in analyzed.architecture)
        slides.append(f"{title}\n{bullets}")

    # ── Slide 4: Tech Stack ──────────────────────────────────────────────────
    ts = analyzed.tech_stack
    tech_lines: List[str] = []
    if ts.languages:
        label = "语言" if is_zh else "Languages"
        tech_lines.append(f"- {label}: {', '.join(ts.languages)}")
    if ts.frameworks:
        label = "框架" if is_zh else "Frameworks"
        tech_lines.append(f"- {label}: {', '.join(ts.frameworks)}")
    if ts.key_libraries:
        label = "核心库" if is_zh else "Key Libraries"
        tech_lines.append(f"- {label}: {', '.join(ts.key_libraries[:6])}")
    if ts.infrastructure:
        label = "基础设施" if is_zh else "Infrastructure"
        tech_lines.append(f"- {label}: {', '.join(ts.infrastructure)}")
    if tech_lines:
        title = "技术栈" if is_zh else "Tech Stack"
        slides.append(f"{title}\n" + "\n".join(tech_lines))

    # ── Slide 5: Key Modules ─────────────────────────────────────────────────
    if analyzed.key_modules:
        title = "核心模块" if is_zh else "Core Modules"
        bullets = "\n".join(
            f"- {m.name}: {m.responsibility}"
            for m in analyzed.key_modules[:7]
        )
        slides.append(f"{title}\n{bullets}")

    # ── Slide 6: Data Flow ───────────────────────────────────────────────────
    if analyzed.data_flow:
        title = "数据流" if is_zh else "Data Flow"
        bullets = "\n".join(f"- {step}" for step in analyzed.data_flow)
        slides.append(f"{title}\n{bullets}")

    # ── Slide 7: API & Integration Points ───────────────────────────────────
    if analyzed.api_points:
        title = "API 接入点" if is_zh else "API & Integration Points"
        bullets = "\n".join(f"- {pt}" for pt in analyzed.api_points)
        slides.append(f"{title}\n{bullets}")

    # ── Slide 8 (optional): Component Hierarchy (webapp) ────────────────────
    if analyzed.component_hierarchy:
        title = "组件层次" if is_zh else "Component Hierarchy"
        items = analyzed.component_hierarchy
        if isinstance(items, list):
            bullets = "\n".join(f"- {c}" for c in items[:6])
        else:
            bullets = str(items)
        slides.append(f"{title}\n{bullets}")

    # ── Slide 9 (optional): Deployment (microservice) ───────────────────────
    if analyzed.deployment_info:
        title = "部署与基础设施" if is_zh else "Deployment & Infrastructure"
        items = analyzed.deployment_info
        if isinstance(items, list):
            bullets = "\n".join(f"- {d}" for d in items[:6])
        else:
            bullets = str(items)
        slides.append(f"{title}\n{bullets}")

    # ── Slide 10 (optional): Data Schemas (data_pipeline) ───────────────────
    if analyzed.data_schemas:
        title = "数据模型" if is_zh else "Data Schemas & Models"
        items = analyzed.data_schemas
        if isinstance(items, list):
            bullets = "\n".join(f"- {s}" for s in items[:6])
        else:
            bullets = str(items)
        slides.append(f"{title}\n{bullets}")

    # ── Slide 11: Target Users ───────────────────────────────────────────────
    if analyzed.target_users:
        title = "目标用户" if is_zh else "Target Users & Use Cases"
        slides.append(f"{title}\n{analyzed.target_users}")

    outline = sep.join(slides)
    logger.info(
        "Gamma outline built: %d slides, %d chars",
        len(slides), len(outline),
    )
    return outline


# ---------------------------------------------------------------------------
# Phase 3: Gamma API — create generation, poll, download PPTX
# ---------------------------------------------------------------------------

def _gamma_headers() -> Dict[str, str]:
    """Build request headers for Gamma API."""
    if not GAMMA_API_KEY:
        raise RuntimeError(
            "GAMMA_API_KEY is not set — required for Gamma PPTX generation. "
            "Get one from your Gamma account settings: Account > API Keys."
        )
    return {
        "X-API-KEY": GAMMA_API_KEY,
        "Content-Type": "application/json",
    }


async def _create_generation(
    input_text: str,
    language: str,
    num_cards: int = GAMMA_NUM_CARDS,
) -> str:
    """
    POST /generations — start an async generation job on Gamma.

    Returns the generationId for polling.
    """
    # Map our language codes to Gamma's textOptions.language
    gamma_lang = "zh" if (language or "").startswith("zh") else "en"

    payload = {
        "inputText": input_text,
        "textMode": "condense",
        "format": "presentation",
        "numCards": num_cards,
        "exportAs": "pptx",
        "textOptions": {
            "tone": "professional",
            "audience": "developers and technical stakeholders",
            "amount": "detailed",
            "language": gamma_lang,
        },
        "imageOptions": {
            "source": "webFreeToUseCommercially",
        },
    }

    logger.info("Creating Gamma generation (numCards=%d, lang=%s) …", num_cards, gamma_lang)

    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(
            f"{GAMMA_API_BASE}/generations",
            json=payload,
            headers=_gamma_headers(),
        )

    if response.status_code not in (200, 201):
        detail = response.text[:500]
        logger.error("Gamma /generations returned %s: %s", response.status_code, detail)
        raise RuntimeError(
            f"Gamma generation creation failed (HTTP {response.status_code}): {detail}"
        )

    data = response.json()
    generation_id = data.get("generationId") or data.get("id")
    if not generation_id:
        raise RuntimeError(f"Gamma response missing generationId: {data}")

    logger.info("Gamma generation created: %s", generation_id)
    return generation_id


async def _poll_generation(generation_id: str) -> Dict[str, Any]:
    """
    GET /generations/{id} — poll until status is 'completed' or 'failed'.

    Returns the full response dict on completion.
    """
    headers = _gamma_headers()
    elapsed = 0

    async with httpx.AsyncClient(timeout=30) as client:
        while elapsed < GAMMA_TIMEOUT:
            await asyncio.sleep(GAMMA_POLL_INTERVAL)
            elapsed += GAMMA_POLL_INTERVAL

            response = await client.get(
                f"{GAMMA_API_BASE}/generations/{generation_id}",
                headers=headers,
            )

            if response.status_code != 200:
                detail = response.text[:500]
                logger.warning(
                    "Gamma poll returned %s (elapsed=%ds): %s",
                    response.status_code, elapsed, detail,
                )
                continue

            data = response.json()
            status = data.get("status", "unknown")

            if status == "completed":
                logger.info(
                    "Gamma generation %s completed after %ds",
                    generation_id, elapsed,
                )
                return data

            if status == "failed":
                error_detail = data.get("error", data.get("message", "unknown error"))
                raise RuntimeError(
                    f"Gamma generation failed: {error_detail}"
                )

            logger.debug(
                "Gamma generation %s status=%s (elapsed=%ds)",
                generation_id, status, elapsed,
            )

    raise RuntimeError(
        f"Gamma generation timed out after {GAMMA_TIMEOUT}s "
        f"(generation_id={generation_id})"
    )


async def _download_pptx(export_url: str) -> bytes:
    """Download the PPTX file from the signed export URL."""
    logger.info("Downloading PPTX from Gamma export URL …")

    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.get(export_url)

    if response.status_code != 200:
        detail = response.text[:500]
        raise RuntimeError(
            f"Failed to download Gamma PPTX (HTTP {response.status_code}): {detail}"
        )

    pptx_bytes = response.content
    logger.info("Gamma PPTX downloaded: %d bytes", len(pptx_bytes))
    return pptx_bytes


async def render_gamma_ppt_from_analyzed(analyzed: "AnalyzedContent") -> bytes:
    """
    Async entry point: AnalyzedContent → PPTX bytes via Gamma.

    Builds a structured outline directly from AnalyzedContent (no LLM call),
    then sends it to Gamma API for generation and downloads the PPTX.
    """
    # Phase 2b: build outline directly from structured analysis (no LLM needed)
    outline = build_gamma_outline(analyzed)

    # Phase 3: send to Gamma, poll, download
    generation_id = await _create_generation(
        input_text=outline,
        language=analyzed.language,
        num_cards=GAMMA_NUM_CARDS,
    )
    result = await _poll_generation(generation_id)

    export_url = result.get("exportUrl")
    if not export_url:
        # Fallback: try to find in nested structure
        export_url = result.get("export", {}).get("url")
    if not export_url:
        raise RuntimeError(
            f"Gamma generation completed but no exportUrl found in response: "
            f"{json.dumps(result, ensure_ascii=False)[:500]}"
        )

    pptx_bytes = await _download_pptx(export_url)
    return pptx_bytes
