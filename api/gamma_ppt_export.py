"""
Gamma PPT export integration.

Converts structured ``AnalyzedContent`` into a narrative outline, sends it to
Gamma's generation API, polls until the deck is ready, and downloads the
exported PPTX file.
"""

import asyncio
import json
import logging
import os
from typing import TYPE_CHECKING, Any, Dict, List

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
GAMMA_THEME_ID = os.environ.get("GAMMA_THEME_ID", "")
GAMMA_IMAGE_SOURCE = os.environ.get("GAMMA_IMAGE_SOURCE", "aiGenerated")
GAMMA_CARD_SPLIT = os.environ.get("GAMMA_CARD_SPLIT", "inputTextBreaks")
GAMMA_CARD_DIMENSIONS = os.environ.get("GAMMA_CARD_DIMENSIONS", "16x9")
GAMMA_IMAGE_STYLE = os.environ.get(
    "GAMMA_IMAGE_STYLE",
    (
        "premium enterprise presentation design, abstract gradient backgrounds, "
        "layered geometric shapes, modern technical editorial style, "
        "navy teal amber palette, high contrast, polished and presentation-ready"
    ),
)
GAMMA_ADDITIONAL_INSTRUCTIONS = os.environ.get("GAMMA_ADDITIONAL_INSTRUCTIONS", "").strip()

# Cached theme id resolved at runtime
_resolved_theme_id: str | None = None


# ---------------------------------------------------------------------------
# Theme helpers
# ---------------------------------------------------------------------------


async def _fetch_themes() -> List[Dict[str, Any]]:
    """Fetch available themes from Gamma workspace."""
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(
                f"{GAMMA_API_BASE}/themes",
                headers=_gamma_headers(),
            )
        if resp.status_code != 200:
            logger.warning("Gamma /themes returned %s", resp.status_code)
            return []
        data = resp.json()
        themes = data if isinstance(data, list) else data.get("themes", [])
        logger.info("Fetched %d Gamma themes", len(themes))
        return themes
    except Exception as exc:
        logger.warning("Failed to fetch Gamma themes: %s", exc)
        return []


def _select_theme(themes: List[Dict[str, Any]]) -> str:
    """Pick the best non-white theme from the list.

    Preference order:
    1. Themes whose name hints at dark / colourful / gradient style
    2. The first theme that is not explicitly named *blank* or *white*
    3. Fall back to the first theme in the list
    """
    if not themes:
        return ""

    preferred_keywords = (
        "dark", "gradient", "bold", "modern", "vibrant", "colorful",
        "colourful", "professional", "corporate", "elegant", "tech",
        "deep", "rich", "midnight", "ocean", "slate", "indigo",
    )
    skip_keywords = ("blank", "white", "plain", "minimal", "empty", "clean", "simple")

    for theme in themes:
        name = (theme.get("name") or "").lower()
        if any(kw in name for kw in preferred_keywords):
            logger.info("Auto-selected Gamma theme: %s (%s)", theme.get("name"), theme.get("id"))
            return theme.get("id") or theme.get("themeId", "")

    for theme in themes:
        name = (theme.get("name") or "").lower()
        if not any(kw in name for kw in skip_keywords):
            logger.info("Auto-selected Gamma theme (fallback): %s (%s)", theme.get("name"), theme.get("id"))
            return theme.get("id") or theme.get("themeId", "")

    first = themes[0]
    return first.get("id") or first.get("themeId", "")


async def _resolve_theme_id() -> str:
    """Return GAMMA_THEME_ID if set, otherwise auto-select from API."""
    global _resolved_theme_id

    if GAMMA_THEME_ID:
        return GAMMA_THEME_ID

    if _resolved_theme_id is not None:
        return _resolved_theme_id

    themes = await _fetch_themes()
    _resolved_theme_id = _select_theme(themes)
    return _resolved_theme_id


# ---------------------------------------------------------------------------
# Outline helpers
# ---------------------------------------------------------------------------

def _truncate(text: str, limit: int) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[: max(limit - 3, 0)].rstrip() + "..."


def _format_list(items: Any, limit: int = 6, numbered: bool = False) -> str:
    if not items:
        return ""
    if isinstance(items, str):
        return items.strip()

    values = [str(item).strip() for item in items if str(item).strip()]
    if not values:
        return ""
    values = values[:limit]

    if numbered:
        return "\n".join(f"{index + 1}. {value}" for index, value in enumerate(values))
    return "\n".join(f"- {value}" for value in values)


def _format_modules(modules: List[Any]) -> str:
    return "\n".join(
        f"- **{module.name}**: {module.responsibility}"
        for module in modules
        if getattr(module, "name", "").strip() or getattr(module, "responsibility", "").strip()
    )


def _default_additional_instructions(language: str) -> str:
    """Deck-level styling guidance so Gamma avoids plain white slides."""
    if (language or "").startswith("zh"):
        return (
            "【重要视觉要求】整个演示文稿必须使用精心设计的背景，严禁使用纯白色背景。"
            "每一页幻灯片都必须有丰富的背景设计：可以是深色系（如深蓝、深灰、墨绿、"
            "深紫等）、柔和渐变（如蓝紫渐变、青蓝渐变）、或带有几何图案装饰的彩色背景。"
            "文字颜色需要与背景形成鲜明对比以确保可读性。"
            "整体设计风格应保持统一、现代、专业，适合技术架构展示。"
            "请在每一页都使用有设计感的配色方案、合理的留白与排版层次。"
            "标题应醒目突出，正文应清晰易读，图表和列表应有结构感。"
        )
    return (
        "[CRITICAL VISUAL REQUIREMENT] The entire presentation MUST use designed "
        "backgrounds — never plain white. Every slide MUST have a rich background: "
        "dark tones (deep blue, charcoal, midnight navy, dark teal), soft gradients "
        "(blue-purple, teal-blue, warm amber), or patterned/textured colored backgrounds. "
        "Text colors must contrast sharply with backgrounds for readability. "
        "Maintain a unified, modern, and polished design language throughout. "
        "Use consistent color palettes, well-structured layouts with layered geometric "
        "accents, and strong visual hierarchy. Titles should be bold and prominent; "
        "body text clean and readable; lists and diagrams should feel structured."
    )


def _build_additional_instructions(language: str) -> str:
    instructions = _default_additional_instructions(language)
    if GAMMA_THEME_ID:
        if (language or "").startswith("zh"):
            instructions += " 请严格沿用已选择的 Gamma 主题，并在整套幻灯片中保持一致。"
        else:
            instructions += " Respect the selected Gamma theme and apply it consistently across the deck."
    if GAMMA_ADDITIONAL_INSTRUCTIONS:
        instructions = f"{instructions}\n\n{GAMMA_ADDITIONAL_INSTRUCTIONS}"
    return instructions


def _build_image_options() -> Dict[str, Any]:
    options: Dict[str, Any] = {"source": GAMMA_IMAGE_SOURCE}
    if GAMMA_IMAGE_SOURCE == "aiGenerated":
        options["style"] = GAMMA_IMAGE_STYLE
    return options


async def _build_generation_payload(
    input_text: str,
    language: str,
    num_cards: int = GAMMA_NUM_CARDS,
) -> Dict[str, Any]:
    """Build the Gamma payload in one place so it is easy to test."""
    gamma_lang = "zh" if (language or "").startswith("zh") else "en"
    theme_id = await _resolve_theme_id()

    payload: Dict[str, Any] = {
        "inputText": input_text,
        "textMode": "generate",
        "format": "presentation",
        "cardSplit": GAMMA_CARD_SPLIT,
        "exportAs": "pptx",
        "textOptions": {
            "tone": "professional",
            "audience": "developers and technical stakeholders",
            "amount": "medium",
            "language": gamma_lang,
        },
        "additionalInstructions": _build_additional_instructions(gamma_lang),
        "imageOptions": _build_image_options(),
        "cardOptions": {
            "dimensions": GAMMA_CARD_DIMENSIONS,
        },
    }

    if GAMMA_CARD_SPLIT == "auto":
        payload["numCards"] = num_cards

    if theme_id:
        payload["themeId"] = theme_id

    return payload


def build_gamma_outline(analyzed: "AnalyzedContent") -> str:
    """
    Convert ``AnalyzedContent`` into a slide outline.

    Gamma recognizes ``---`` as a card break when ``cardSplit`` is set to
    ``inputTextBreaks``.
    """
    is_zh = (analyzed.language or "").startswith("zh")
    slides: List[str] = []

    title_tagline = (
        "深入解析项目架构、技术栈与核心设计"
        if is_zh
        else "A Deep Dive into Architecture, Tech Stack & Core Design"
    )
    title_overview = _truncate(analyzed.project_overview, 180)
    title_body = f"\n\n{title_overview}" if title_overview else ""
    slides.append(f"{analyzed.repo_name}\n{title_tagline}{title_body}")

    if analyzed.project_overview:
        slides.append(
            (
                f"项目概览\n核心定位与价值主张\n\n{analyzed.project_overview}"
                if is_zh
                else f"Project Overview\nCore Purpose & Value Proposition\n\n{analyzed.project_overview}"
            )
        )

    if analyzed.architecture:
        narrative = analyzed.architecture[0]
        bullet_block = _format_list(analyzed.architecture[1:5], limit=4)
        content = narrative if not bullet_block else f"{narrative}\n\n{bullet_block}"
        slides.append(
            (
                f"架构设计\n系统的整体架构与设计哲学\n\n{content}"
                if is_zh
                else f"Architecture & Design\nSystem Architecture & Design Philosophy\n\n{content}"
            )
        )

    tech_groups: List[str] = []
    ts = analyzed.tech_stack
    if ts.languages:
        tech_groups.append(
            f"**{'编程语言' if is_zh else 'Languages'}**: {', '.join(ts.languages)}"
        )
    if ts.frameworks:
        tech_groups.append(
            f"**{'框架' if is_zh else 'Frameworks'}**: {', '.join(ts.frameworks)}"
        )
    if ts.key_libraries:
        tech_groups.append(
            f"**{'核心库' if is_zh else 'Key Libraries'}**: {', '.join(ts.key_libraries[:6])}"
        )
    if ts.infrastructure:
        tech_groups.append(
            f"**{'基础设施' if is_zh else 'Infrastructure'}**: {', '.join(ts.infrastructure)}"
        )
    if tech_groups:
        tech_body = "\n".join(f"- {item}" for item in tech_groups)
        slides.append(
            (
                f"技术栈\n驱动项目的核心技术\n\n{tech_body}"
                if is_zh
                else f"Technology Stack\nThe Technologies Powering This Project\n\n{tech_body}"
            )
        )

    if analyzed.key_modules:
        modules = analyzed.key_modules[:8]
        if len(modules) > 4:
            midpoint = len(modules) // 2
            first_half = _format_modules(modules[:midpoint])
            second_half = _format_modules(modules[midpoint:])
            slides.append(
                (
                    f"核心模块（上）\n系统的主要组件与职责\n\n{first_half}"
                    if is_zh
                    else f"Core Modules - Part 1\nPrimary Components & Responsibilities\n\n{first_half}"
                )
            )
            slides.append(
                (
                    f"核心模块（下）\n更多关键组件\n\n{second_half}"
                    if is_zh
                    else f"Core Modules - Part 2\nAdditional Key Components\n\n{second_half}"
                )
            )
        else:
            slides.append(
                (
                    f"核心模块\n关键组件与各自职责\n\n{_format_modules(modules)}"
                    if is_zh
                    else f"Core Modules\nKey Components & Their Responsibilities\n\n{_format_modules(modules)}"
                )
            )

    if analyzed.data_flow:
        data_flow = _format_list(analyzed.data_flow, limit=8, numbered=True)
        slides.append(
            (
                f"数据流\n数据在系统中的流转路径\n\n{data_flow}"
                if is_zh
                else f"Data Flow & Processing\nHow Data Moves Through the System\n\n{data_flow}"
            )
        )

    if analyzed.api_points:
        api_points = _format_list(analyzed.api_points, limit=6)
        slides.append(
            (
                f"API 与集成接口\n系统对外暴露的接口与集成方式\n\n{api_points}"
                if is_zh
                else f"API & Integration Points\nInterfaces & External Integrations\n\n{api_points}"
            )
        )

    if analyzed.component_hierarchy:
        component_hierarchy = _format_list(analyzed.component_hierarchy, limit=6)
        slides.append(
            (
                f"组件层级结构\n前端组件的组织与层级\n\n{component_hierarchy}"
                if is_zh
                else f"Component Hierarchy\nHow Frontend Components Are Organized\n\n{component_hierarchy}"
            )
        )

    if analyzed.deployment_info:
        deployment_info = _format_list(analyzed.deployment_info, limit=6)
        slides.append(
            (
                f"部署与基础设施\n生产环境的部署架构\n\n{deployment_info}"
                if is_zh
                else f"Deployment & Infrastructure\nProduction Deployment Architecture\n\n{deployment_info}"
            )
        )

    if analyzed.data_schemas:
        data_schemas = _format_list(analyzed.data_schemas, limit=6)
        slides.append(
            (
                f"数据模型\n核心数据结构与持久化设计\n\n{data_schemas}"
                if is_zh
                else f"Data Schemas & Models\nCore Data Structures & Persistence Design\n\n{data_schemas}"
            )
        )

    if analyzed.target_users:
        slides.append(
            (
                f"目标用户与应用场景\n谁在使用，如何使用\n\n{analyzed.target_users}"
                if is_zh
                else f"Target Users & Use Cases\nWho Uses It & How\n\n{analyzed.target_users}"
            )
        )

    slides.append(
        (
            f"感谢观看\n{analyzed.repo_name} - 技术全景概览"
            if is_zh
            else f"Thank You\n{analyzed.repo_name} - Technical Overview"
        )
    )

    outline = "\n---\n".join(slides)
    logger.info("Gamma outline built: %d slides, %d chars", len(slides), len(outline))
    return outline


# ---------------------------------------------------------------------------
# Gamma API helpers
# ---------------------------------------------------------------------------

def _gamma_headers() -> Dict[str, str]:
    """Build request headers for Gamma API."""
    if not GAMMA_API_KEY:
        raise RuntimeError(
            "GAMMA_API_KEY is not set - required for Gamma PPTX generation. "
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
    """Start an async Gamma generation and return the generation id."""
    gamma_lang = "zh" if (language or "").startswith("zh") else "en"
    payload = await _build_generation_payload(
        input_text=input_text,
        language=language,
        num_cards=num_cards,
    )

    logger.info(
        "Creating Gamma generation (split=%s, lang=%s, theme=%s)",
        GAMMA_CARD_SPLIT,
        gamma_lang,
        bool(GAMMA_THEME_ID),
    )

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
    """Poll Gamma until a generation completes or fails."""
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
                    response.status_code,
                    elapsed,
                    detail,
                )
                continue

            data = response.json()
            status = data.get("status", "unknown")

            if status == "completed":
                logger.info(
                    "Gamma generation %s completed after %ds",
                    generation_id,
                    elapsed,
                )
                return data

            if status == "failed":
                error_detail = data.get("error", data.get("message", "unknown error"))
                raise RuntimeError(f"Gamma generation failed: {error_detail}")

            logger.debug(
                "Gamma generation %s status=%s (elapsed=%ds)",
                generation_id,
                status,
                elapsed,
            )

    raise RuntimeError(
        f"Gamma generation timed out after {GAMMA_TIMEOUT}s "
        f"(generation_id={generation_id})"
    )


async def _download_pptx(export_url: str) -> bytes:
    """Download the PPTX file from Gamma's export URL."""
    logger.info("Downloading PPTX from Gamma export URL")

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
    """Render ``AnalyzedContent`` to PPTX bytes via Gamma."""
    outline = build_gamma_outline(analyzed)

    generation_id = await _create_generation(
        input_text=outline,
        language=analyzed.language,
        num_cards=GAMMA_NUM_CARDS,
    )
    result = await _poll_generation(generation_id)

    export_url = result.get("exportUrl")
    if not export_url:
        export_url = result.get("export", {}).get("url")
    if not export_url:
        raise RuntimeError(
            "Gamma generation completed but no exportUrl found in response: "
            f"{json.dumps(result, ensure_ascii=False)[:500]}"
        )

    return await _download_pptx(export_url)
