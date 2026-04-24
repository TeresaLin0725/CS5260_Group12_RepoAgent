"""
PDF Rendering Module

Responsible ONLY for rendering structured summary text into PDF bytes (fpdf2).
The content analysis (Phase 1 + Phase 2) is handled by content_analyzer.py.
The orchestration is handled by export_service.py.

This module also exposes a backward-compatible wrapper function
(generate_direct_pdf) so existing API routes continue to work without changes.
"""

import logging
import os
import re
from io import BytesIO
from typing import List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class DirectPDFExportRequest(BaseModel):
    """Request for generating PDF directly from repo embeddings (no wiki needed)."""
    repo_url: str = Field(..., description="Repository URL or local path")
    repo_name: str = Field("", description="Short repo name (owner/repo)")
    provider: str = Field("ollama", description="LLM provider")
    model: Optional[str] = Field(None, description="LLM model name")
    language: str = Field("en", description="Target language code")
    repo_type: str = Field("github", description="Repository type")
    access_token: Optional[str] = Field(None, description="Access token for private repos")
    excluded_dirs: Optional[str] = Field(None, description="Comma-separated dirs to exclude")
    excluded_files: Optional[str] = Field(None, description="Comma-separated files to exclude")
    included_dirs: Optional[str] = Field(None, description="Comma-separated dirs to include")
    included_files: Optional[str] = Field(None, description="Comma-separated files to include")


# ---------------------------------------------------------------------------
# PDF Rendering (fpdf2)
# ---------------------------------------------------------------------------

def _has_cjk(text: str) -> bool:
    """Detect if text contains CJK characters."""
    for ch in text:
        cp = ord(ch)
        if (0x4E00 <= cp <= 0x9FFF or  # CJK Unified
            0x3400 <= cp <= 0x4DBF or  # CJK Extension A
            0x3000 <= cp <= 0x303F or  # CJK Symbols
            0x30A0 <= cp <= 0x30FF or  # Katakana
            0x3040 <= cp <= 0x309F or  # Hiragana
            0xAC00 <= cp <= 0xD7AF):   # Hangul
            return True
    return False


def _find_system_cjk_font() -> Optional[str]:
    """Try to locate a CJK-capable TTF font on the system."""
    candidates = []

    if os.name == "nt":
        windir = os.environ.get("WINDIR", r"C:\Windows")
        font_dir = os.path.join(windir, "Fonts")
        candidates = [
            os.path.join(font_dir, "msyh.ttc"),      # Microsoft YaHei
            os.path.join(font_dir, "simsun.ttc"),     # SimSun
            os.path.join(font_dir, "simhei.ttf"),     # SimHei
            os.path.join(font_dir, "malgun.ttf"),     # Malgun Gothic (Korean)
            os.path.join(font_dir, "meiryo.ttc"),     # Meiryo (Japanese)
            os.path.join(font_dir, "msgothic.ttc"),   # MS Gothic
        ]
    else:
        # Linux / Docker typical paths
        candidates = [
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/noto-cjk/NotoSansCJKsc-Regular.otf",
            "/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        ]

    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def _sanitize_text(text: str) -> str:
    """Replace non-latin1 characters that break built-in PDF fonts."""
    # Replace common unicode chars with ASCII equivalents
    replacements = {
        '\u200b': '',    # zero-width space (from _soft_wrap_long_tokens)
        '\u200c': '',    # zero-width non-joiner
        '\u200d': '',    # zero-width joiner
        '\ufeff': '',    # BOM / zero-width no-break space
        '\u2014': '-',   # em-dash
        '\u2013': '-',   # en-dash
        '\u2018': "'",   # left single quote
        '\u2019': "'",   # right single quote
        '\u201c': '"',   # left double quote
        '\u201d': '"',   # right double quote
        '\u2026': '...', # ellipsis
        '\u2022': '-',   # bullet
        '\u00a0': ' ',   # non-breaking space
        '\u2192': '->',  # right arrow
        '\u25a0': '-',   # black square
        '\u25a1': '-',   # white square
        '\u25a3': '-',   # white square containing black small square
        '\u25b2': '-',   # black up-pointing triangle
        '\u25b6': '-',   # black right-pointing triangle
        '\u25b7': '-',   # white right-pointing triangle
        '\u25c6': '-',   # black diamond
        '\u25c7': '-',   # white diamond
        '\u25cb': '-',   # white circle
        '\u25cf': '-',   # black circle
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    # Final safety net: strip any remaining non-latin-1 characters
    text = text.encode('latin-1', errors='replace').decode('latin-1')
    return text

def _wrap_to_width(pdf, s: str, max_w: float) -> list[str]:
    """
    Greedy wrap string into lines so that pdf.get_string_width(line) <= max_w.
    Works for CJK and long tokens (no spaces).
    """
    s = s.replace("\r", "")
    if not s:
        return [""]

    lines = []
    cur = ""

    for ch in s:
        # keep explicit newlines handled outside
        if ch == "\n":
            lines.append(cur)
            cur = ""
            continue

        # try append
        cand = cur + ch
        if pdf.get_string_width(cand) <= max_w:
            cur = cand
        else:
            # if cur empty and even one char too wide (rare), force place it
            if cur:
                lines.append(cur)
                cur = ch
            else:
                lines.append(ch)
                cur = ""

    if cur or not lines:
        lines.append(cur)

    return lines


def _soft_wrap_long_tokens(text: str) -> str:
    """
    Insert zero-width break opportunities into long tokens (URLs, paths, identifiers)
    so fpdf2 can wrap them instead of overflowing to the right.
    """
    ZWSP = "\u200b"  # zero-width space

    # 1) URLs / paths: allow break after common separators
    seps = ["/", "\\", ".", "_", "-", "?", "&", "=", ":", "#", "%", "@", "+", ","]
    for s in seps:
        text = text.replace(s, s + ZWSP)

    # 2) Very long continuous alnum chunks (identifiers) – break every N chars
    #    (keeps it safe for cases like "aaaaaaaa...." with no separators)
    def break_long(m):
        w = m.group(0)
        n = 18  # you can tune: 14~24
        return ZWSP.join(w[i:i+n] for i in range(0, len(w), n))

    text = re.sub(r"[A-Za-z0-9]{30,}", break_long, text)

    return text

def _strip_json_artifacts(text: str) -> str:
    """
    Last-resort cleanup: strip any remaining JSON syntax artifacts
    from text that will be rendered in the PDF. This handles the case
    where raw JSON leaked through upstream parsing.
    """
    # If the text looks like raw JSON (starts with { or contains lots of JSON keys),
    # do aggressive cleanup. Otherwise, do minimal cleanup.
    json_indicators = 0
    if re.search(r'^\s*\{', text):
        json_indicators += 1
    if re.search(r'"repo_type_hint"\s*:', text):
        json_indicators += 1
    if re.search(r'"project_overview"\s*:', text):
        json_indicators += 1
    if re.search(r'"architecture"\s*:', text):
        json_indicators += 1

    if json_indicators >= 2:
        # This looks like raw JSON — try to parse and convert
        try:
            import json as _json
            cleaned = re.sub(r"```(?:json)?\s*", "", text)
            cleaned = re.sub(r"```\s*$", "", cleaned)
            # Try repair
            from api.content_analyzer import _repair_json_string, _dict_to_readable_text
            repaired = _repair_json_string(cleaned)
            data = _json.loads(repaired)
            if isinstance(data, dict) and data:
                return _dict_to_readable_text(data)
        except Exception:
            pass

        # Regex fallback: strip JSON syntax
        _key_map = {
            "project_overview": "Project Overview:",
            "architecture": "Architecture & Design:",
            "tech_stack": "Tech Stack & Dependencies:",
            "key_modules": "Key Modules & Components:",
            "data_flow": "Data Flow & Processing:",
            "api_points": "API & Integration Points:",
            "target_users": "Target Users & Use Cases:",
            "deployment_info": "Deployment & Infrastructure:",
            "component_hierarchy": "Component Hierarchy:",
            "data_schemas": "Data Schemas & Models:",
        }
        for jk, header in _key_map.items():
            text = re.sub(rf'"\s*{jk}\s*"\s*:\s*', f"\n{header}\n", text, flags=re.IGNORECASE)
        # Remove skip keys
        for jk in ("repo_type_hint", "repo_name", "project_name"):
            text = re.sub(rf'"\s*{jk}\s*"\s*:\s*"[^"]*"\s*,?\s*', "", text, flags=re.IGNORECASE)
        # Remove inner dict keys (name, responsibility, languages, etc.)
        text = re.sub(r'"\s*name\s*"\s*:\s*"([^"]*)"', r'\1', text, flags=re.IGNORECASE)
        text = re.sub(r'"\s*responsibility\s*"\s*:\s*"([^"]*)"', r': \1', text, flags=re.IGNORECASE)
        for inner in ("languages", "frameworks", "key_libraries", "infrastructure"):
            text = re.sub(rf'"\s*{inner}\s*"\s*:\s*', '', text, flags=re.IGNORECASE)
        # Strip structural chars
        text = re.sub(r'^\s*\{\s*', '', text)
        text = re.sub(r'\s*\}\s*$', '', text)
        text = text.replace("[", "").replace("]", "")
        text = text.replace("{", "").replace("}", "")
        # Clean stray quotes
        text = re.sub(r'(?<!\w)"([^"]{3,})"(?!\w)', r'\1', text)
        # Clean trailing/leading commas from lines
        text = re.sub(r',\s*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*,\s*', '', text, flags=re.MULTILINE)
        # Collapse blank lines
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = text.strip()

    return text


def render_pdf(summary_text: str, repo_name: str) -> bytes:
    """
    Render plain-text summary into a polished single-page A4 PDF.
    Uses auto font-sizing to fit content on one page.
    Returns raw PDF bytes.

    This is the main public API for PDF rendering.
    """
    from fpdf import FPDF
    from fpdf.enums import XPos, YPos

    # --- Strip any JSON artifacts that leaked through ---
    summary_text = _strip_json_artifacts(summary_text)

    # --- Font setup ---
    need_cjk = _has_cjk(summary_text) or _has_cjk(repo_name)
    font_path = None

    if need_cjk:
        font_path = _find_system_cjk_font()
        # If CJK is needed but no CJK font is available on this system, we
        # must downgrade to Helvetica. Helvetica cannot render CJK or ZWSPs,
        # so sanitize the text (strips non-latin-1 chars with '?' fallback).
        if not font_path:
            logger.warning("CJK text detected but no CJK font available; downgrading to Helvetica and sanitizing")
            need_cjk = False

    if not need_cjk:
        summary_text = _sanitize_text(summary_text)
        repo_name = _sanitize_text(repo_name)

    # Insert break hints for CJK text only; for non-CJK _wrap_to_width
    # already handles character-level wrapping and ZWSP breaks latin-1 fonts.
    if need_cjk:
        summary_text = _soft_wrap_long_tokens(summary_text)

    # --- Try font sizes from 12 down to 8 to fit on one page ---
    for font_size in [12, 11.5, 11, 10.5, 10, 9.5, 9, 8.5, 8]:
        pdf_bytes = _render_single_page_attempt(
            summary_text, repo_name, font_size,
            need_cjk, font_path,
        )
        if pdf_bytes is not None:
            return pdf_bytes

    # Fallback at smallest size; final renderer is forced to one page.
    return _render_pdf_final(summary_text, repo_name, 8, need_cjk, font_path)


def _render_single_page_attempt(
    summary_text: str, repo_name: str, font_size: float,
    need_cjk: bool, font_path: Optional[str],
) -> Optional[bytes]:
    """Try to render at given font_size. Return bytes if fits one page, else None."""
    from fpdf import FPDF
    from fpdf.enums import XPos, YPos

    trial = FPDF(orientation="P", unit="mm", format="A4")
    trial.set_margins(8, 6, 8)
    trial.set_auto_page_break(auto=False)
    trial.add_page()

    font_family = "Helvetica"
    if need_cjk and font_path:
        try:
            trial.add_font("CJK", "", font_path, uni=True)
            font_family = "CJK"
        except Exception:
            pass

    # Simulate title + accent line (must match _render_pdf_final layout)
    pw = trial.w - trial.l_margin - trial.r_margin
    trial.set_font(font_family if font_family == "CJK" else "Helvetica", "" if font_family == "CJK" else "B", size=14)
    trial.set_xy(trial.l_margin, 6)
    trial.cell(pw, 8, "Title", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    # Subtitle line
    trial.set_font(font_family if font_family == "CJK" else "Helvetica", "" if font_family == "CJK" else "I", size=7.5)
    trial.cell(pw, 4, "Architecture Overview", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    trial.cell(pw, 0.5, "", new_x=XPos.LMARGIN, new_y=YPos.NEXT)  # accent line
    trial.ln(1.5)

    # Simulate body
    _render_body(trial, summary_text, pw, font_size, font_family, need_cjk, font_path)

    # Reserve space for footer (~10 mm) so body does not collide with it.
    usable_h = 297 - 6 - 12  # = 279 mm
    if trial.get_y() < usable_h:
        # Fits on one page. Render the real pretty version.
        return _render_pdf_final(summary_text, repo_name, font_size, need_cjk, font_path)
    return None


def _render_pdf_final(
    summary_text: str, repo_name: str, font_size: float,
    need_cjk: bool, font_path: Optional[str],
) -> bytes:
    """Render the final polished single-page PDF."""
    from fpdf import FPDF
    from fpdf.enums import XPos, YPos

    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_margins(8, 6, 8)
    # Force single-page output to avoid accidental blank second page.
    pdf.set_auto_page_break(auto=False)
    pdf.add_page()

    font_family = "Helvetica"
    if need_cjk and font_path:
        try:
            pdf.add_font("CJK", "", font_path, uni=True)
            font_family = "CJK"
        except Exception as e:
            logger.warning("Failed to load CJK font: %s", e)

    page_width = pdf.w - pdf.l_margin - pdf.r_margin

    # ── Title bar with gradient-like appearance ──
    if font_family == "CJK":
        pdf.set_font("CJK", size=14)
    else:
        pdf.set_font("Helvetica", "B", size=14)
    pdf.set_fill_color(25, 42, 86)
    pdf.set_text_color(255, 255, 255)
    pdf.set_xy(pdf.l_margin, 6)
    pdf.cell(page_width, 8, f"  {repo_name}", new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True, align="L")

    # Subtitle line
    pdf.set_fill_color(35, 55, 100)
    if font_family == "CJK":
        pdf.set_font("CJK", size=7.5)
    else:
        pdf.set_font("Helvetica", "I", size=7.5)
    pdf.set_text_color(200, 215, 245)
    pdf.cell(page_width, 4, "  Architecture Overview  |  Generated by RepoHelper", new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True, align="L")

    # Thin accent line
    pdf.set_fill_color(65, 135, 245)
    pdf.cell(page_width, 0.5, "", new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True)
    pdf.ln(1.5)

    # ── Body ──
    pdf.set_text_color(30, 30, 30)
    if font_family == "CJK":
        pdf.set_font("CJK", size=font_size)
    else:
        pdf.set_font("Helvetica", size=font_size)

    _render_body(pdf, summary_text, page_width, font_size, font_family, need_cjk, font_path)

    # ── Footer line ──
    pdf.set_y(-6)
    pdf.set_x(pdf.l_margin)
    pdf.set_draw_color(200, 200, 200)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.l_margin + page_width, pdf.get_y())
    pdf.ln(0.5)
    if font_family == "CJK":
        pdf.set_font("CJK", size=6)
    else:
        pdf.set_font("Helvetica", "I", size=6)
    pdf.set_text_color(150, 150, 150)
    pdf.cell(page_width, 2.5, "Generated by RepoHelper  |  AI-powered repository documentation", align="C")

    # ── Enforce single page: remove any accidental trailing pages ──
    try:
        while pdf.pages_count > 1:
            pdf.pages.pop()
        pdf.page = 1
    except Exception:
        pass  # fpdf2 internal structure varies; best-effort

    buf = BytesIO()
    pdf.output(buf)
    return buf.getvalue()


def _is_section_header(line: str) -> bool:
    """Check if a line is a section header using robust pattern matching."""
    s = line.strip().replace("\u200b", "").upper()
    # Normalize ampersand variants so headers like "A＆B", "A & B", "A&amp;B"
    # are recognized consistently.
    s = s.replace("&AMP;", "&").replace("\uFF06", "&")
    s = re.sub(r"\s*&\s*", " & ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        return False

    # Canonical English headers (exact prefix match)
    en_headers = [
        "PROJECT NAME:", "PROJECT OVERVIEW:", "ARCHITECTURE & DESIGN:",
        "ARCHITECTURE AND DESIGN:", "ARCHITECTURE:",
        "TECH STACK:", "TECHNOLOGY STACK:", "TECH STACK & DEPENDENCIES:",
        "TECH STACK AND DEPENDENCIES:",
        "KEY MODULES & COMPONENTS:", "KEY MODULES AND COMPONENTS:",
        "KEY MODULES:", "KEY COMPONENTS:", "MODULES & COMPONENTS:",
        "MODULES AND COMPONENTS:",
        "DATA FLOW & PROCESSING:", "DATA FLOW AND PROCESSING:",
        "DATA FLOW:", "DATA PROCESSING:", "DATA FLOW & PIPELINE STAGES:",
        "API & INTEGRATION POINTS:", "API AND INTEGRATION POINTS:",
        "API & INTEGRATIONS:", "API AND INTEGRATIONS:", "API:",
        "TARGET USERS & USE CASES:", "TARGET USERS AND USE CASES:",
        "TARGET USERS:", "USE CASES:",
        # Extended headers from repo-type-specific adapters
        "SERVICE TOPOLOGY & ARCHITECTURE:", "SERVICE TOPOLOGY:",
        "MESSAGE FLOW & DATA PROCESSING:", "MESSAGE FLOW:",
        "DEPLOYMENT & INFRASTRUCTURE:", "DEPLOYMENT:",
        "KEY SERVICES & COMPONENTS:", "KEY SERVICES:",
        "COMPONENT HIERARCHY & ROUTING:", "COMPONENT HIERARCHY:",
        "DATA SCHEMAS & MODELS:", "DATA SCHEMAS:",
        "COMMANDS & INTERFACE:", "COMMANDS:",
    ]
    for h in en_headers:
        if s.startswith(h):
            return True

    # Chinese headers
    cn_prefixes = [
        "项目名称", "项目概述", "功能概述",
        "架构", "技术栈", "关键模块", "核心模块", "关键组件", "核心组件",
        "数据流", "API", "目标用户", "适用人群", "使用场景",
    ]
    for p in cn_prefixes:
        if s.startswith(p) and (":" in s or "：" in s):
            return True

    # Japanese headers
    ja_prefixes = [
        "プロジェクト名", "機能概要", "アーキテクチャ", "技術スタック",
        "主要モジュール", "主要コンポーネント", "データフロー",
        "API", "対象ユーザー",
    ]
    for p in ja_prefixes:
        if s.startswith(p) and (":" in s or "：" in s):
            return True

    return False


def _get_section_key(line: str) -> str:
    """Map a recognized section header to a canonical key.
    Supports English and Chinese headers.
    """
    s = line.strip().replace("\u200b", "")
    su = s.upper()  # for English matching
    # Keep normalization aligned with _is_section_header.
    su = su.replace("&AMP;", "&").replace("\uFF06", "&")
    su = re.sub(r"\s*&\s*", " & ", su)
    su = re.sub(r"\s+", " ", su).strip()

    # --- English headers ---
    if su.startswith("PROJECT NAME:"):
        return "project_name"
    if su.startswith("PROJECT OVERVIEW:"):
        return "project_overview"
    if su.startswith(("ARCHITECTURE & DESIGN:", "ARCHITECTURE AND DESIGN:", "ARCHITECTURE:")):
        return "architecture_design"
    if su.startswith(("TECH STACK:", "TECHNOLOGY STACK:", "TECH STACK & DEPENDENCIES:", "TECH STACK AND DEPENDENCIES:")):
        return "tech_stack"
    if su.startswith(("KEY MODULES & COMPONENTS:", "KEY MODULES AND COMPONENTS:", "KEY MODULES:", "KEY COMPONENTS:", "MODULES & COMPONENTS:", "MODULES AND COMPONENTS:")):
        return "key_modules_components"
    if su.startswith(("DATA FLOW & PROCESSING:", "DATA FLOW AND PROCESSING:", "DATA FLOW:", "DATA PROCESSING:")):
        return "data_flow_processing"
    if su.startswith(("API & INTEGRATION POINTS:", "API AND INTEGRATION POINTS:", "API & INTEGRATIONS:", "API AND INTEGRATIONS:", "API:")):
        return "api_integration_points"
    if su.startswith(("TARGET USERS & USE CASES:", "TARGET USERS AND USE CASES:", "TARGET USERS:", "USE CASES:")):
        return "target_users_use_cases"

    # --- Extended headers from repo-type-specific adapters ---
    if su.startswith(("SERVICE TOPOLOGY & ARCHITECTURE:", "SERVICE TOPOLOGY:")):
        return "architecture_design"
    if su.startswith(("MESSAGE FLOW & DATA PROCESSING:", "MESSAGE FLOW:")):
        return "data_flow_processing"
    if su.startswith(("DEPLOYMENT & INFRASTRUCTURE:", "DEPLOYMENT:")):
        return "deployment"
    if su.startswith(("KEY SERVICES & COMPONENTS:", "KEY SERVICES:")):
        return "key_modules_components"
    if su.startswith(("COMPONENT HIERARCHY & ROUTING:", "COMPONENT HIERARCHY:")):
        return "component_hierarchy"
    if su.startswith(("DATA FLOW & PIPELINE STAGES:",)):
        return "data_flow_processing"
    if su.startswith(("DATA SCHEMAS & MODELS:", "DATA SCHEMAS:")):
        return "data_schemas"
    if su.startswith(("COMMANDS & INTERFACE:", "COMMANDS:")):
        return "api_integration_points"

    # --- Chinese headers (fallback if _postprocess_summary missed) ---
    if s.startswith("项目名称"):
        return "project_name"
    for kw in ("项目概述", "功能概述", "概述"):
        if s.startswith(kw):
            return "project_overview"
    for kw in ("架构", "系统架构"):
        if s.startswith(kw):
            return "architecture_design"
    if s.startswith("技术栈"):
        return "tech_stack"
    for kw in ("关键模块", "核心模块", "关键组件", "核心组件", "主要模块", "主要组件"):
        if s.startswith(kw):
            return "key_modules_components"
    if s.startswith("数据流"):
        return "data_flow_processing"
    for kw in ("API", "接口"):
        if s.startswith(kw) and ("集成" in s or "接口" in s or "端点" in s or ":" in s or "：" in s):
            return "api_integration_points"
    for kw in ("目标用户", "适用人群", "使用场景"):
        if s.startswith(kw):
            return "target_users_use_cases"

    # --- Japanese headers ---
    if s.startswith("プロジェクト名"):
        return "project_name"
    if s.startswith("機能概要"):
        return "project_overview"
    if s.startswith("アーキテクチャ"):
        return "architecture_design"
    if s.startswith("技術スタック"):
        return "tech_stack"
    for kw in ("主要モジュール", "主要コンポーネント"):
        if s.startswith(kw):
            return "key_modules_components"
    if s.startswith("データフロー"):
        return "data_flow_processing"
    if s.startswith("対象ユーザー"):
        return "target_users_use_cases"

    return ""


def _render_body(pdf, text: str, page_width: float, font_size: float, font_family: str, need_cjk: bool, cjk_font_path: Optional[str]):
    """Render the summary body with bold section headers and compact, dense layout."""
    from fpdf.enums import XPos, YPos

    # Section header background colors — vivid, highly saturated.
    section_header_colors = {
        "project_overview": (220, 235, 255),   # blue tint
        "architecture_design": (215, 245, 225), # green tint
        "tech_stack": (230, 242, 255),          # sky blue tint
        "key_modules_components": (215, 240, 248), # teal tint
        "data_flow_processing": (220, 245, 235),  # mint tint
        "api_integration_points": (225, 230, 255), # lavender tint
        "target_users_use_cases": (225, 245, 235), # sage tint
        # Extended section types from repo-type-specific adapters
        "deployment": (248, 235, 220),          # warm amber
        "component_hierarchy": (230, 230, 255), # periwinkle
        "data_schemas": (235, 248, 225),        # olive
    }
    # Accent bar colors — darker, matching the section theme.
    section_accent_colors = {
        "project_overview": (40, 90, 180),
        "architecture_design": (30, 130, 80),
        "tech_stack": (30, 100, 190),
        "key_modules_components": (20, 120, 150),
        "data_flow_processing": (30, 140, 100),
        "api_integration_points": (60, 60, 180),
        "target_users_use_cases": (40, 130, 90),
        "deployment": (160, 100, 30),
        "component_hierarchy": (70, 70, 180),
        "data_schemas": (80, 140, 40),
    }
    # Section icon prefixes — Latin-1 safe symbols to visually distinguish sections.
    # NOTE: We use only characters within the Latin-1 range (<=0xFF) so that the
    # built-in Helvetica font can render them without error.  The original Unicode
    # glyphs (■ ▶ ◆ …) are outside Latin-1 and crash fpdf2's normalize_text().
    section_icons = {
        "project_overview": ">> ",
        "architecture_design": ">> ",
        "tech_stack": ">> ",
        "key_modules_components": ">> ",
        "data_flow_processing": ">> ",
        "api_integration_points": ">> ",
        "target_users_use_cases": ">> ",
        "deployment": ">> ",
        "component_hierarchy": ">> ",
        "data_schemas": ">> ",
    }
    default_header_bg = (230, 230, 248)
    default_accent_color = (60, 60, 140)
    header_text_color = (10, 30, 90)
    body_text_color = (35, 35, 35)
    bullet_dot_color = (50, 100, 200)

    lh = font_size * 0.42  # compact line height in mm
    current_section = ""

    lines = text.split("\n")
    project_name_skipped = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            pdf.ln(lh * 0.18)
            continue

        is_header = _is_section_header(stripped)

        # Skip PROJECT NAME: line (already shown in title bar)
        if is_header and not project_name_skipped:
            if stripped.upper().startswith("PROJECT NAME"):
                project_name_skipped = True
                continue

        if is_header:
            current_section = _get_section_key(stripped)
            # ── Prominent section header with colored band + bold accent bar ──
            pdf.ln(lh * 0.55)

            bg = section_header_colors.get(current_section, default_header_bg)
            accent_color = section_accent_colors.get(current_section, default_accent_color)

            pdf.set_fill_color(*bg)
            pdf.set_text_color(*header_text_color)

            header_font_size = font_size + 1.8
            if font_family == "CJK":
                pdf.set_font("CJK", size=header_font_size)
            else:
                pdf.set_font("Helvetica", "B", size=header_font_size)

            # Prepare header text with icon prefix
            icon = section_icons.get(current_section, ">> ") if not need_cjk else ""
            header_display = f"  {icon}{stripped}"

            header_h = lh + 2.8
            accent_w = 1.8

            pdf.set_x(pdf.l_margin)
            y_before = pdf.get_y()

            wrapped = _wrap_to_width(pdf, header_display, page_width - accent_w)
            for wline in wrapped:
                pdf.set_x(pdf.l_margin)
                pdf.cell(page_width, header_h, wline, new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True, align="L")

            y_after = pdf.get_y()

            # Draw bold left accent bar
            pdf.set_fill_color(*accent_color)
            pdf.rect(pdf.l_margin, y_before, accent_w, y_after - y_before, style="F")

            # Reset to body style
            pdf.set_text_color(*body_text_color)
            if font_family == "CJK":
                pdf.set_font("CJK", size=font_size)
            else:
                pdf.set_font("Helvetica", size=font_size)
            pdf.ln(lh * 0.15)

        elif stripped.startswith("- ") or stripped.startswith("* "):
            # ── Bullet point with colored dot ──
            content = stripped[2:].strip()
            pdf.set_text_color(*body_text_color)

            indent = 4.5
            bullet_x = pdf.l_margin + indent - 2.0

            wrapped = _wrap_to_width(pdf, content, page_width - indent)

            if current_section == "target_users_use_cases":
                for i, wline in enumerate(wrapped):
                    pdf.set_x(pdf.l_margin)
                    if i == 0:
                        pdf.cell(page_width, lh, f"- {wline}", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="L")
                    else:
                        pdf.cell(page_width, lh, f"  {wline}", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="L")
                continue

            for i, wline in enumerate(wrapped):
                pdf.set_x(pdf.l_margin + indent)
                if i == 0:
                    bullet_y = pdf.get_y() + lh * 0.42
                    pdf.set_fill_color(*bullet_dot_color)
                    pdf.circle(bullet_x, bullet_y, 0.5, style="F")
                pdf.cell(page_width - indent, lh, wline, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="L")

        else:
            # ── Regular paragraph text ──
            if current_section == "target_users_use_cases":
                pdf.set_text_color(45, 45, 45)
            else:
                pdf.set_text_color(*body_text_color)
            wrapped = _wrap_to_width(pdf, stripped, page_width)
            for wline in wrapped:
                pdf.set_x(pdf.l_margin)
                pdf.cell(page_width, lh, wline, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="L")


# ---------------------------------------------------------------------------
# Backward-compatible aliases (keep phase3_render_pdf for any internal refs)
# ---------------------------------------------------------------------------

phase3_render_pdf = render_pdf


# ---------------------------------------------------------------------------
# Phase 2b-PDF: Structured AnalyzedContent → format-specific plain text
# ---------------------------------------------------------------------------

def _adapt_pdf_text_generic(analyzed) -> str:
    """Generic 7-section layout (backward compatible)."""
    return analyzed.summary_text


def _adapt_pdf_text_library(analyzed) -> str:
    """Library / SDK — emphasise API & module docs, trim target users."""
    lines: list[str] = []
    lines.append(f"Project Name: {analyzed.repo_name}")
    lines.append("")

    if analyzed.project_overview:
        lines.append("Project Overview:")
        lines.append(analyzed.project_overview)
        lines.append("")

    # API points — promoted to top
    if analyzed.api_points:
        lines.append("API & Integration Points:")
        for pt in analyzed.api_points:
            lines.append(f"- {pt}")
        lines.append("")

    if analyzed.key_modules:
        lines.append("Key Modules & Components:")
        for mod in analyzed.key_modules:
            lines.append(f"- {mod.name}: {mod.responsibility}")
        lines.append("")

    if analyzed.architecture:
        lines.append("Architecture & Design:")
        for item in analyzed.architecture:
            lines.append(f"- {item}")
        lines.append("")

    ts = analyzed.tech_stack
    if ts and (ts.languages or ts.frameworks or ts.key_libraries or ts.infrastructure):
        lines.append("Tech Stack & Dependencies:")
        for x in ts.languages + ts.frameworks + ts.key_libraries + ts.infrastructure:
            lines.append(f"- {x}")
        lines.append("")

    if analyzed.data_flow:
        lines.append("Data Flow & Processing:")
        for step in analyzed.data_flow:
            lines.append(f"- {step}")
        lines.append("")

    return "\n".join(lines).strip()


def _adapt_pdf_text_webapp(analyzed) -> str:
    """Web application — emphasise component hierarchy, routes, state."""
    lines: list[str] = []
    lines.append(f"Project Name: {analyzed.repo_name}")
    lines.append("")

    if analyzed.project_overview:
        lines.append("Project Overview:")
        lines.append(analyzed.project_overview)
        lines.append("")

    if analyzed.component_hierarchy:
        lines.append("Component Hierarchy & Routing:")
        lines.append(analyzed.component_hierarchy)
        lines.append("")

    if analyzed.architecture:
        lines.append("Architecture & Design:")
        for item in analyzed.architecture:
            lines.append(f"- {item}")
        lines.append("")

    ts = analyzed.tech_stack
    if ts and (ts.languages or ts.frameworks or ts.key_libraries or ts.infrastructure):
        lines.append("Tech Stack & Dependencies:")
        for x in ts.languages + ts.frameworks + ts.key_libraries + ts.infrastructure:
            lines.append(f"- {x}")
        lines.append("")

    if analyzed.key_modules:
        lines.append("Key Modules & Components:")
        for mod in analyzed.key_modules:
            lines.append(f"- {mod.name}: {mod.responsibility}")
        lines.append("")

    if analyzed.data_flow:
        lines.append("Data Flow & Processing:")
        for step in analyzed.data_flow:
            lines.append(f"- {step}")
        lines.append("")

    if analyzed.api_points:
        lines.append("API & Integration Points:")
        for pt in analyzed.api_points:
            lines.append(f"- {pt}")
        lines.append("")

    return "\n".join(lines).strip()


def _adapt_pdf_text_microservice(analyzed) -> str:
    """Microservice system — emphasise service topology, message flow, deployment."""
    lines: list[str] = []
    lines.append(f"Project Name: {analyzed.repo_name}")
    lines.append("")

    if analyzed.project_overview:
        lines.append("Project Overview:")
        lines.append(analyzed.project_overview)
        lines.append("")

    if analyzed.architecture:
        lines.append("Service Topology & Architecture:")
        for item in analyzed.architecture:
            lines.append(f"- {item}")
        lines.append("")

    if analyzed.data_flow:
        lines.append("Message Flow & Data Processing:")
        for step in analyzed.data_flow:
            lines.append(f"- {step}")
        lines.append("")

    if analyzed.deployment_info:
        lines.append("Deployment & Infrastructure:")
        lines.append(analyzed.deployment_info)
        lines.append("")

    ts = analyzed.tech_stack
    if ts and (ts.languages or ts.frameworks or ts.key_libraries or ts.infrastructure):
        lines.append("Tech Stack & Dependencies:")
        for x in ts.languages + ts.frameworks + ts.key_libraries + ts.infrastructure:
            lines.append(f"- {x}")
        lines.append("")

    if analyzed.key_modules:
        lines.append("Key Services & Components:")
        for mod in analyzed.key_modules:
            lines.append(f"- {mod.name}: {mod.responsibility}")
        lines.append("")

    if analyzed.api_points:
        lines.append("API & Integration Points:")
        for pt in analyzed.api_points:
            lines.append(f"- {pt}")
        lines.append("")

    return "\n".join(lines).strip()


def _adapt_pdf_text_data_pipeline(analyzed) -> str:
    """Data / ML pipeline — emphasise data flow, schemas, pipeline stages."""
    lines: list[str] = []
    lines.append(f"Project Name: {analyzed.repo_name}")
    lines.append("")

    if analyzed.project_overview:
        lines.append("Project Overview:")
        lines.append(analyzed.project_overview)
        lines.append("")

    if analyzed.data_flow:
        lines.append("Data Flow & Pipeline Stages:")
        for step in analyzed.data_flow:
            lines.append(f"- {step}")
        lines.append("")

    if analyzed.data_schemas:
        lines.append("Data Schemas & Models:")
        lines.append(analyzed.data_schemas)
        lines.append("")

    if analyzed.architecture:
        lines.append("Architecture & Design:")
        for item in analyzed.architecture:
            lines.append(f"- {item}")
        lines.append("")

    ts = analyzed.tech_stack
    if ts and (ts.languages or ts.frameworks or ts.key_libraries or ts.infrastructure):
        lines.append("Tech Stack & Dependencies:")
        for x in ts.languages + ts.frameworks + ts.key_libraries + ts.infrastructure:
            lines.append(f"- {x}")
        lines.append("")

    if analyzed.key_modules:
        lines.append("Key Modules & Components:")
        for mod in analyzed.key_modules:
            lines.append(f"- {mod.name}: {mod.responsibility}")
        lines.append("")

    if analyzed.target_users:
        lines.append("Target Users & Use Cases:")
        lines.append(analyzed.target_users)
        lines.append("")

    return "\n".join(lines).strip()


def _adapt_pdf_text_cli_tool(analyzed) -> str:
    """CLI tool — emphasise commands, flags, module structure."""
    lines: list[str] = []
    lines.append(f"Project Name: {analyzed.repo_name}")
    lines.append("")

    if analyzed.project_overview:
        lines.append("Project Overview:")
        lines.append(analyzed.project_overview)
        lines.append("")

    if analyzed.api_points:
        lines.append("Commands & Interface:")
        for pt in analyzed.api_points:
            lines.append(f"- {pt}")
        lines.append("")

    if analyzed.key_modules:
        lines.append("Key Modules & Components:")
        for mod in analyzed.key_modules:
            lines.append(f"- {mod.name}: {mod.responsibility}")
        lines.append("")

    if analyzed.architecture:
        lines.append("Architecture & Design:")
        for item in analyzed.architecture:
            lines.append(f"- {item}")
        lines.append("")

    ts = analyzed.tech_stack
    if ts and (ts.languages or ts.frameworks or ts.key_libraries or ts.infrastructure):
        lines.append("Tech Stack & Dependencies:")
        for x in ts.languages + ts.frameworks + ts.key_libraries + ts.infrastructure:
            lines.append(f"- {x}")
        lines.append("")

    if analyzed.data_flow:
        lines.append("Data Flow & Processing:")
        for step in analyzed.data_flow:
            lines.append(f"- {step}")
        lines.append("")

    return "\n".join(lines).strip()


# Adapter dispatch table
_PDF_ADAPTERS = {
    "generic": _adapt_pdf_text_generic,
    "library": _adapt_pdf_text_library,
    "sdk": _adapt_pdf_text_library,
    "webapp": _adapt_pdf_text_webapp,
    "microservice": _adapt_pdf_text_microservice,
    "data_pipeline": _adapt_pdf_text_data_pipeline,
    "cli_tool": _adapt_pdf_text_cli_tool,
}


def _format_evolution_section(analyzed) -> str:
    """Format commit history + evolution narrative as a PDF text section.

    Returns empty string when no commit data is available. Always reads
    directly from ``analyzed.commit_timeline`` so this works even when
    the LLM failed to produce valid JSON and structured fields are empty.
    """
    timeline = getattr(analyzed, "commit_timeline", None)
    if not timeline or timeline.is_empty():
        return ""

    lines: list[str] = []
    lines.append("Project Evolution & Commit History:")

    narrative = getattr(analyzed, "evolution_narrative", "") or ""
    if narrative.strip():
        lines.append(narrative.strip())
        lines.append("")

    # Range summary
    if timeline.first_commit_date or timeline.latest_commit_date:
        first = timeline.first_commit_date[:10] if timeline.first_commit_date else "?"
        latest = timeline.latest_commit_date[:10] if timeline.latest_commit_date else "?"
        lines.append(f"- Timeline range: {first} to {latest} ({timeline.total_commits_scanned} commits analyzed)")

    # Contributors
    if timeline.contributors:
        top = timeline.contributors[:5]
        names = ", ".join(f"{c.login} ({c.commit_count})" for c in top)
        lines.append(f"- Top contributors: {names}")

    # Releases
    if timeline.releases:
        rels = [f"{r.tag} ({r.date[:10]})" for r in timeline.releases[:5] if r.tag]
        if rels:
            lines.append(f"- Releases: {', '.join(rels)}")

    # Recent commits
    if timeline.commits:
        lines.append("")
        lines.append("Recent commits:")
        for c in timeline.commits[:8]:
            date = c.date[:10] if c.date else ""
            msg = c.message[:80] if c.message else ""
            lines.append(f"- {date} [{c.sha}] {c.author}: {msg}")

    return "\n".join(lines).strip()


def render_pdf_from_analyzed(analyzed) -> bytes:
    """
    Phase 2b-PDF + Phase 3: AnalyzedContent → PDF bytes.

    Selects a layout adapter based on ``analyzed.repo_type_hint``,
    assembles a plain-text summary in the appropriate order / emphasis,
    then delegates to the existing ``render_pdf()`` renderer.
    """
    adapter = _PDF_ADAPTERS.get(analyzed.repo_type_hint, _adapt_pdf_text_generic)
    summary_text = adapter(analyzed)

    # Safety: if the adapted text is essentially empty (just "Project Name: ..."),
    # fall back to the generic adapter which uses summary_text (including raw_llm_text fallback)
    stripped_lines = [l.strip() for l in summary_text.split("\n") if l.strip()]
    if len(stripped_lines) <= 1:
        logger.warning(
            "PDF adapter '%s' produced near-empty text (%d lines); falling back to generic",
            analyzed.repo_type_hint, len(stripped_lines),
        )
        summary_text = _adapt_pdf_text_generic(analyzed)

    # Append commit-history section directly from structured data.
    # Works regardless of whether the LLM produced valid JSON.
    evolution_block = _format_evolution_section(analyzed)
    if evolution_block:
        summary_text = (summary_text.rstrip() + "\n\n" + evolution_block).strip()

    return render_pdf(summary_text, analyzed.repo_name)


# ---------------------------------------------------------------------------
# Debug helper: print AnalyzedContent (LLM structured output)
# ---------------------------------------------------------------------------

def _print_analyzed_content(analyzed, label: str = "PDF"):
    """Pretty-print the AnalyzedContent structured fields to console & log."""
    import json as _json

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
        "evolution_narrative": analyzed.evolution_narrative,
        "commit_timeline": (
            {
                "total_commits_scanned": analyzed.commit_timeline.total_commits_scanned,
                "first_commit_date": analyzed.commit_timeline.first_commit_date,
                "latest_commit_date": analyzed.commit_timeline.latest_commit_date,
                "commits_preview": [
                    {"date": c.date[:10], "sha": c.sha, "author": c.author, "message": c.message}
                    for c in analyzed.commit_timeline.commits[:10]
                ],
                "contributors": [
                    {"login": c.login, "commits": c.commit_count}
                    for c in analyzed.commit_timeline.contributors[:5]
                ],
                "releases": [
                    {"tag": r.tag, "date": r.date[:10] if r.date else "", "name": r.name}
                    for r in analyzed.commit_timeline.releases[:5]
                ],
            }
            if analyzed.commit_timeline else None
        ),
    }

    json_str = _json.dumps(data, ensure_ascii=False, indent=2)

    header = f"\n{'='*60}\nAnalyzedContent (input to {label} renderer)\n{'='*60}"
    print(header)
    print(json_str)
    print('='*60 + '\n')

    logger.info("AnalyzedContent for %s export:\n%s", label, json_str)


# ---------------------------------------------------------------------------
# Backward-compatible orchestrators
# These delegate to the decoupled content_analyzer + render_pdf pipeline.
# Existing API routes (api.py) import these, so they must stay.
# ---------------------------------------------------------------------------

async def generate_direct_pdf(request: DirectPDFExportRequest) -> bytes:
    """
    Backward-compatible wrapper: repo embeddings → PDF bytes.
    Delegates to content_analyzer for Phase 1+2, then render_pdf for Phase 3.
    """
    from api.content_analyzer import RepoAnalysisRequest, analyze_repo_content

    analysis_request = RepoAnalysisRequest(
        repo_url=request.repo_url,
        repo_name=request.repo_name,
        provider=request.provider,
        model=request.model,
        language=request.language,
        repo_type=request.repo_type,
        access_token=request.access_token,
        excluded_dirs=request.excluded_dirs,
        excluded_files=request.excluded_files,
        included_dirs=request.included_dirs,
        included_files=request.included_files,
    )

    analyzed = await analyze_repo_content(analysis_request)
    _print_analyzed_content(analyzed, "PDF (direct)")

    pdf_bytes = render_pdf_from_analyzed(analyzed)
    logger.info("Direct PDF export complete — %d bytes", len(pdf_bytes))
    return pdf_bytes
