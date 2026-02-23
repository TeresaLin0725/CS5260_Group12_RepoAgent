"""
PDF Technical Report Export Module

Three-phase pipeline:
  Phase 1: Content Extraction          (from cached wiki pages — NO LLM call)
  Phase 2: Technical Report Generation (single LLM call — detailed report)
  Phase 3: PDF Rendering               (fpdf2 → multi-page PDF bytes)
"""

import json
import logging
import os
import re
from io import BytesIO
from typing import List, Dict, Optional, Any

import google.generativeai as genai
from adalflow.components.model_client.ollama_client import OllamaClient
from adalflow.core.types import ModelType
from pydantic import BaseModel, Field

from api.config import (
    get_model_config,
    configs,
    OPENROUTER_API_KEY,
    OPENAI_API_KEY,
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY,
)
from api.openai_client import OpenAIClient
from api.openrouter_client import OpenRouterClient
from api.bedrock_client import BedrockClient
from api.azureai_client import AzureAIClient
from api.dashscope_client import DashscopeClient
from api.prompts import PDF_ONEPAGE_SUMMARY_PROMPT

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class WikiPageForPDF(BaseModel):
    id: str
    title: str
    content: str
    importance: str = "medium"


class PDFExportRequest(BaseModel):
    repo_url: str = Field(..., description="Repository URL")
    repo_name: str = Field("", description="Short repo name (owner/repo)")
    pages: List[WikiPageForPDF] = Field(..., description="Wiki pages to summarise")
    provider: str = Field("ollama", description="LLM provider")
    model: Optional[str] = Field(None, description="LLM model name")
    language: str = Field("en", description="Target language code")


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
# Phase 1 — Code Semantics Extraction (pure Python, no LLM)
# ---------------------------------------------------------------------------

def _extract_page_content(text: str, max_chars: int = 1500) -> str:
    """Extract meaningful content from a wiki page, preserving technical details.
    Strips markdown formatting but keeps code references, architecture details, etc.
    """
    lines = text.splitlines()
    result_lines = []
    in_code_block = False
    total_chars = 0

    for line in lines:
        stripped = line.strip()

        # Track code blocks — skip long code blocks but note them
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            continue
        if in_code_block:
            continue

        # Skip empty lines but preserve paragraph separation
        if not stripped:
            if result_lines and result_lines[-1] != "":
                result_lines.append("")
            continue

        # Skip pure table rows
        if stripped.startswith("|") and stripped.endswith("|"):
            continue
        if re.match(r'^[\|\-\s:]+$', stripped):
            continue

        # Strip markdown heading markers but keep heading text
        clean = re.sub(r'^#{1,6}\s+', '', stripped)
        # Strip markdown bold but keep text
        clean = re.sub(r'\*\*(.+?)\*\*', r'\1', clean)
        # Strip markdown links but keep text
        clean = re.sub(r'\[(.+?)\]\(.+?\)', r'\1', clean)
        # Keep inline code references — just remove backticks but keep content
        clean = re.sub(r'`([^`]+)`', r'\1', clean)
        clean = clean.strip()

        if not clean:
            continue

        result_lines.append(clean)
        total_chars += len(clean)

        if total_chars >= max_chars:
            break

    return "\n".join(result_lines).strip()[:max_chars]


def _summarize_page(page: WikiPageForPDF, max_chars: int = 1500) -> str:
    """Create a content-rich summary of a page for the LLM prompt."""
    content = _extract_page_content(page.content, max_chars)
    if content:
        return f"[{page.title}]\n{content}"
    return f"[{page.title}]\n(No content available)"


def phase1_extract(pages: List[WikiPageForPDF], repo_url: str, repo_name: str) -> str:
    """
    Extract comprehensive content from existing wiki pages for the LLM summary.
    Returns a rich text block that preserves technical details from the wiki.
    This reuses cached content — zero LLM cost.
    """
    high = [p for p in pages if p.importance == "high"]
    medium = [p for p in pages if p.importance == "medium"]
    low = [p for p in pages if p.importance == "low"]

    lines = []
    lines.append(f"Repository: {repo_name}")
    lines.append(f"URL: {repo_url}")
    lines.append(f"Total documentation pages: {len(pages)}")
    lines.append("")

    # Include ALL high-importance pages with rich content
    if high:
        lines.append("=" * 60)
        lines.append("HIGH-IMPORTANCE DOCUMENTATION:")
        lines.append("=" * 60)
        for p in high:
            lines.append("")
            lines.append(_summarize_page(p, max_chars=2000))
            lines.append("")

    # Include medium-importance pages with moderate content
    if medium:
        lines.append("=" * 60)
        lines.append("MEDIUM-IMPORTANCE DOCUMENTATION:")
        lines.append("=" * 60)
        for p in medium[:10]:  # up to 10 medium pages
            lines.append("")
            lines.append(_summarize_page(p, max_chars=1200))
            lines.append("")

    # Include low-importance pages with brief content
    if low:
        lines.append("=" * 60)
        lines.append("ADDITIONAL DOCUMENTATION:")
        lines.append("=" * 60)
        for p in low[:8]:  # up to 8 low pages
            lines.append("")
            lines.append(_summarize_page(p, max_chars=600))
            lines.append("")

    result = "\n".join(lines)

    # Cap total size to avoid exceeding context window, but keep it generous
    max_total = 30000
    if len(result) > max_total:
        result = result[:max_total] + "\n\n[... additional documentation truncated for brevity ...]"

    logger.info(
        "Phase 1 complete — %d high, %d medium, %d low pages, %d total chars",
        len(high),
        len(medium),
        len(low),
        len(result),
    )
    return result


# ---------------------------------------------------------------------------
# Phase 2 — Analogy Mapping (single LLM call)
# ---------------------------------------------------------------------------

def _extract_ollama_content(response) -> str:
    """
    Robustly extract the text content from an Ollama response object.
    Handles both streaming and non-streaming, and strips think tags.
    """
    text = ""

    # Try message.content first (non-streaming response)
    if hasattr(response, "message"):
        msg = response.message
        if hasattr(msg, "content") and msg.content:
            text = msg.content
        elif isinstance(msg, dict) and "content" in msg:
            text = msg["content"]

    # Fallback: response attribute
    if not text and hasattr(response, "response") and response.response:
        text = response.response

    # Fallback: text attribute
    if not text and hasattr(response, "text") and response.text:
        text = response.text

    # Last resort: str(response), but try to extract content from it
    if not text:
        raw = str(response)
        # Try to extract content='...' from the string representation
        content_match = re.search(r"content='(.*?)'(?:,\s*(?:images|tool))", raw, re.DOTALL)
        if content_match:
            text = content_match.group(1)
        else:
            # Try another pattern
            content_match = re.search(r"content='(.*)'", raw, re.DOTALL)
            if content_match:
                text = content_match.group(1)
            else:
                text = raw

    # Strip think tags (qwen3 thinking output)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = text.replace('<think>', '').replace('</think>', '')

    return text.strip()


def _postprocess_summary(text: str) -> str:
    """
    Clean up and repair LLM output to ensure the 6-section structure.
    Small models often produce:
    - Broken section headers (missing colon, extra spaces)
    - Mixed-up sections
    - Extra markdown formatting
    """
    # Remove markdown bold markers
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    # Remove markdown heading markers
    text = re.sub(r'^#{1,4}\s+', '', text, flags=re.MULTILINE)
    # Remove stray markdown fences
    text = text.replace('```', '')

    # Normalize section headers — fix common LLM variants
    header_patterns = {
        r'(?i)^[\s]*PROJECT\s+NAME\s*[:：]': 'PROJECT NAME:',
        r'(?i)^[\s]*PROJECT\s+OVERVIEW\s*[:：]': 'PROJECT OVERVIEW:',
        r'(?i)^[\s]*ARCHITECTURE\s*[&\s]+DESIGN\s*[:：]': 'ARCHITECTURE & DESIGN:',
        r'(?i)^[\s]*ARCHITECTURE\s*[:：]': 'ARCHITECTURE & DESIGN:',
        r'(?i)^[\s]*TECH(?:NOLOGY)?\s+STACK\s*(?:[&\s]+DEPENDENCIES)?\s*[:：]': 'TECH STACK:',
        r'(?i)^[\s]*KEY\s+MODULES\s*(?:[&\s]+COMPONENTS)?\s*[:：]': 'KEY MODULES & COMPONENTS:',
        r'(?i)^[\s]*KEY\s+COMPONENTS\s*[:：]': 'KEY MODULES & COMPONENTS:',
        r'(?i)^[\s]*MODULES?\s*(?:[&\s]+COMPONENTS)?\s*[:：]': 'KEY MODULES & COMPONENTS:',
        r'(?i)^[\s]*DATA\s+FLOW\s*(?:[&\s]+PROCESSING)?\s*[:：]': 'DATA FLOW & PROCESSING:',
        r'(?i)^[\s]*API\s*(?:[&\s]+INTEGRATIONS?\s*(?:POINTS?)?)?\s*[:：]': 'API & INTEGRATION POINTS:',
        r'(?i)^[\s]*TARGET\s+USERS?\s*(?:[&\s]+USE\s+CASES?)?\s*[:：]': 'TARGET USERS & USE CASES:',
        r'(?i)^[\s]*USE\s+CASES?\s*(?:[&\s]+TARGET\s+USERS?)?\s*[:：]': 'TARGET USERS & USE CASES:',
        r'(?i)^[\s]*WHO\s+WOULD\s+USE\s+THIS\s*[:：]': 'TARGET USERS & USE CASES:',
        r'(?i)^[\s]*WHAT\s+IT\s+DOES\s*[:：]': 'PROJECT OVERVIEW:',
        r'(?i)^[\s]*HOW\s+IT\s+WORKS\s*(?:\(SIMPLIFIED\))?\s*[:：]': 'ARCHITECTURE & DESIGN:',
        # Chinese variants (broad patterns to catch many LLM outputs)
        r'(?i)^[\s]*项目名称\s*[:：]': 'PROJECT NAME:',
        r'(?i)^[\s]*项目概述\s*[:：]': 'PROJECT OVERVIEW:',
        r'(?i)^[\s]*功能概述\s*[:：]': 'PROJECT OVERVIEW:',
        r'(?i)^[\s]*概述\s*[:：]': 'PROJECT OVERVIEW:',
        r'(?i)^[\s]*架构[与和&]?\s*设计\s*[:：]': 'ARCHITECTURE & DESIGN:',
        r'(?i)^[\s]*系统架构\s*[:：]': 'ARCHITECTURE & DESIGN:',
        r'(?i)^[\s]*工作原理[（(]?简化版[）)]?\s*[:：]': 'ARCHITECTURE & DESIGN:',
        r'(?i)^[\s]*技术栈[与和&]?\s*(?:依赖)?\s*[:：]': 'TECH STACK:',
        r'(?i)^[\s]*(?:关键|核心|主要)?\s*(?:模块|组件)[与和&]?\s*(?:模块|组件)?\s*[:：]': 'KEY MODULES & COMPONENTS:',
        r'(?i)^[\s]*数据流[与和&]?\s*处理\s*[:：]': 'DATA FLOW & PROCESSING:',
        r'(?i)^[\s]*数据流\s*[:：]': 'DATA FLOW & PROCESSING:',
        r'(?i)^[\s]*API\s*[与和&]?\s*集成点?\s*[:：]': 'API & INTEGRATION POINTS:',
        r'(?i)^[\s]*API\s*[与和&]?\s*(?:接口|端点)\s*[:：]': 'API & INTEGRATION POINTS:',
        r'(?i)^[\s]*接口[与和&]?\s*(?:集成)?\s*[:：]': 'API & INTEGRATION POINTS:',
        r'(?i)^[\s]*(?:适用人群|目标用户|使用场景)\s*[与和&]?\s*(?:使用场景|用例|目标用户)?\s*[:：]': 'TARGET USERS & USE CASES:',
    }

    for pattern, replacement in header_patterns.items():
        text = re.sub(pattern, replacement, text, flags=re.MULTILINE)

    # Fix section headers that appear mid-line (small models often do this)
    # e.g. "- Step 2 blah KEY COMPONENTS:" → "- Step 2 blah\nKEY COMPONENTS:"
    inline_headers = [
        'PROJECT NAME:', 'PROJECT OVERVIEW:', 'ARCHITECTURE & DESIGN:',
        'TECH STACK:', 'KEY MODULES & COMPONENTS:',
        'DATA FLOW & PROCESSING:', 'API & INTEGRATION POINTS:',
        'TARGET USERS & USE CASES:',
    ]
    for h in inline_headers:
        # If the header appears in mid-line (not at line start), split it
        pattern = re.compile(r'([^\n]+\S)\s+(' + re.escape(h) + r')', re.IGNORECASE)
        text = pattern.sub(r'\1\n\2', text)

    # Remove excessive blank lines
    text = re.sub(r'\n{4,}', '\n\n\n', text)

    # Remove the example output if the model accidentally repeated it
    example_start = text.find('--- EXAMPLE OUTPUT')
    if example_start > 0:
        text = text[:example_start].strip()
    example_start2 = text.find('--- END EXAMPLE')
    if example_start2 > 0:
        text = text[:example_start2].strip()

    # Remove lines that are just the separator
    text = re.sub(r'^---+\s*$', '', text, flags=re.MULTILINE)

    # Clean escaped newlines that some models produce
    text = text.replace('\\n', '\n')

    # Ensure bullet points are normalized
    text = re.sub(r'^[\s]*[•*]\s+', '- ', text, flags=re.MULTILINE)

    # Strip SUMMARY section if present (we no longer include it)
    text = re.sub(r'\n*(?:SUMMARY|总结|一句话总结|まとめ)\s*[:：].*', '', text, flags=re.DOTALL | re.IGNORECASE)

    return text.strip()


async def phase2_analogy(
    semantics: str,
    provider: str,
    model: Optional[str],
    language: str,
) -> str:
    """
    One LLM call: turn the Phase-1 JSON into a human-friendly summary.
    Returns plain text ready for Phase 3.
    """

    # Resolve language display name
    supported_langs = configs.get("lang_config", {}).get("supported_languages", {})
    language_name = supported_langs.get(language, "English")

    # semantics is now a plain text string from Phase 1
    repo_name = semantics.split("\n")[0].replace("Repository: ", "").strip() if semantics else "unknown"

    prompt = PDF_ONEPAGE_SUMMARY_PROMPT.format(
        repo_name=repo_name,
        language_name=language_name,
        input_json=semantics,
    )

    full_text = ""

    model_config = get_model_config(provider, model)["model_kwargs"]

    if provider == "ollama":
        # /no_think must be at the very beginning for qwen3
        prompt = "/no_think\n" + prompt

        client = OllamaClient()
        model_kwargs = {
            "model": model_config["model"],
            "stream": False,
            "options": {
                "temperature": 0.3,  # very low for structured output from small models
                "top_p": 0.7,
                "num_ctx": min(model_config.get("num_ctx", 8000), 8000),  # cap ctx for memory
            },
        }

        api_kwargs = client.convert_inputs_to_api_kwargs(
            input=prompt,
            model_kwargs=model_kwargs,
            model_type=ModelType.LLM,
        )

        response = await client.acall(api_kwargs=api_kwargs, model_type=ModelType.LLM)

        # Extract content from Ollama response properly
        full_text = _extract_ollama_content(response)

    elif provider == "openai":
        client = OpenAIClient()
        model_kwargs = {
            "model": model_config["model"],
            "stream": False,
            "temperature": 0.5,
        }
        if "top_p" in model_config:
            model_kwargs["top_p"] = model_config["top_p"]

        api_kwargs = client.convert_inputs_to_api_kwargs(
            input=prompt,
            model_kwargs=model_kwargs,
            model_type=ModelType.LLM,
        )
        response = await client.acall(api_kwargs=api_kwargs, model_type=ModelType.LLM)
        # Handle OpenAI non-streaming response
        if hasattr(response, "choices") and response.choices:
            full_text = response.choices[0].message.content or ""
        else:
            full_text = str(response)

    elif provider == "openrouter":
        client = OpenRouterClient()
        model_kwargs = {
            "model": model_config["model"],
            "stream": False,
            "temperature": 0.5,
        }
        if "top_p" in model_config:
            model_kwargs["top_p"] = model_config["top_p"]

        api_kwargs = client.convert_inputs_to_api_kwargs(
            input=prompt,
            model_kwargs=model_kwargs,
            model_type=ModelType.LLM,
        )
        response = await client.acall(api_kwargs=api_kwargs, model_type=ModelType.LLM)
        if hasattr(response, "choices") and response.choices:
            full_text = response.choices[0].message.content or ""
        else:
            full_text = str(response)

    elif provider == "bedrock":
        client = BedrockClient()
        model_kwargs = {
            "model": model_config["model"],
            "temperature": 0.5,
            "top_p": model_config.get("top_p", 0.8),
        }
        api_kwargs = client.convert_inputs_to_api_kwargs(
            input=prompt,
            model_kwargs=model_kwargs,
            model_type=ModelType.LLM,
        )
        response = await client.acall(api_kwargs=api_kwargs, model_type=ModelType.LLM)
        full_text = str(response)

    elif provider == "azure":
        client = AzureAIClient()
        model_kwargs = {
            "model": model_config["model"],
            "stream": False,
            "temperature": 0.5,
            "top_p": model_config.get("top_p", 0.8),
        }
        api_kwargs = client.convert_inputs_to_api_kwargs(
            input=prompt,
            model_kwargs=model_kwargs,
            model_type=ModelType.LLM,
        )
        response = await client.acall(api_kwargs=api_kwargs, model_type=ModelType.LLM)
        full_text = str(response)

    elif provider == "dashscope":
        client = DashscopeClient()
        model_kwargs = {
            "model": model_config["model"],
            "stream": False,
            "temperature": 0.5,
            "top_p": model_config.get("top_p", 0.8),
        }
        api_kwargs = client.convert_inputs_to_api_kwargs(
            input=prompt,
            model_kwargs=model_kwargs,
            model_type=ModelType.LLM,
        )
        response = await client.acall(api_kwargs=api_kwargs, model_type=ModelType.LLM)
        full_text = str(response)

    else:
        # Default: Google Generative AI
        gen_model = genai.GenerativeModel(
            model_name=model_config["model"],
            generation_config={
                "temperature": 0.5,
                "top_p": model_config.get("top_p", 0.8),
                "top_k": model_config.get("top_k", 20),
            },
        )
        response = gen_model.generate_content(prompt)
        full_text = response.text if hasattr(response, "text") else str(response)

    logger.info("Phase 2 raw LLM output — %d chars", len(full_text))

    # Post-process: clean up and repair the structure
    full_text = _postprocess_summary(full_text)
    logger.info("Phase 2 post-processed — %d chars", len(full_text))

    return full_text.strip()


# ---------------------------------------------------------------------------
# Phase 3 — PDF Rendering (fpdf2)
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
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
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

def phase3_render_pdf(summary_text: str, repo_name: str) -> bytes:
    """
    Render plain-text summary into a polished single-page A4 PDF.
    Uses auto font-sizing to fit content on one page.
    Returns raw PDF bytes.
    """
    from fpdf import FPDF
    from fpdf.enums import XPos, YPos

    # --- Font setup ---
    need_cjk = _has_cjk(summary_text) or _has_cjk(repo_name)
    font_path = None

    if need_cjk:
        font_path = _find_system_cjk_font()

    if not need_cjk:
        summary_text = _sanitize_text(summary_text)
        repo_name = _sanitize_text(repo_name)

    # Insert break hints for CJK text only; for non-CJK _wrap_to_width
    # already handles character-level wrapping and ZWSP breaks latin-1 fonts.
    if need_cjk:
        summary_text = _soft_wrap_long_tokens(summary_text)

    # --- Try font sizes from 10.5 down to 7 to fit on one page ---
    for font_size in [10.5, 10, 9.5, 9, 8.5, 8, 7.5, 7]:
        pdf_bytes = _render_single_page_attempt(
            summary_text, repo_name, font_size,
            need_cjk, font_path,
        )
        if pdf_bytes is not None:
            return pdf_bytes

    # Fallback at smallest size; final renderer is forced to one page.
    return _render_pdf_final(summary_text, repo_name, 7.0, need_cjk, font_path)


def _render_single_page_attempt(
    summary_text: str, repo_name: str, font_size: float,
    need_cjk: bool, font_path: Optional[str],
) -> Optional[bytes]:
    """Try to render at given font_size. Return bytes if fits one page, else None."""
    from fpdf import FPDF
    from fpdf.enums import XPos, YPos

    trial = FPDF(orientation="P", unit="mm", format="A4")
    trial.set_margins(10, 8, 10)
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
    trial.set_font(font_family if font_family == "CJK" else "Helvetica", "" if font_family == "CJK" else "B", size=15)
    trial.set_xy(trial.l_margin, 8)
    trial.cell(pw, 9, "Title", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    trial.cell(pw, 0.6, "", new_x=XPos.LMARGIN, new_y=YPos.NEXT)  # accent line
    trial.ln(2.5)

    # Simulate body
    _render_body(trial, summary_text, pw, font_size, font_family, need_cjk, font_path)

    # Reserve space for footer (~12 mm) so body does not collide with it.
    usable_h = 297 - 8 - 16  # = 273 mm
    if trial.get_y() < usable_h:
        # Fits on one page. Render the real pretty version.
        return _render_pdf_final(summary_text, repo_name, font_size, need_cjk, font_path)
    return None


def _render_pdf_final(
    summary_text: str, repo_name: str, font_size: float,
    need_cjk: bool, font_path: Optional[str],
) -> bytes:
    """Render the final polished PDF."""
    from fpdf import FPDF
    from fpdf.enums import XPos, YPos

    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_margins(10, 8, 10)
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

    # ── Title bar ──
    if font_family == "CJK":
        pdf.set_font("CJK", size=15)
    else:
        pdf.set_font("Helvetica", "B", size=15)
    pdf.set_fill_color(35, 55, 100)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(page_width, 9, f"  {repo_name}", new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True, align="L")

    # Thin accent line
    pdf.set_fill_color(70, 130, 220)
    pdf.cell(page_width, 0.6, "", new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True)
    pdf.ln(2.5)

    # ── Body ──
    pdf.set_text_color(30, 30, 30)
    if font_family == "CJK":
        pdf.set_font("CJK", size=font_size)
    else:
        pdf.set_font("Helvetica", size=font_size)

    _render_body(pdf, summary_text, page_width, font_size, font_family, need_cjk, font_path)

    # ── Footer line ──
    pdf.set_y(-8)
    pdf.set_x(pdf.l_margin)
    pdf.set_draw_color(200, 200, 200)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.l_margin + page_width, pdf.get_y())
    pdf.ln(0.8)
    if font_family == "CJK":
        pdf.set_font("CJK", size=7)
    else:
        pdf.set_font("Helvetica", "I", size=7)
    pdf.set_text_color(150, 150, 150)
    pdf.cell(page_width, 3, "Generated by DeepWiki  |  AI-powered repository documentation", align="C")

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
    s = line.strip().upper()
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
        "DATA FLOW:", "DATA PROCESSING:",
        "API & INTEGRATION POINTS:", "API AND INTEGRATION POINTS:",
        "API & INTEGRATIONS:", "API AND INTEGRATIONS:", "API:",
        "TARGET USERS & USE CASES:", "TARGET USERS AND USE CASES:",
        "TARGET USERS:", "USE CASES:",
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
    Supports English, Chinese, and Japanese headers.
    """
    s = line.strip()
    su = s.upper()  # for English matching

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
    """Render the summary body with polished section headers and compact layout."""
    from fpdf.enums import XPos, YPos

    # Vivid color rendering for ALL section titles (more saturated for visibility).
    section_header_colors = {
        "project_overview": (210, 225, 248),   # blue tint
        "architecture_design": (205, 240, 220), # green tint
        "tech_stack": (215, 230, 250),          # sky blue
        "key_modules_components": (200, 235, 245), # teal tint
        "data_flow_processing": (208, 240, 225),  # mint
        "api_integration_points": (218, 225, 248), # lavender
        "target_users_use_cases": (215, 238, 228), # sage
    }
    default_header_bg = (220, 220, 235)  # light purple fallback
    header_text_color = (20, 45, 100)
    body_text_color = (40, 40, 40)
    bullet_dot_color = (80, 120, 190)

    lh = font_size * 0.48  # line height in mm
    current_section = ""

    lines = text.split("\n")
    project_name_skipped = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            pdf.ln(lh * 0.35)
            continue

        is_header = _is_section_header(stripped)

        # Skip PROJECT NAME: line (already shown in title bar)
        if is_header and not project_name_skipped:
            if stripped.upper().startswith("PROJECT NAME"):
                project_name_skipped = True
                continue

        if is_header:
            current_section = _get_section_key(stripped)
            # ── Section header with colored background band ──
            pdf.ln(lh * 0.5)

            bg = section_header_colors.get(current_section, default_header_bg)

            pdf.set_fill_color(*bg)
            pdf.set_text_color(*header_text_color)

            if font_family == "CJK":
                pdf.set_font("CJK", size=font_size + 1.0)
            else:
                pdf.set_font("Helvetica", "B", size=font_size + 1.0)

            # Left accent bar + header text
            header_h = lh + 2.2
            accent_w = 1.2

            pdf.set_x(pdf.l_margin)
            y_before = pdf.get_y()

            wrapped = _wrap_to_width(pdf, "   " + stripped, page_width)
            for wline in wrapped:
                pdf.set_x(pdf.l_margin)
                pdf.cell(page_width, header_h, wline, new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True, align="L")

            y_after = pdf.get_y()

            # Draw left accent bar
            pdf.set_fill_color(*header_text_color)
            pdf.rect(pdf.l_margin, y_before, accent_w, y_after - y_before, style="F")

            # Reset to body style
            pdf.set_text_color(*body_text_color)
            if font_family == "CJK":
                pdf.set_font("CJK", size=font_size)
            else:
                pdf.set_font("Helvetica", size=font_size)
            pdf.ln(lh * 0.2)

        elif stripped.startswith("- ") or stripped.startswith("* "):
            # ── Bullet point with colored dot ──
            content = stripped[2:].strip()
            pdf.set_text_color(*body_text_color)

            indent = 5
            bullet_x = pdf.l_margin + indent - 2.2
            bullet_y = pdf.get_y() + lh * 0.42

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
                    pdf.set_fill_color(*bullet_dot_color)
                    pdf.circle(bullet_x, bullet_y, 0.55, style="F")
                pdf.cell(page_width - indent, lh, wline, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="L")

        else:
            # ── Regular paragraph text ──
            # TARGET USERS section: plain black text, no special color
            if current_section == "target_users_use_cases":
                pdf.set_text_color(50, 50, 50)
            else:
                pdf.set_text_color(*body_text_color)
            wrapped = _wrap_to_width(pdf, stripped, page_width)
            for wline in wrapped:
                pdf.set_x(pdf.l_margin)
                pdf.cell(page_width, lh, wline, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="L")


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

async def generate_onepage_pdf(request: PDFExportRequest) -> bytes:
    """
    Run the 3-phase pipeline and return PDF bytes.
    """
    repo_name = request.repo_name or request.repo_url.rstrip("/").split("/")[-1]

    logger.info("=== PDF Export: Phase 1 — Code Semantics Extraction ===")
    semantics = phase1_extract(request.pages, request.repo_url, repo_name)

    logger.info("=== PDF Export: Phase 2 — Analogy Mapping (LLM call) ===")
    summary_text = await phase2_analogy(semantics, request.provider, request.model, request.language)

    logger.info("=== PDF Export: Phase 3 — PDF Rendering ===")
    pdf_bytes = phase3_render_pdf(summary_text, repo_name)

    logger.info("PDF export complete — %d bytes", len(pdf_bytes))
    return pdf_bytes


# ---------------------------------------------------------------------------
# Direct PDF generation (from repo embeddings, no wiki needed)
# ---------------------------------------------------------------------------

def _phase1_extract_from_repo(
    repo_url: str,
    repo_name: str,
    repo_type: str = "github",
    access_token: Optional[str] = None,
    excluded_dirs: Optional[List[str]] = None,
    excluded_files: Optional[List[str]] = None,
    included_dirs: Optional[List[str]] = None,
    included_files: Optional[List[str]] = None,
) -> str:
    """
    Phase 1 for direct PDF: use the existing embedding pipeline + FAISS retrieval
    to extract the most relevant documents from the repo.
    Memory-safe: limits top-K results and caps total context size.
    """
    from api.data_pipeline import DatabaseManager
    from api.config import get_embedder_type
    from api.tools.embedder import get_embedder
    from adalflow.components.retriever.faiss_retriever import FAISSRetriever

    embedder_type = get_embedder_type()

    # --- Step 1: Prepare / reuse the embedding database ---
    db_manager = DatabaseManager()
    try:
        transformed_docs = db_manager.prepare_database(
            repo_url,
            repo_type=repo_type,
            access_token=access_token,
            embedder_type=embedder_type,
            excluded_dirs=excluded_dirs,
            excluded_files=excluded_files,
            included_dirs=included_dirs,
            included_files=included_files,
        )
    except Exception as e:
        logger.error("Failed to prepare embedding database: %s", e)
        # Fallback: return minimal context
        return f"Repository: {repo_name}\nURL: {repo_url}\n\n(Embedding failed: {e})"

    if not transformed_docs:
        return f"Repository: {repo_name}\nURL: {repo_url}\n\n(No documents found in repository)"

    # --- Step 2: Filter docs with valid embeddings ---
    valid_docs = []
    target_size = None
    size_counts: Dict[int, int] = {}
    for doc in transformed_docs:
        vec = getattr(doc, "vector", None)
        if vec is None:
            continue
        try:
            sz = len(vec) if isinstance(vec, list) else (vec.shape[-1] if hasattr(vec, "shape") else len(vec))
        except Exception:
            continue
        if sz == 0:
            continue
        size_counts[sz] = size_counts.get(sz, 0) + 1

    if not size_counts:
        # No usable embeddings; just dump file names
        file_list = "\n".join(
            getattr(d, "meta_data", {}).get("file_path", "?") for d in transformed_docs[:30]
        )
        return f"Repository: {repo_name}\nURL: {repo_url}\nFiles:\n{file_list}"

    target_size = max(size_counts, key=lambda k: size_counts[k])
    for doc in transformed_docs:
        vec = getattr(doc, "vector", None)
        if vec is None:
            continue
        try:
            sz = len(vec) if isinstance(vec, list) else (vec.shape[-1] if hasattr(vec, "shape") else len(vec))
        except Exception:
            continue
        if sz == target_size:
            valid_docs.append(doc)

    logger.info("Direct PDF Phase 1: %d valid docs (embedding dim=%d)", len(valid_docs), target_size)

    # --- Step 3: Retrieve top-K most relevant docs via FAISS ---
    # Use a broad query to get the most informative documents
    TOP_K = min(12, len(valid_docs))  # keep small for memory
    MAX_CONTEXT_CHARS = 15000  # hard cap on total context to keep LLM prompt small

    embedder = get_embedder(embedder_type=embedder_type)
    is_ollama = embedder_type == "ollama"

    def query_embedder(query):
        if isinstance(query, list):
            query = query[0]
        return embedder(input=query)

    retrieve_embedder = query_embedder if is_ollama else embedder

    retriever = None
    try:
        retriever = FAISSRetriever(
            **configs["retriever"],
            embedder=retrieve_embedder,
            documents=valid_docs,
            document_map_func=lambda doc: doc.vector,
        )
        overview_query = (
            f"What is {repo_name}? Architecture overview, main modules, "
            f"tech stack, API endpoints, data flow, entry point"
        )
        results = retriever(overview_query)
        retrieved_indices = results[0].doc_indices[:TOP_K] if results and results[0].doc_indices is not None else []
        retrieved_docs = [valid_docs[i] for i in retrieved_indices if i < len(valid_docs)]
    except Exception as e:
        logger.warning("FAISS retrieval failed, falling back to first N docs: %s", e)
        retrieved_docs = valid_docs[:TOP_K]
    finally:
        # Release FAISS memory
        if retriever is not None:
            del retriever
        del valid_docs
        import gc
        gc.collect()

    # --- Step 4: Build compact context string ---
    lines = []
    lines.append(f"Repository: {repo_name}")
    lines.append(f"URL: {repo_url}")
    lines.append(f"Total source documents: {len(transformed_docs)}")
    lines.append("")

    total_chars = 0
    for doc in retrieved_docs:
        meta = getattr(doc, "meta_data", {})
        fp = meta.get("file_path", "unknown")
        text = getattr(doc, "text", "") or ""
        # Truncate individual doc content
        max_per_doc = min(2000, (MAX_CONTEXT_CHARS - total_chars))
        if max_per_doc <= 0:
            break
        snippet = text[:max_per_doc]
        lines.append(f"--- {fp} ---")
        lines.append(snippet)
        lines.append("")
        total_chars += len(snippet) + len(fp) + 10

    result = "\n".join(lines)
    if len(result) > MAX_CONTEXT_CHARS:
        result = result[:MAX_CONTEXT_CHARS] + "\n\n[... truncated for memory safety ...]"

    # Release db_manager
    del db_manager
    del transformed_docs
    import gc
    gc.collect()

    logger.info("Direct PDF Phase 1 complete — %d docs retrieved, %d chars context", len(retrieved_docs), len(result))
    return result


async def generate_direct_pdf(request: DirectPDFExportRequest) -> bytes:
    """
    Generate PDF directly from repo embeddings (no wiki needed).
    Memory-safe pipeline for Ollama users.
    """
    repo_name = request.repo_name or request.repo_url.rstrip("/").split("/")[-1]

    # Parse file filter strings into lists
    excluded_dirs = (
        [d.strip() for d in request.excluded_dirs.split("\n") if d.strip()]
        if request.excluded_dirs else None
    )
    excluded_files = (
        [f.strip() for f in request.excluded_files.split("\n") if f.strip()]
        if request.excluded_files else None
    )
    included_dirs = (
        [d.strip() for d in request.included_dirs.split("\n") if d.strip()]
        if request.included_dirs else None
    )
    included_files = (
        [f.strip() for f in request.included_files.split("\n") if f.strip()]
        if request.included_files else None
    )

    logger.info("=== Direct PDF Export: Phase 1 — Repo Embedding Extraction ===")
    semantics = _phase1_extract_from_repo(
        repo_url=request.repo_url,
        repo_name=repo_name,
        repo_type=request.repo_type,
        access_token=request.access_token,
        excluded_dirs=excluded_dirs,
        excluded_files=excluded_files,
        included_dirs=included_dirs,
        included_files=included_files,
    )

    logger.info("=== Direct PDF Export: Phase 2 — LLM Summary ===")
    summary_text = await phase2_analogy(semantics, request.provider, request.model, request.language)

    # Free semantics string before rendering
    del semantics
    import gc
    gc.collect()

    logger.info("=== Direct PDF Export: Phase 3 — PDF Rendering ===")
    pdf_bytes = phase3_render_pdf(summary_text, repo_name)

    logger.info("Direct PDF export complete — %d bytes", len(pdf_bytes))
    return pdf_bytes
