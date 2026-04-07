"""
Scene Renderer - HTML/CSS to Playwright Screenshot.

Replaces Pillow renderer with modern HTML/CSS layouts captured via headless
Chromium. Each scene type gets a distinct visual template with CSS Flexbox
layouts that prevent overlapping, text overflow, and arrow misalignment.
"""

import asyncio
import html
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720

_PALETTES = {
    "overview": {
        "bg": "linear-gradient(135deg, #0a1628 0%, #0f2035 50%, #0a1628 100%)",
        "header_bg": "#0e1828",
        "accent": "#5d9bff",
        "accent2": "#5ed6a0",
        "node_bg": "rgba(24, 40, 68, 0.95)",
        "node_border": "#5d9bff",
        "text": "#f4f8ff",
        "text_dim": "#a5bad6",
        "arrow": "#5d9bff",
    },
    "core": {
        "bg": "linear-gradient(135deg, #091418 0%, #0d2028 50%, #091418 100%)",
        "header_bg": "#0c1a20",
        "accent": "#5ed6a0",
        "accent2": "#5d9bff",
        "node_bg": "rgba(21, 53, 66, 0.95)",
        "node_border": "#5ed6a0",
        "text": "#f0f6ff",
        "text_dim": "#a6c2b8",
        "arrow": "#5ed6a0",
    },
    "expansion": {
        "bg": "linear-gradient(135deg, #1a1008 0%, #1f1610 50%, #1a1008 100%)",
        "header_bg": "#1a1210",
        "accent": "#ffa84c",
        "accent2": "#ff6b6b",
        "node_bg": "rgba(72, 47, 27, 0.95)",
        "node_border": "#ffa84c",
        "text": "#fcf1e4",
        "text_dim": "#d4b896",
        "arrow": "#ffa84c",
    },
    "summary": {
        "bg": "linear-gradient(135deg, #12081a 0%, #1a1028 50%, #12081a 100%)",
        "header_bg": "#140e1e",
        "accent": "#c98dff",
        "accent2": "#ff8dd2",
        "node_bg": "rgba(54, 41, 82, 0.95)",
        "node_border": "#c98dff",
        "text": "#f4edff",
        "text_dim": "#bfaad6",
        "arrow": "#c98dff",
    },
}


def _esc(text):
    return html.escape(str(text or ""), quote=True)


def _get_palette(section):
    return _PALETTES.get(section, _PALETTES["overview"])


def _base_css(p):
    return f"""
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{
        width: {VIDEO_WIDTH}px; height: {VIDEO_HEIGHT}px;
        background: {p['bg']};
        font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
        color: {p['text']}; overflow: hidden; position: relative;
    }}
    .header {{
        background: {p['header_bg']};
        padding: 18px 48px 16px;
        border-bottom: 2px solid {p['accent']}33;
    }}
    .header .subtitle {{ font-size: 15px; color: {p['text_dim']}; letter-spacing: 0.5px; margin-bottom: 4px; }}
    .header .title {{
        font-size: 32px; font-weight: 700; color: {p['text']};
        line-height: 1.2; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
    }}
    .footer {{ position: absolute; bottom: 12px; left: 48px; font-size: 14px; color: {p['text_dim']}88; }}
    .content {{ padding: 28px 48px 48px; height: calc(100% - 88px); display: flex; flex-direction: column; }}
    .node {{
        background: {p['node_bg']}; border: 2px solid {p['node_border']};
        border-radius: 14px; padding: 16px 20px; text-align: center;
        display: flex; flex-direction: column; justify-content: center; align-items: center;
        min-width: 0; overflow: hidden;
    }}
    .node .node-label {{ font-size: 18px; font-weight: 600; word-break: break-word; line-height: 1.3; }}
    .node .node-desc {{
        font-size: 13px; color: {p['text_dim']}; margin-top: 6px; line-height: 1.4;
        overflow: hidden; display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical;
    }}
    .highlight {{ box-shadow: 0 0 20px {p['accent']}66, 0 0 40px {p['accent']}22; border-color: {p['accent']}; }}
    .badge {{
        display: inline-block; background: {p['accent']}22; border: 1px solid {p['accent']}66;
        border-radius: 20px; padding: 5px 14px; font-size: 13px;
        color: {p['accent']}; font-weight: 500; white-space: nowrap;
    }}
    .arrow-container {{ display: flex; align-items: center; justify-content: center; flex-shrink: 0; }}
    .arrow-label {{ font-size: 12px; color: {p['arrow']}; text-align: center; white-space: nowrap; padding: 0 4px; }}
    .arrow-line {{
        width: 40px; height: 2px; background: {p['arrow']}; position: relative;
    }}
    .arrow-line::after {{
        content: ''; position: absolute; right: -1px; top: -5px;
        border: 6px solid transparent; border-left: 8px solid {p['arrow']};
    }}
    .arrow-down {{
        width: 2px; height: 30px; background: {p['arrow']}; margin: 0 auto; position: relative;
    }}
    .arrow-down::after {{
        content: ''; position: absolute; bottom: -1px; left: -5px;
        border: 6px solid transparent; border-top: 8px solid {p['arrow']};
    }}
    """


def _render_overview_html(card, p):
    entities = card.get("entities") or []
    relations = card.get("relations") or []
    tech_chips = card.get("tech_chips") or []
    overview_descs = card.get("overview_descriptions") or []
    narration = card.get("narration", "")

    nodes_html = []
    for i, ent in enumerate(entities[:4]):
        label = _esc(ent.get("label", ""))
        hl = " highlight" if i == 0 else ""
        desc_html = ""
        if i < len(overview_descs) and overview_descs[i]:
            desc_html = '<div class="node-desc">' + _esc(overview_descs[i]) + '</div>'
        nodes_html.append(
            '<div class="node' + hl + '" style="flex:1;min-width:160px;max-width:280px;">'
            + '<div class="node-label">' + label + '</div>'
            + desc_html
            + '</div>'
        )

    flow = []
    for i, nh in enumerate(nodes_html):
        flow.append(nh)
        if i < len(nodes_html) - 1:
            rl = _esc(relations[i].get("type", "")) if i < len(relations) else ""
            flow.append(
                '<div class="arrow-container" style="flex-direction:column;padding:0 6px;">'
                + '<div class="arrow-label">' + rl + '</div>'
                + '<div class="arrow-line"></div>'
                + '</div>'
            )

    badges = ''.join('<span class="badge">' + _esc(t) + '</span>' for t in tech_chips[:6])

    return (
        '<div class="content">'
        + '<div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:20px;">'
        + badges + '</div>'
        + '<div style="display:flex;align-items:center;justify-content:center;gap:0;flex:1;padding:10px 0;">'
        + ''.join(flow) + '</div>'
        + '<div style="background:rgba(93,155,255,0.08);border:1px solid rgba(93,155,255,0.25);'
        + 'border-radius:12px;padding:14px 22px;margin-top:16px;font-size:15px;line-height:1.5;'
        + 'color:' + p['text_dim'] + ';overflow:hidden;display:-webkit-box;-webkit-line-clamp:3;'
        + '-webkit-box-orient:vertical;">' + _esc(narration) + '</div>'
        + '</div>'
    )


def _render_core_html(card, p):
    entities = card.get("entities") or []
    relations = card.get("relations") or []
    core_descs = card.get("core_descriptions") or []
    microcopy = card.get("microcopy") or []

    nodes = []
    for i, ent in enumerate(entities[:4]):
        label = _esc(ent.get("label", ""))
        desc_html = ""
        if i < len(core_descs) and core_descs[i]:
            desc_html = '<div class="node-desc">' + _esc(core_descs[i]) + '</div>'
        nodes.append(
            '<div class="node highlight" style="flex:1;min-width:140px;max-width:300px;">'
            + '<div class="node-label" style="font-size:20px;">' + label + '</div>'
            + desc_html
            + '</div>'
        )

    flow = []
    for i, nh in enumerate(nodes):
        flow.append(nh)
        if i < len(nodes) - 1:
            rl = _esc(relations[i].get("type", "")) if i < len(relations) else ""
            flow.append(
                '<div class="arrow-container" style="flex-direction:column;padding:0 8px;">'
                + '<div class="arrow-label">' + rl + '</div>'
                + '<div class="arrow-line" style="width:50px;"></div>'
                + '</div>'
            )

    # Use narration as the bottom info bar
    narration = card.get("narration", "")
    return (
        '<div class="content">'
        + '<div style="text-align:center;font-size:15px;color:' + p['accent']
        + ';margin-bottom:14px;font-weight:500;letter-spacing:1px;text-transform:uppercase;">'
        + 'Minimum Viable System</div>'
        + '<div style="display:flex;align-items:center;justify-content:center;gap:0;flex:1;padding:20px 0;">'
        + ''.join(flow) + '</div>'
        + '<div style="background:rgba(94,214,160,0.08);border:1px solid rgba(94,214,160,0.2);'
        + 'border-radius:12px;padding:14px 22px;margin-top:10px;font-size:14px;line-height:1.5;'
        + 'color:' + p['text_dim'] + ';overflow:hidden;display:-webkit-box;-webkit-line-clamp:3;'
        + '-webkit-box-orient:vertical;">' + _esc(narration) + '</div>'
        + '</div>'
    )


def _render_expansion_html(card, p, expansion_index=1):
    entities = card.get("entities") or []
    focus_modules = card.get("focus_modules") or []
    module_details = card.get("module_details") or []
    narration = card.get("narration", "")

    # Use module_details for accurate data
    if module_details:
        md = module_details[0]
        module_name = _esc(md.get("name", "Module"))
        problem_text = _esc(md.get("solves", "Gap in the system"))
        solution_text = _esc(md.get("role", "Extends the core"))
        capability = _esc(md.get("position", "Enhanced capability"))
    else:
        module_name = _esc(entities[0].get("label", "Module")) if entities else _esc(focus_modules[0] if focus_modules else "Module")
        core_label_raw = entities[1].get("label", "Core") if len(entities) > 1 else "Core System"
        problem_text = _esc(entities[2].get("label", "Gap")) if len(entities) > 2 else "Gap in the system"
        solution_text = "Extends the core"
        capability = _esc(core_label_raw)

    if expansion_index % 2 == 1:
        # Variant A: left-right problem -> solution
        return (
            '<div class="content">'
            + '<div style="display:flex;gap:24px;flex:1;align-items:stretch;">'
            # Problem panel
            + '<div style="flex:1;background:rgba(255,107,107,0.06);border:1px solid rgba(255,107,107,0.3);'
            + 'border-radius:16px;padding:24px;display:flex;flex-direction:column;justify-content:center;">'
            + '<div style="font-size:13px;color:#ff6b6b;font-weight:600;text-transform:uppercase;'
            + 'letter-spacing:1px;margin-bottom:12px;">&#9888; Problem</div>'
            + '<div style="font-size:16px;line-height:1.5;color:' + p['text'] + 'cc;">'
            + problem_text + '</div>'
            + '<div style="margin-top:16px;font-size:14px;color:' + p['text_dim'] + ';">'
            + 'Without <strong style="color:' + p['accent'] + ';">' + module_name + '</strong>, the system is limited.</div>'
            + '</div>'
            # Arrow
            + '<div style="display:flex;flex-direction:column;justify-content:center;align-items:center;padding:0 4px;">'
            + '<div class="arrow-line" style="width:50px;"></div>'
            + '</div>'
            # Solution panel
            + '<div style="flex:1.2;display:flex;flex-direction:column;gap:16px;justify-content:center;">'
            + '<div class="node highlight" style="padding:20px 24px;">'
            + '<div class="node-label" style="font-size:22px;">' + module_name + '</div>'
            + '<div class="node-desc" style="font-size:14px;">' + solution_text + '</div>'
            + '</div>'
            + '<div style="display:flex;align-items:center;gap:12px;padding:0 8px;"><div class="arrow-down"></div></div>'
            + '<div class="node" style="background:rgba(94,214,160,0.1);border-color:rgba(94,214,160,0.4);padding:14px 20px;">'
            + '<div class="node-label" style="font-size:16px;color:#5ed6a0;">&#10003; ' + capability + '</div>'
            + '</div>'
            + '</div>'
            + '</div>'
            # Narration bar
            + '<div style="background:rgba(255,168,76,0.06);border:1px solid rgba(255,168,76,0.2);'
            + 'border-radius:12px;padding:12px 20px;margin-top:14px;font-size:14px;line-height:1.5;'
            + 'color:' + p['text_dim'] + ';overflow:hidden;display:-webkit-box;-webkit-line-clamp:2;'
            + '-webkit-box-orient:vertical;">' + _esc(narration) + '</div>'
            + '</div>'
        )
    else:
        # Variant B: top-down layered architecture
        core_label = _esc(entities[1].get("label", "Core System")) if len(entities) > 1 else "Core System"
        return (
            '<div class="content">'
            + '<div style="display:flex;flex-direction:column;gap:16px;flex:1;justify-content:center;">'
            # Existing foundation
            + '<div style="display:flex;align-items:center;gap:16px;padding:12px 20px;'
            + 'background:rgba(93,155,255,0.06);border:1px solid rgba(93,155,255,0.2);border-radius:12px;">'
            + '<div style="width:8px;min-height:40px;background:' + p['accent2'] + ';border-radius:4px;flex-shrink:0;"></div>'
            + '<div>'
            + '<div style="font-size:13px;color:' + p['accent2'] + ';font-weight:600;text-transform:uppercase;letter-spacing:1px;">Existing Foundation</div>'
            + '<div style="font-size:16px;color:' + p['text'] + 'cc;margin-top:4px;">' + core_label + '</div>'
            + '</div></div>'
            # Gap
            + '<div style="text-align:center;padding:4px 0;">'
            + '<span style="display:inline-block;background:rgba(255,107,107,0.12);border:1px dashed rgba(255,107,107,0.4);'
            + 'border-radius:8px;padding:8px 20px;font-size:14px;color:#ff6b6b;max-width:90%;overflow:hidden;text-overflow:ellipsis;">' + problem_text + '</span>'
            + '</div>'
            # New module
            + '<div class="node highlight" style="padding:20px 28px;align-self:center;max-width:80%;">'
            + '<div style="font-size:13px;color:' + p['accent'] + ';font-weight:600;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;">New Module</div>'
            + '<div class="node-label" style="font-size:22px;">' + module_name + '</div>'
            + '<div class="node-desc" style="font-size:14px;margin-top:8px;">' + solution_text + '</div>'
            + '</div>'
            # Result
            + '<div style="display:flex;align-items:center;gap:16px;padding:12px 20px;'
            + 'background:rgba(94,214,160,0.06);border:1px solid rgba(94,214,160,0.25);border-radius:12px;">'
            + '<div style="font-size:20px;flex-shrink:0;">&#10003;</div>'
            + '<div>'
            + '<div style="font-size:13px;color:#5ed6a0;font-weight:600;text-transform:uppercase;letter-spacing:1px;">Result</div>'
            + '<div style="font-size:16px;color:' + p['text'] + 'cc;margin-top:4px;">' + capability + '</div>'
            + '</div></div>'
            + '</div></div>'
        )


def _render_summary_html(card, p):
    entities = card.get("entities") or []
    use_cases = card.get("use_cases") or []
    keywords = card.get("keywords") or []
    narration = card.get("narration", "")

    labels = [_esc(ent.get("label", "Step")) for ent in entities[:4]]
    if not labels:
        labels = ["User", "Workflow", "Outcome"]

    icons = ["&#128100;", "&#9881;&#65039;", "&#10024;", "&#127942;"]

    steps = []
    for i, label in enumerate(labels):
        desc_html = ""
        if i < len(use_cases):
            desc_html = '<div class="node-desc">' + _esc(use_cases[i]) + '</div>'
        is_last = i == len(labels) - 1
        icon = icons[i] if i < len(icons) else icons[-1]
        hl = " highlight" if is_last else ""
        steps.append(
            '<div style="flex:1;display:flex;flex-direction:column;align-items:center;gap:8px;min-width:0;">'
            + '<div class="node' + hl + '" style="width:100%;padding:18px 16px;">'
            + '<div style="font-size:28px;margin-bottom:6px;">' + icon + '</div>'
            + '<div class="node-label" style="font-size:18px;">' + label + '</div>'
            + desc_html
            + '</div></div>'
        )

    flow = []
    for i, sh in enumerate(steps):
        flow.append(sh)
        if i < len(steps) - 1:
            flow.append(
                '<div class="arrow-container" style="flex-direction:column;padding:0 6px;flex-shrink:0;">'
                + '<div class="arrow-line" style="width:36px;"></div></div>'
            )

    badges = ''.join('<span class="badge">' + _esc(k) + '</span>' for k in keywords[:4])

    return (
        '<div class="content">'
        + '<div style="text-align:center;font-size:15px;color:' + p['accent']
        + ';margin-bottom:18px;font-weight:500;letter-spacing:1px;text-transform:uppercase;">End-to-End Journey</div>'
        + '<div style="display:flex;align-items:center;justify-content:center;gap:0;flex:1;padding:10px 20px;">'
        + ''.join(flow) + '</div>'
        + '<div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-top:12px;justify-content:center;">'
        + badges + '</div>'
        + '<div style="background:rgba(201,141,255,0.06);border:1px solid rgba(201,141,255,0.2);'
        + 'border-radius:12px;padding:12px 20px;margin-top:14px;font-size:14px;line-height:1.5;'
        + 'color:' + p['text_dim'] + ';overflow:hidden;display:-webkit-box;-webkit-line-clamp:3;'
        + '-webkit-box-orient:vertical;">' + _esc(narration) + '</div>'
        + '</div>'
    )


def build_scene_html(card, expansion_index=1):
    section = card.get("section", "overview")
    p = _get_palette(section)

    if section == "overview":
        content = _render_overview_html(card, p)
    elif section == "core":
        content = _render_core_html(card, p)
    elif section == "expansion":
        content = _render_expansion_html(card, p, expansion_index)
    elif section == "summary":
        content = _render_summary_html(card, p)
    else:
        content = _render_overview_html(card, p)

    return (
        '<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">'
        + '<meta name="viewport" content="width=' + str(VIDEO_WIDTH) + '">'
        + '<style>' + _base_css(p) + '</style></head><body>'
        + '<div class="header">'
        + '<div class="subtitle">' + _esc(card.get("subtitle", "")) + '</div>'
        + '<div class="title">' + _esc(card.get("title", "")) + '</div>'
        + '</div>'
        + content
        + '<div class="footer">' + _esc(card.get("footer", "")) + '</div>'
        + '</body></html>'
    )


# ---------------------------------------------------------------------------
# Playwright screenshot engine
# ---------------------------------------------------------------------------

_browser_instance = None
_browser_lock = asyncio.Lock()


async def _get_browser():
    global _browser_instance
    async with _browser_lock:
        if _browser_instance is None or not _browser_instance.is_connected():
            from playwright.async_api import async_playwright
            pw = await async_playwright().start()
            # Try Firefox first (fewer system deps), fall back to Chromium
            for launcher, name in [(pw.firefox, "firefox"), (pw.chromium, "chromium")]:
                try:
                    _browser_instance = await launcher.launch(headless=True)
                    logger.info("Playwright %s browser launched", name)
                    break
                except Exception as e:
                    logger.warning("Failed to launch %s: %s", name, e)
            if _browser_instance is None:
                raise RuntimeError("No Playwright browser could be launched")
        return _browser_instance


async def render_scene_to_png(card, output_path, expansion_index=1):
    html_content = build_scene_html(card, expansion_index)
    browser = await _get_browser()
    page = await browser.new_page(viewport={"width": VIDEO_WIDTH, "height": VIDEO_HEIGHT})
    try:
        await page.set_content(html_content, wait_until="networkidle")
        await page.screenshot(path=output_path, type="png")
        logger.info("Scene rendered to PNG: %s", output_path)
    finally:
        await page.close()


async def close_browser():
    global _browser_instance
    async with _browser_lock:
        if _browser_instance and _browser_instance.is_connected():
            await _browser_instance.close()
            _browser_instance = None
            logger.info("Playwright browser closed")
