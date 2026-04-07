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


# ---------------------------------------------------------------------------
# Inline SVG stick-figure illustrations (120x120 viewBox, currentColor stroke)
# ---------------------------------------------------------------------------

SVG_PERSON_THINKING = '''<svg viewBox="0 0 120 120" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
  <circle cx="60" cy="28" r="14"/>
  <line x1="60" y1="42" x2="60" y2="78"/>
  <line x1="60" y1="78" x2="44" y2="105"/>
  <line x1="60" y1="78" x2="76" y2="105"/>
  <line x1="60" y1="55" x2="38" y2="48"/>
  <line x1="60" y1="55" x2="82" y2="42"/>
  <circle cx="92" cy="22" r="8" stroke-dasharray="4 3" opacity="0.6"/>
  <text x="89" y="26" font-size="12" fill="currentColor" stroke="none" text-anchor="middle">?</text>
</svg>'''

SVG_PERSON_AT_DESK = '''<svg viewBox="0 0 120 120" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
  <circle cx="45" cy="25" r="12"/>
  <line x1="45" y1="37" x2="45" y2="68"/>
  <line x1="45" y1="68" x2="32" y2="92"/>
  <line x1="45" y1="68" x2="58" y2="92"/>
  <line x1="45" y1="50" x2="30" y2="58"/>
  <line x1="45" y1="50" x2="65" y2="52"/>
  <rect x="60" y="40" width="36" height="26" rx="3"/>
  <line x1="60" y1="66" x2="96" y2="66"/>
  <line x1="68" y1="66" x2="62" y2="76"/>
  <line x1="88" y1="66" x2="94" y2="76"/>
  <line x1="66" y1="50" x2="90" y2="50" stroke-dasharray="3 2" opacity="0.5"/>
  <line x1="66" y1="56" x2="84" y2="56" stroke-dasharray="3 2" opacity="0.5"/>
</svg>'''

SVG_PERSON_HAPPY = '''<svg viewBox="0 0 120 120" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
  <circle cx="60" cy="28" r="14"/>
  <line x1="60" y1="42" x2="60" y2="78"/>
  <line x1="60" y1="78" x2="44" y2="105"/>
  <line x1="60" y1="78" x2="76" y2="105"/>
  <line x1="60" y1="52" x2="36" y2="36"/>
  <line x1="60" y1="52" x2="84" y2="36"/>
  <path d="M52 30 Q60 38 68 30" fill="none"/>
  <circle cx="90" cy="18" r="5" fill="currentColor" opacity="0.3"/>
  <circle cx="98" cy="28" r="3" fill="currentColor" opacity="0.2"/>
  <text x="30" y="30" font-size="14" fill="currentColor" stroke="none">!</text>
</svg>'''

SVG_PROCESS_GEAR = '''<svg viewBox="0 0 120 120" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
  <circle cx="60" cy="60" r="18"/>
  <circle cx="60" cy="60" r="8"/>
  <line x1="60" y1="38" x2="60" y2="30"/>
  <line x1="60" y1="82" x2="60" y2="90"/>
  <line x1="38" y1="60" x2="30" y2="60"/>
  <line x1="82" y1="60" x2="90" y2="60"/>
  <line x1="44.4" y1="44.4" x2="38.4" y2="38.4"/>
  <line x1="75.6" y1="75.6" x2="81.6" y2="81.6"/>
  <line x1="44.4" y1="75.6" x2="38.4" y2="81.6"/>
  <line x1="75.6" y1="44.4" x2="81.6" y2="38.4"/>
</svg>'''

_SVG_MAP = {
    "person_thinking": SVG_PERSON_THINKING,
    "person_at_desk": SVG_PERSON_AT_DESK,
    "person_happy": SVG_PERSON_HAPPY,
    "process_gear": SVG_PROCESS_GEAR,
}

SVG_ARROW_RIGHT = '''<svg viewBox="0 0 60 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round">
  <line x1="4" y1="12" x2="48" y2="12"/>
  <polyline points="42,6 48,12 42,18"/>
</svg>'''


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
    .content {{ padding: 20px 48px 28px; height: calc(100% - 88px); display: flex; flex-direction: column; }}
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
    .highlight {{ box-shadow: 0 0 20px {p['accent']}66, 0 0 40px {p['accent']}22; border-color: {p['accent']}; transform: scale(1.05); transition: all 0.2s; }}
    .dim {{ opacity: 0.4; transition: opacity 0.2s; }}
    .subtitle-bar {{
        position: absolute; bottom: 36px; left: 48px; right: 48px;
        background: rgba(0,0,0,0.72); border-radius: 10px; padding: 12px 24px;
        font-size: 17px; color: #fff; text-align: center; line-height: 1.5;
        overflow: hidden; display: -webkit-box; -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
    }}
    .comic-panel {{
        flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: center;
        min-width: 0; max-width: 340px; text-align: center; padding: 0 8px;
        transition: opacity 0.2s, transform 0.2s;
        border: none; background: none;
    }}
    .comic-panel.highlight {{
        box-shadow: none; border: none; transform: scale(1.08);
    }}
    .comic-panel.highlight .comic-bubble {{
        box-shadow: 0 0 24px {p['accent']}66, 0 0 48px {p['accent']}22;
        border-color: {p['accent']};
    }}
    .comic-panel.highlight .comic-svg {{ filter: drop-shadow(0 0 8px {p['accent']}88); }}
    .comic-panel.highlight .comic-label {{ color: {p['accent']}; }}
    .comic-panel.dim {{ opacity: 0.35; transform: scale(0.95); }}
    .comic-bubble {{
        position: relative; background: {p['node_bg']}; border: 2px solid {p['node_border']};
        border-radius: 16px; padding: 14px 20px; margin-bottom: 16px;
        font-size: 16px; color: {p['text']}; line-height: 1.4; text-align: center;
        max-width: 280px; word-break: break-word;
    }}
    .comic-bubble::after {{
        content: ''; position: absolute; bottom: -12px; left: 50%; transform: translateX(-50%);
        border-left: 10px solid transparent; border-right: 10px solid transparent;
        border-top: 12px solid {p['node_border']};
    }}
    .comic-bubble-inner::after {{
        content: ''; position: absolute; bottom: -9px; left: 50%; transform: translateX(-50%);
        border-left: 8px solid transparent; border-right: 8px solid transparent;
        border-top: 10px solid {p['node_bg']};
    }}
    .comic-svg {{ width: 90px; height: 90px; margin-bottom: 8px; }}
    .comic-svg svg {{ width: 100%; height: 100%; }}
    .comic-label {{ font-size: 16px; font-weight: 700; color: {p['accent']}; letter-spacing: 0.5px; }}
    .comic-arrow {{ flex-shrink: 0; width: 50px; padding: 0 2px; display: flex; align-items: center; }}
    .comic-arrow svg {{ width: 100%; height: 24px; }}
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


def _render_comic_panels_html(personas, p, tech_chips=None, section_label="", subtitle_text="", highlight_labels=None):
    """Shared comic renderer: stick figures with speech bubbles above them."""
    highlight_labels = highlight_labels or []
    tech_chips = tech_chips or []

    panels = []
    for i, persona in enumerate(personas[:3]):
        svg_key = persona.get("svg", "person_thinking")
        svg_html = _SVG_MAP.get(svg_key, SVG_PERSON_THINKING)
        label = _esc(persona.get("label", ""))
        caption = _esc(persona.get("caption", ""))
        hl_class = ""
        if highlight_labels:
            if label.lower() in [h.lower() for h in highlight_labels]:
                hl_class = " highlight"
            else:
                hl_class = " dim"

        # Speech bubble above the figure, then SVG, then label below
        bubble_border = p['accent'] if hl_class == " highlight" else p['node_border']
        panels.append(
            '<div class="comic-panel' + hl_class + '">'
            + '<div class="comic-bubble" style="border-color:' + bubble_border + ';">'
            + '<span class="comic-bubble-inner"></span>'
            + caption
            + '</div>'
            + '<div class="comic-svg" style="color:' + p['accent'] + ';">' + svg_html + '</div>'
            + '<div class="comic-label">' + label + '</div>'
            + '</div>'
        )

    flow = []
    for i, panel in enumerate(panels):
        flow.append(panel)
        if i < len(panels) - 1:
            flow.append(
                '<div class="comic-arrow" style="color:' + p['arrow'] + ';">' + SVG_ARROW_RIGHT + '</div>'
            )

    badges = ''.join('<span class="badge">' + _esc(t) + '</span>' for t in tech_chips[:6])
    section_header = ""
    if section_label:
        section_header = (
            '<div style="text-align:center;font-size:14px;color:' + p['accent']
            + ';margin-bottom:10px;font-weight:500;letter-spacing:1px;text-transform:uppercase;">'
            + _esc(section_label) + '</div>'
        )

    subtitle_html = ""
    if subtitle_text:
        subtitle_html = '<div class="subtitle-bar">' + _esc(subtitle_text) + '</div>'

    return (
        '<div class="content" style="justify-content:center;align-items:center;padding-bottom:64px;">'
        + section_header
        + '<div style="display:flex;align-items:center;justify-content:center;gap:0;flex:1;padding:0 20px;">'
        + ''.join(flow) + '</div>'
        + ('<div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-top:14px;justify-content:center;">'
           + badges + '</div>' if badges else '')
        + '</div>'
        + subtitle_html
    )


def _render_overview_comic_html(card, p, subtitle_text="", highlight_labels=None):
    """Comic-style overview: stick figures with speech bubbles."""
    return _render_comic_panels_html(
        card.get("personas") or [],
        p,
        tech_chips=card.get("tech_chips"),
        subtitle_text=subtitle_text,
        highlight_labels=highlight_labels,
    )


def _render_summary_comic_html(card, p, subtitle_text="", highlight_labels=None):
    """Comic-style summary: user journey with stick figures and speech bubbles."""
    return _render_comic_panels_html(
        card.get("personas") or [],
        p,
        tech_chips=card.get("keywords"),
        section_label="End-to-End Journey",
        subtitle_text=subtitle_text,
        highlight_labels=highlight_labels,
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


def build_scene_html(card, expansion_index=1, subtitle_text="", highlight_labels=None):
    section = card.get("section", "overview")
    p = _get_palette(section)
    highlight_labels = highlight_labels or []

    if section == "overview":
        content = _render_overview_comic_html(card, p, subtitle_text=subtitle_text, highlight_labels=highlight_labels)
    elif section == "core":
        content = _render_core_html(card, p)
    elif section == "expansion":
        content = _render_expansion_html(card, p, expansion_index)
    elif section == "summary":
        content = _render_summary_comic_html(card, p, subtitle_text=subtitle_text, highlight_labels=highlight_labels)
    else:
        content = _render_overview_comic_html(card, p, subtitle_text=subtitle_text, highlight_labels=highlight_labels)

    # For non-comic templates, append subtitle bar if provided
    if section in ("core", "expansion") and subtitle_text:
        content += '<div class="subtitle-bar">' + _esc(subtitle_text) + '</div>'

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


async def render_scene_to_png(card, output_path, expansion_index=1, subtitle_text="", highlight_labels=None):
    html_content = build_scene_html(card, expansion_index, subtitle_text=subtitle_text, highlight_labels=highlight_labels)
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
