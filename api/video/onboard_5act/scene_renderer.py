"""
HTML / CSS scene templates for the 5-act onboard video.

Standalone — does not import the legacy ``api/scene_renderer.py`` template
functions. Palettes, SVG illustrations, and base CSS are copied so future
visual iteration here cannot accidentally affect the baseline pipeline
(see ``api/video/onboard_5act/__init__.py`` for the rationale).

The Playwright render machinery (browser lifecycle, sync renderer, thread
executor) IS imported from ``api.scene_renderer`` because it is pure
infrastructure (HTML → PNG) and we want a single Chromium instance per
process, not two.

Public template functions (one per act):
    - render_act1_intro_html(card)    — IMPLEMENTED here (timeline + stats + contributors)
    - render_act2_metaphor_html(card) — IMPLEMENTED here (comic-bullet dialogue)
    - render_act3_io_html(card)       — IMPLEMENTED here
    - render_act4_usecase_html(card)  — IMPLEMENTED here
    - render_act5_setup_html(card)    — IMPLEMENTED here
"""

from __future__ import annotations

import html
from typing import Dict, List, Optional

from api.video.constants import VIDEO_SIZE

VIDEO_WIDTH, VIDEO_HEIGHT = VIDEO_SIZE


# ---------------------------------------------------------------------------
# Colour palettes — 5 act-keyed gradient themes
# (Copied + adapted from api/scene_renderer.py:115-160)
# ---------------------------------------------------------------------------

_PALETTES: Dict[str, Dict[str, str]] = {
    "intro": {
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
    "metaphor": {
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
    "io": {
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
    "usecase": {
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
    "setup": {
        "bg": "linear-gradient(135deg, #08161a 0%, #0c2228 50%, #08161a 100%)",
        "header_bg": "#0c1c20",
        "accent": "#7dd3c0",
        "accent2": "#5d9bff",
        "node_bg": "rgba(20, 50, 56, 0.95)",
        "node_border": "#7dd3c0",
        "text": "#f0fbf6",
        "text_dim": "#a6c8c0",
        "arrow": "#7dd3c0",
    },
}


def _get_palette(section: str) -> Dict[str, str]:
    return _PALETTES.get(section, _PALETTES["intro"])


# ---------------------------------------------------------------------------
# Inline SVG stick-figure illustrations
# (Copied verbatim from api/scene_renderer.py:171-228)
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

_SVG_MAP: Dict[str, str] = {
    "person_thinking": SVG_PERSON_THINKING,
    "person_at_desk": SVG_PERSON_AT_DESK,
    "person_happy": SVG_PERSON_HAPPY,
    "process_gear": SVG_PROCESS_GEAR,
}

SVG_ARROW_RIGHT = '''<svg viewBox="0 0 60 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round">
  <line x1="4" y1="12" x2="48" y2="12"/>
  <polyline points="42,6 48,12 42,18"/>
</svg>'''


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _esc(text) -> str:
    return html.escape(str(text or ""), quote=True)


def _base_css(p: Dict[str, str]) -> str:
    """Shared CSS for all 5-act scenes. Copied from api/scene_renderer.py:248."""
    return f"""
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{
        width: {VIDEO_WIDTH}px; height: {VIDEO_HEIGHT}px;
        background: {p['bg']};
        font-family: 'Segoe UI', system-ui, -apple-system, 'Noto Sans SC',
                     'Apple Color Emoji', 'Segoe UI Emoji', 'Noto Color Emoji', sans-serif;
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
    .content {{ padding: 28px 48px 28px; height: calc(100% - 88px); display: flex; flex-direction: column; }}
    .badge {{
        display: inline-block; background: {p['accent']}22; border: 1px solid {p['accent']}66;
        border-radius: 20px; padding: 5px 14px; font-size: 13px;
        color: {p['accent']}; font-weight: 500; white-space: nowrap;
    }}
    .scene-bar {{
        background: rgba(0,0,0,0.4); border-left: 4px solid {p['accent']};
        border-radius: 6px; padding: 10px 18px; font-size: 16px;
        color: {p['text']}; font-weight: 500; margin-bottom: 16px;
    }}
    """


# ---------------------------------------------------------------------------
# Act 1 — Intro: project card + commit timeline + stats + contributors
# ---------------------------------------------------------------------------

def render_act1_intro_html(card: dict) -> str:
    """Render the Act 1 intro card.

    card:
        {
            "repo_name": str,
            "one_liner": str,
            "timeline": [{"date": "YYYY-MM-DD", "label": str}, ...],
            "stats": {"stars": int, "forks": int, "license": str, "pushed_at": str} | None,
            "headline_contributors": [
                {"rank": 1..3, "login": str, "name": str,
                 "commits": int, "followers": int, "medal": str}, ...
            ],
        }
    """
    p = _get_palette("intro")
    repo_name = card.get("repo_name") or "Repository"
    one_liner = card.get("one_liner") or ""
    milestones = card.get("timeline") or []
    stats = card.get("stats") or {}
    headliners = card.get("headline_contributors") or []

    # ── Stats badges (top right corner; only shown when present) ──
    stat_chips: list[str] = []
    if stats.get("stars"):
        stat_chips.append(f'<span class="stat">⭐ {stats["stars"]:,}</span>')
    if stats.get("forks"):
        stat_chips.append(f'<span class="stat">🍴 {stats["forks"]:,}</span>')
    if stats.get("license"):
        stat_chips.append(f'<span class="stat">📄 {_esc(stats["license"])}</span>')
    if stats.get("pushed_at"):
        stat_chips.append(f'<span class="stat">📅 {_esc(stats["pushed_at"][:10])}</span>')
    stats_html = (
        f'<div class="stats-row">{"".join(stat_chips)}</div>' if stat_chips else ''
    )

    # ── Project card (huge title + tagline) ──
    one_liner_html = (
        f'<div class="oneliner">{_esc(one_liner)}</div>' if one_liner else ''
    )
    project_card = f'''
        <div class="project-card">
            <div class="project-emoji">📦</div>
            <div class="project-text">
                <div class="repo-name">{_esc(repo_name)}</div>
                {one_liner_html}
            </div>
        </div>
    '''

    # ── Timeline ribbon (horizontal milestones with dots + dates) ──
    timeline_html = ''
    if milestones:
        nodes: list[str] = []
        for i, m in enumerate(milestones[:5]):
            is_first = i == 0
            is_last = i == len(milestones) - 1
            classes = "milestone"
            if is_first:
                classes += " first"
            if is_last:
                classes += " last"
            nodes.append(f'''
                <div class="{classes}">
                    <div class="milestone-dot"></div>
                    <div class="milestone-date">{_esc(m.get("date", ""))}</div>
                    <div class="milestone-label">{_esc(m.get("label", ""))}</div>
                </div>
            ''')
        timeline_html = f'''
            <div class="timeline-section">
                <div class="timeline-track">
                    <div class="timeline-line"></div>
                    {"".join(nodes)}
                </div>
            </div>
        '''

    # ── Contributor showcase (medals + names) ──
    contributor_html = ''
    if headliners:
        cards = []
        for c in headliners[:3]:
            avatar_or_medal = (
                f'<img class="ctr-avatar" src="{_esc(c.get("avatar_url", ""))}" alt=""/>'
                if c.get("avatar_url") else
                f'<div class="ctr-medal">{_esc(c.get("medal", "▫️"))}</div>'
            )
            stats_line = f'{c.get("commits", 0)} commits'
            if c.get("followers"):
                stats_line += f' · {c["followers"]:,} followers'
            cards.append(f'''
                <div class="ctr-card">
                    {avatar_or_medal}
                    <div class="ctr-name">{_esc(c.get("name") or c.get("login", ""))}</div>
                    <div class="ctr-stats">{_esc(stats_line)}</div>
                </div>
            ''')
        contributor_html = f'''
            <div class="contributors-section">
                <div class="section-label">CORE CONTRIBUTORS</div>
                <div class="contributors-row">{"".join(cards)}</div>
            </div>
        '''

    extra_css = f"""
        .content {{ padding: 20px 48px 28px; gap: 16px; }}
        .stats-row {{
            position: absolute; top: 24px; right: 48px;
            display: flex; gap: 8px; flex-wrap: wrap;
        }}
        .stat {{
            background: {p['accent']}22; border: 1px solid {p['accent']}55;
            border-radius: 16px; padding: 4px 12px;
            font-size: 14px; color: {p['accent']}; font-weight: 500;
            white-space: nowrap;
        }}

        .project-card {{
            display: flex; align-items: center; gap: 22px;
            background: {p['node_bg']}; border: 2px solid {p['node_border']};
            border-radius: 18px; padding: 22px 28px;
        }}
        .project-emoji {{ font-size: 64px; line-height: 1; flex-shrink: 0; }}
        .project-text {{ flex: 1; min-width: 0; }}
        .repo-name {{
            font-size: 32px; font-weight: 700; color: {p['text']};
            line-height: 1.2; word-break: break-word;
        }}
        .oneliner {{
            margin-top: 6px; font-size: 17px; color: {p['text_dim']};
            line-height: 1.4;
        }}

        .timeline-section {{ flex: 1; display: flex; flex-direction: column; justify-content: center; }}
        .timeline-track {{
            position: relative; display: flex; justify-content: space-between;
            align-items: flex-start; padding: 0 20px;
        }}
        .timeline-line {{
            position: absolute; top: 8px; left: 30px; right: 30px;
            height: 2px; background: {p['accent']}55;
        }}
        .milestone {{
            position: relative; display: flex; flex-direction: column; align-items: center;
            min-width: 0; max-width: 200px; text-align: center;
            z-index: 1; gap: 6px;
        }}
        .milestone-dot {{
            width: 16px; height: 16px; border-radius: 50%;
            background: {p['accent']}; border: 3px solid {p['header_bg']};
            box-shadow: 0 0 12px {p['accent']}99;
        }}
        .milestone.first .milestone-dot {{ background: {p['accent2']}; }}
        .milestone.last .milestone-dot {{
            background: {p['accent']}; box-shadow: 0 0 18px {p['accent']}cc;
            transform: scale(1.2);
        }}
        .milestone-date {{
            font-size: 13px; color: {p['accent']}; font-weight: 600;
            letter-spacing: 0.5px; margin-top: 4px;
        }}
        .milestone-label {{
            font-size: 13px; color: {p['text_dim']}; line-height: 1.3;
            word-break: break-word;
        }}

        .contributors-section {{ display: flex; flex-direction: column; gap: 10px; }}
        .section-label {{
            font-size: 12px; letter-spacing: 2px; color: {p['accent']};
            font-weight: 700;
        }}
        .contributors-row {{
            display: flex; gap: 16px; align-items: stretch;
        }}
        .ctr-card {{
            flex: 1; max-width: 240px;
            background: {p['node_bg']}; border: 1px solid {p['node_border']}66;
            border-radius: 12px; padding: 12px 16px;
            display: flex; align-items: center; gap: 12px;
            min-width: 0;
        }}
        .ctr-avatar {{
            width: 44px; height: 44px; border-radius: 50%;
            border: 2px solid {p['accent']}; flex-shrink: 0;
            object-fit: cover;
        }}
        .ctr-medal {{
            width: 44px; height: 44px; flex-shrink: 0;
            display: flex; align-items: center; justify-content: center;
            font-size: 32px; line-height: 1;
        }}
        .ctr-name {{
            font-size: 15px; font-weight: 600; color: {p['text']};
            line-height: 1.2; overflow: hidden; text-overflow: ellipsis;
            white-space: nowrap;
        }}
        .ctr-stats {{
            font-size: 12px; color: {p['text_dim']}; margin-top: 2px;
            white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
        }}
        .ctr-card > div:nth-child(2) {{ flex: 1; min-width: 0; }}
    """

    return _wrap_page(
        title=card.get("title", repo_name),
        subtitle=card.get("subtitle", ""),
        body=(
            '<div class="content">'
            + stats_html
            + project_card
            + timeline_html
            + contributor_html
            + '</div>'
        ),
        palette=p,
        extra_css=extra_css,
        footer=card.get("footer", ""),
    )


# ---------------------------------------------------------------------------
# Act 2 — Metaphor: comic-bullet dialogue (image gen comes in v2)
# ---------------------------------------------------------------------------

def render_act2_metaphor_html(card: dict) -> str:
    """Render the Act 2 metaphor as a comic-bullet dialogue.

    card:
        {
            "segments": [{"detail": str, "brief": str}, ...],   # 2-5 entries
            "fallback_subject": str,   # used when segments is empty
        }

    Each segment becomes one speech bubble in a vertical conversation
    flow, alternating left/right alignment for visual rhythm. v1 = no
    images. v2 will add an AI-generated illustration per segment using
    the `detail` field as the image-gen prompt.
    """
    p = _get_palette("metaphor")
    segments = card.get("segments") or []
    fallback_subject = card.get("fallback_subject") or "this project"

    # Fallback when LLM didn't produce a metaphor.
    if not segments or len(segments) < 2:
        segments = [
            {"detail": "", "brief": f"Imagine {fallback_subject} as a friendly assistant."},
            {"detail": "", "brief": "It quietly listens to what you need…"},
            {"detail": "", "brief": "…and hands you back exactly the right thing."},
        ]

    # Cap at 5 segments, build alternating bubbles.
    bubble_html: list[str] = []
    for i, seg in enumerate(segments[:5]):
        side = "left" if i % 2 == 0 else "right"
        text = (seg.get("brief") or seg.get("detail") or "").strip()
        if not text:
            continue
        # Alternate the SVG character so the eye has someone "to listen to".
        svg_key = "person_thinking" if side == "left" else "person_happy"
        svg_html = _SVG_MAP.get(svg_key, SVG_PERSON_THINKING)
        bubble_html.append(f'''
            <div class="bubble-row {side}">
                <div class="bubble-svg" style="color:{p['accent']};">{svg_html}</div>
                <div class="bubble-card">
                    <div class="bubble-text">{_esc(text)}</div>
                </div>
            </div>
        ''')

    extra_css = f"""
        .content {{ padding: 24px 64px 28px; gap: 14px; justify-content: center; }}
        .bubble-row {{
            display: flex; align-items: center; gap: 18px;
            max-width: 100%;
        }}
        .bubble-row.left {{ justify-content: flex-start; }}
        .bubble-row.right {{ justify-content: flex-end; flex-direction: row-reverse; }}
        .bubble-svg {{
            flex-shrink: 0; width: 64px; height: 64px;
        }}
        .bubble-svg svg {{ width: 100%; height: 100%; }}
        .bubble-card {{
            position: relative; flex: 1; max-width: 720px;
            background: {p['node_bg']}; border: 2px solid {p['node_border']};
            border-radius: 18px; padding: 14px 22px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.25);
        }}
        .bubble-row.left .bubble-card::before {{
            content: ''; position: absolute; left: -10px; top: 22px;
            border-top: 8px solid transparent; border-bottom: 8px solid transparent;
            border-right: 12px solid {p['node_border']};
        }}
        .bubble-row.right .bubble-card::before {{
            content: ''; position: absolute; right: -10px; top: 22px;
            border-top: 8px solid transparent; border-bottom: 8px solid transparent;
            border-left: 12px solid {p['node_border']};
        }}
        .bubble-text {{
            font-size: 18px; color: {p['text']}; line-height: 1.5;
            font-style: italic; word-break: break-word;
        }}
    """

    return _wrap_page(
        title=card.get("title", "Think of it like this…"),
        subtitle=card.get("subtitle", ""),
        body=(
            '<div class="content">'
            + ''.join(bubble_html)
            + '</div>'
        ),
        palette=p,
        extra_css=extra_css,
        footer=card.get("footer", ""),
    )


# ---------------------------------------------------------------------------
# Act 3 — input → process → output 3-box diagram with icons
# ---------------------------------------------------------------------------

def render_act3_io_html(card: dict) -> str:
    """Render the 3-box I/O diagram for Act 3.

    card:
        {"boxes": [{"label": str, "icon": str}, x3]}
    """
    p = _get_palette("io")
    boxes = card.get("boxes") or []
    if len(boxes) != 3:
        # acts.py guarantees 3, but be defensive.
        boxes = (boxes + [{"label": "...", "icon": "❓"}] * 3)[:3]

    role_titles = ["INPUT", "PROCESS", "OUTPUT"]
    box_html: list[str] = []
    for i, b in enumerate(boxes):
        is_last = i == len(boxes) - 1
        role = role_titles[i] if i < len(role_titles) else ""
        hl = " highlight" if is_last else ""
        box_html.append(f'''
            <div style="flex:1;display:flex;flex-direction:column;align-items:center;gap:10px;min-width:0;max-width:300px;">
                <div style="font-size:12px;letter-spacing:2px;color:{p['accent']};font-weight:700;">{role}</div>
                <div class="iobox{hl}">
                    <div class="iobox-icon">{_esc(b.get('icon', ''))}</div>
                    <div class="iobox-label">{_esc(b.get('label', ''))}</div>
                </div>
            </div>
        ''')

    flow: list[str] = []
    for i, h in enumerate(box_html):
        flow.append(h)
        if i < len(box_html) - 1:
            flow.append(f'''
                <div style="flex-shrink:0;color:{p['arrow']};display:flex;align-items:center;padding:0 10px;">
                    <svg viewBox="0 0 60 24" width="60" height="24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round">
                        <line x1="4" y1="12" x2="48" y2="12"/>
                        <polyline points="42,6 48,12 42,18"/>
                    </svg>
                </div>
            ''')

    extra_css = f"""
        .iobox {{
            background: {p['node_bg']}; border: 2px solid {p['node_border']};
            border-radius: 18px; padding: 24px 24px;
            display: flex; flex-direction: column; align-items: center;
            gap: 14px; width: 100%;
        }}
        .iobox.highlight {{
            border-color: {p['accent']}; box-shadow: 0 0 28px {p['accent']}44;
        }}
        .iobox-icon {{ font-size: 64px; line-height: 1; }}
        .iobox-label {{
            font-size: 18px; font-weight: 600; color: {p['text']};
            text-align: center; line-height: 1.4; word-break: break-word;
        }}
    """

    return _wrap_page(
        title=card.get("title", "How it works"),
        subtitle=card.get("subtitle", ""),
        body=f'<div class="content"><div style="display:flex;align-items:center;justify-content:center;gap:0;flex:1;padding:20px 0;">{"".join(flow)}</div></div>',
        palette=p,
        extra_css=extra_css,
        footer=card.get("footer", ""),
    )


# ---------------------------------------------------------------------------
# Act 4 — 3-panel use-case comic (problem / use / value)
# ---------------------------------------------------------------------------

def render_act4_usecase_html(card: dict) -> str:
    """Render the 3-panel use-case comic for Act 4.

    card:
        {
            "scene_context": str,    # one-line setting at top
            "panels": [
                {"role": "problem"|"use"|"value", "label": str,
                 "speech": str, "svg": str},
                ...
            ],
        }
    """
    p = _get_palette("usecase")
    panels = card.get("panels") or []
    scene_context = card.get("scene_context") or ""

    panel_html: list[str] = []
    for panel in panels[:5]:
        svg_key = panel.get("svg", "person_thinking")
        svg = _SVG_MAP.get(svg_key, SVG_PERSON_THINKING)
        speech = _esc(panel.get("speech", ""))
        label = _esc(panel.get("label", ""))
        panel_html.append(f'''
            <div class="comic-panel">
                <div class="comic-bubble">{speech}</div>
                <div class="comic-svg" style="color:{p['accent']};">{svg}</div>
                <div class="comic-label">{label}</div>
            </div>
        ''')

    flow: list[str] = []
    for i, h in enumerate(panel_html):
        flow.append(h)
        if i < len(panel_html) - 1:
            flow.append(f'''
                <div class="comic-arrow" style="color:{p['arrow']};">{SVG_ARROW_RIGHT}</div>
            ''')

    scene_bar = (
        f'<div class="scene-bar">📍 {_esc(scene_context)}</div>'
        if scene_context else ''
    )

    extra_css = f"""
        .comic-panel {{
            flex: 1; display: flex; flex-direction: column; align-items: center;
            justify-content: flex-start; min-width: 0; max-width: 340px;
            text-align: center; padding: 0 8px;
        }}
        .comic-bubble {{
            position: relative; background: {p['node_bg']}; border: 2px solid {p['node_border']};
            border-radius: 16px; padding: 14px 20px; margin-bottom: 16px;
            font-size: 16px; color: {p['text']}; line-height: 1.4;
            text-align: center; max-width: 280px; word-break: break-word;
        }}
        .comic-bubble::after {{
            content: ''; position: absolute; bottom: -10px; left: 50%; transform: translateX(-50%);
            border-left: 9px solid transparent; border-right: 9px solid transparent;
            border-top: 11px solid {p['node_border']};
        }}
        .comic-svg {{ width: 90px; height: 90px; margin-bottom: 8px; }}
        .comic-svg svg {{ width: 100%; height: 100%; }}
        .comic-label {{
            font-size: 16px; font-weight: 700; color: {p['accent']};
            letter-spacing: 0.5px;
        }}
        .comic-arrow {{ flex-shrink: 0; width: 50px; padding: 0 2px; display: flex; align-items: center; }}
        .comic-arrow svg {{ width: 100%; height: 24px; }}
    """

    return _wrap_page(
        title=card.get("title", "A typical scenario"),
        subtitle=card.get("subtitle", ""),
        body=(
            '<div class="content">'
            + scene_bar
            + '<div style="display:flex;align-items:flex-start;justify-content:center;gap:0;flex:1;padding:0 20px;">'
            + ''.join(flow) + '</div>'
            + '</div>'
        ),
        palette=p,
        extra_css=extra_css,
        footer=card.get("footer", ""),
    )


# ---------------------------------------------------------------------------
# Act 5 — Setup checklist (prerequisites + numbered steps with icons)
# ---------------------------------------------------------------------------

def render_act5_setup_html(card: dict) -> str:
    """Render the Act 5 setup checklist.

    card:
        {
            "prerequisites": [str, ...],   # 0-3 items
            "steps": [{"text": str, "icon": str}, ...],  # 3-5 items
        }
    """
    p = _get_palette("setup")
    prereqs = card.get("prerequisites") or []
    steps = card.get("steps") or []

    # Prerequisites (top section, optional)
    prereq_html = ""
    if prereqs:
        chips = "".join(
            f'<span class="prereq-chip">📋 {_esc(pr)}</span>'
            for pr in prereqs[:3]
        )
        prereq_html = f'''
            <div class="prereq-row">
                <div class="prereq-label">Before you start:</div>
                <div class="prereq-chips">{chips}</div>
            </div>
        '''

    # Steps (numbered checklist)
    step_html_parts: list[str] = []
    for i, step in enumerate(steps[:5], start=1):
        text = _esc(step.get("text", ""))
        icon = _esc(step.get("icon", "🔸"))
        step_html_parts.append(f'''
            <div class="step-row">
                <div class="step-num">{i}</div>
                <div class="step-icon">{icon}</div>
                <div class="step-text">{text}</div>
            </div>
        ''')

    extra_css = f"""
        .prereq-row {{
            display: flex; align-items: center; gap: 14px; flex-wrap: wrap;
            margin-bottom: 20px; padding: 12px 16px;
            background: rgba(125, 211, 192, 0.08);
            border-left: 3px solid {p['accent']}; border-radius: 6px;
        }}
        .prereq-label {{
            font-size: 14px; color: {p['text_dim']};
            letter-spacing: 0.5px; font-weight: 500; flex-shrink: 0;
        }}
        .prereq-chips {{ display: flex; gap: 8px; flex-wrap: wrap; }}
        .prereq-chip {{
            font-size: 14px; color: {p['text']};
            background: {p['node_bg']}; border: 1px solid {p['node_border']}66;
            border-radius: 14px; padding: 4px 12px;
        }}
        .step-row {{
            display: flex; align-items: center; gap: 18px;
            background: {p['node_bg']}; border: 2px solid {p['node_border']}55;
            border-radius: 14px; padding: 14px 22px;
            margin-bottom: 12px;
        }}
        .step-num {{
            flex-shrink: 0; width: 36px; height: 36px;
            background: {p['accent']}; color: #0a1620;
            border-radius: 50%; display: flex; align-items: center; justify-content: center;
            font-size: 18px; font-weight: 700;
        }}
        .step-icon {{ flex-shrink: 0; font-size: 32px; line-height: 1; }}
        .step-text {{
            flex: 1; font-size: 17px; color: {p['text']};
            line-height: 1.4; word-break: break-word;
            font-family: 'SF Mono', Menlo, Consolas, 'DejaVu Sans Mono', monospace;
        }}
    """

    return _wrap_page(
        title=card.get("title", "Get it running in 5 minutes"),
        subtitle=card.get("subtitle", ""),
        body=(
            '<div class="content">'
            + prereq_html
            + '<div style="flex:1;display:flex;flex-direction:column;justify-content:center;">'
            + ''.join(step_html_parts)
            + '</div>'
            + '</div>'
        ),
        palette=p,
        extra_css=extra_css,
        footer=card.get("footer", ""),
    )


# ---------------------------------------------------------------------------
# Page wrapper (header + body + footer + base css + extra css)
# ---------------------------------------------------------------------------

def _wrap_page(
    *, title: str, subtitle: str, body: str, palette: Dict[str, str],
    extra_css: str = "", footer: str = "",
) -> str:
    """Wrap an act body in the standard page chrome (header + footer)."""
    return (
        '<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">'
        + f'<meta name="viewport" content="width={VIDEO_WIDTH}">'
        + '<style>' + _base_css(palette) + extra_css + '</style></head><body>'
        + '<div class="header">'
        + (f'<div class="subtitle">{_esc(subtitle)}</div>' if subtitle else '')
        + f'<div class="title">{_esc(title)}</div>'
        + '</div>'
        + body
        + (f'<div class="footer">{_esc(footer)}</div>' if footer else '')
        + '</body></html>'
    )
