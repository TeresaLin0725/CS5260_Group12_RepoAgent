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
import os
from typing import Dict, List, Optional

from api.video.constants import VIDEO_SIZE

VIDEO_WIDTH, VIDEO_HEIGHT = VIDEO_SIZE


# ---------------------------------------------------------------------------
# Bundled emoji font (Noto Color Emoji extracted to .venv/system_libs/)
# Without this, Chromium falls back to "tofu" boxes for ⭐ 🥇 📄 etc. on WSL,
# where no system color-emoji font is installed. We reference the .ttf via
# a file:// URL inside @font-face so set_content() (which has no base URL)
# can still resolve it.
# ---------------------------------------------------------------------------

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
_EMOJI_FONT_PATH = os.path.join(
    _REPO_ROOT, ".venv", "system_libs", "usr", "share", "fonts",
    "truetype", "noto", "NotoColorEmoji.ttf",
)


def _font_face_css() -> str:
    """Return @font-face block for the bundled emoji font, or empty string
    if the font is not present (graceful fallback to system fonts).

    @font-face with a file:// URL alone proved unreliable in Chromium when
    pages are loaded via set_content() (no base URL). The primary delivery
    path is therefore _ensure_emoji_font_registered() below, which makes
    fontconfig aware of the .ttf. This @font-face block is kept as a
    belt-and-suspenders fallback for environments without fontconfig.
    """
    if not os.path.exists(_EMOJI_FONT_PATH):
        return ""
    url = "file://" + _EMOJI_FONT_PATH.replace(os.sep, "/")
    return f"""
    @font-face {{
        font-family: 'NotoColorEmoji';
        src: url('{url}') format('truetype');
        font-display: block;
    }}
    """


def _ensure_emoji_font_registered() -> None:
    """Idempotently expose the bundled Noto Color Emoji to fontconfig.

    Chromium discovers fonts via fontconfig on Linux. Symlinking the
    bundled .ttf into ~/.local/share/fonts and refreshing the fontconfig
    cache makes emoji available without sudo. This runs once at import
    time; subsequent imports skip the work because the symlink already
    exists.
    """
    if not os.path.exists(_EMOJI_FONT_PATH):
        return
    user_fonts_dir = os.path.join(
        os.path.expanduser("~"), ".local", "share", "fonts"
    )
    target = os.path.join(user_fonts_dir, "NotoColorEmoji.ttf")
    if os.path.exists(target) or os.path.islink(target):
        return
    try:
        os.makedirs(user_fonts_dir, exist_ok=True)
        os.symlink(_EMOJI_FONT_PATH, target)
    except OSError:
        # Symlinks may require privileges on Windows; the @font-face
        # fallback covers it. Silently skip — never block import.
        return
    try:
        import subprocess
        subprocess.run(
            ["fc-cache", "-f", user_fonts_dir],
            check=False, timeout=15,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, OSError):
        pass


_ensure_emoji_font_registered()


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
    {_font_face_css()}
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{
        width: {VIDEO_WIDTH}px; height: {VIDEO_HEIGHT}px;
        background: {p['bg']};
        font-family: 'Segoe UI', system-ui, -apple-system, 'Noto Sans SC',
                     'NotoColorEmoji', 'Apple Color Emoji', 'Segoe UI Emoji',
                     'Noto Color Emoji', sans-serif;
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
            summary = (m.get("summary") or "").strip()
            summary_html = (
                f'<div class="milestone-summary">{_esc(summary)}</div>'
                if summary else ''
            )
            nodes.append(f'''
                <div class="{classes}">
                    <div class="milestone-dot"></div>
                    <div class="milestone-date">{_esc(m.get("date", ""))}</div>
                    <div class="milestone-label">{_esc(m.get("label", ""))}</div>
                    {summary_html}
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
            medal = _esc(c.get("medal", "▫️"))
            avatar_url = c.get("avatar_url") or ""
            if avatar_url:
                # Both <img> and <div class="ctr-medal"> are emitted; the
                # img's onerror handler swaps to the medal if the GitHub CDN
                # blocks/fails (Chromium occasionally can't load the avatar).
                avatar_or_medal = (
                    f'<img class="ctr-avatar" src="{_esc(avatar_url)}" alt=""'
                    f' onerror="this.style.display=\'none\';'
                    f'this.nextElementSibling.style.display=\'flex\';"/>'
                    f'<div class="ctr-medal" style="display:none">{medal}</div>'
                )
            else:
                avatar_or_medal = f'<div class="ctr-medal">{medal}</div>'
            stats_parts = [f'\U0001f4dd {c.get("commits", 0)}']  # 📝 commits
            if c.get("followers"):
                stats_parts.append(f'\U0001f465 {c["followers"]:,}')  # 👥 followers
            stats_line = '  ·  '.join(stats_parts)
            cards.append(f'''
                <div class="ctr-card">
                    {avatar_or_medal}
                    <div class="ctr-text">
                        <div class="ctr-name">{_esc(c.get("name") or c.get("login", ""))}</div>
                        <div class="ctr-stats">{stats_line}</div>
                    </div>
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
        .milestone-summary {{
            font-size: 11px; color: {p['text_dim']}cc; line-height: 1.35;
            margin-top: 2px; word-break: break-word;
            display: -webkit-box;
            -webkit-line-clamp: 3;
            -webkit-box-orient: vertical;
            overflow: hidden;
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
            font-size: 13px; color: {p['text_dim']}; margin-top: 2px;
            white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
        }}
        .ctr-text {{ flex: 1; min-width: 0; }}
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
        narration=card.get("narration", ""),
    )


# ---------------------------------------------------------------------------
# Act 2 — Metaphor: comic-bullet dialogue (image gen comes in v2)
# ---------------------------------------------------------------------------

_SPEAKER_PATTERN = __import__("re").compile(r"^\s*([^:：]{1,30})\s*[:：]\s+(.+)$")


def _split_speaker(text: str):
    """If text looks like 'Speaker: ...', return (speaker, utterance);
    otherwise ('', text). Treats both ':' and full-width '：' as separators."""
    if not text:
        return "", ""
    m = _SPEAKER_PATTERN.match(text)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return "", text.strip()


def render_act2_metaphor_html(card: dict) -> str:
    """Render the Act 2 metaphor as a comic-bullet dialogue.

    card:
        {
            "segments": [{"detail": str, "brief": str}, ...],   # 3-6 entries
            "fallback_subject": str,   # used when segments is empty
        }

    Each segment becomes one speech bubble. If the brief begins with
    "Speaker: ..." the speaker is lifted out into a small uppercase label
    above the bubble (so the bubble itself only carries the utterance).
    Sizing/spacing scales with segment count so 3 segments fill the frame
    and 6 segments still fit without overflow.
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

    usable = [
        s for s in segments[:6]
        if (s.get("brief") or s.get("detail") or "").strip()
    ]
    n = max(1, len(usable))

    # Dynamic sizing: fewer segments → bigger text, taller padding.
    # 3 → 22px / gap 24, 4 → 19/20, 5 → 17/16, 6 → 15/12
    text_size = max(15, 28 - (n - 2) * 2)
    row_gap = max(10, 28 - (n - 2) * 4)
    bubble_pad_v = max(10, 18 - (n - 3) * 2)
    svg_size = max(48, 78 - (n - 3) * 6)

    # Build a global role → repo_concept lookup from every segment's
    # entities list. The same role can appear in multiple segments;
    # last write wins (LLM is supposed to be consistent — last-write
    # is just a defensive default).
    role_to_concept: dict = {}
    for s in usable:
        for ent in (s.get("entities") or []):
            role = (ent.get("role") if isinstance(ent, dict) else getattr(ent, "role", "")) or ""
            concept = (ent.get("repo_concept") if isinstance(ent, dict) else getattr(ent, "repo_concept", "")) or ""
            role = role.strip().lower()
            concept = concept.strip()
            if role and concept:
                role_to_concept[role] = concept

    bubble_html: list[str] = []
    for i, seg in enumerate(usable):
        side = "left" if i % 2 == 0 else "right"
        raw = (seg.get("brief") or seg.get("detail") or "").strip()
        speaker, utterance = _split_speaker(raw)
        svg_key = "person_thinking" if side == "left" else "person_happy"
        svg_html = _SVG_MAP.get(svg_key, SVG_PERSON_THINKING)

        # Look up repo_concept for this speaker so the audience can see
        # what the metaphor character represents in the codebase.
        concept = role_to_concept.get(speaker.lower(), "") if speaker else ""
        speaker_html = ""
        if speaker:
            mapping_html = (
                f'<span class="bubble-mapping">= {_esc(concept)}</span>'
                if concept else ''
            )
            speaker_html = (
                f'<div class="bubble-speaker">'
                f'<span class="bubble-role">{_esc(speaker)}</span>'
                f'{mapping_html}'
                f'</div>'
            )
        bubble_html.append(f'''
            <div class="bubble-row {side}">
                <div class="bubble-svg" style="color:{p['accent']};">{svg_html}</div>
                <div class="bubble-stack">
                    {speaker_html}
                    <div class="bubble-card">
                        <div class="bubble-text">{_esc(utterance)}</div>
                    </div>
                </div>
            </div>
        ''')

    extra_css = f"""
        .content {{
            padding: 28px 72px 28px; gap: {row_gap}px;
            justify-content: center; align-items: stretch;
        }}
        .bubble-row {{
            display: flex; align-items: center; gap: 18px;
            max-width: 100%;
        }}
        .bubble-row.left {{ justify-content: flex-start; }}
        .bubble-row.right {{ justify-content: flex-end; flex-direction: row-reverse; }}
        .bubble-svg {{
            flex-shrink: 0; width: {svg_size}px; height: {svg_size}px;
        }}
        .bubble-svg svg {{ width: 100%; height: 100%; }}
        .bubble-stack {{
            display: flex; flex-direction: column; min-width: 0;
            flex: 1; max-width: 880px; gap: 4px;
        }}
        .bubble-row.right .bubble-stack {{ align-items: flex-end; }}
        .bubble-speaker {{
            font-size: 12px; letter-spacing: 1.8px;
            color: {p['accent']}; font-weight: 700;
            text-transform: uppercase;
            padding: 0 4px;
            display: flex; align-items: baseline; gap: 8px;
            flex-wrap: wrap;
        }}
        .bubble-row.right .bubble-speaker {{ justify-content: flex-end; }}
        .bubble-role {{
            color: {p['accent']};
        }}
        .bubble-mapping {{
            font-size: 11px;
            letter-spacing: 0.3px;
            color: {p['text_dim']};
            font-weight: 500;
            text-transform: none;
        }}
        .bubble-card {{
            position: relative; align-self: stretch;
            background: {p['node_bg']}; border: 2px solid {p['node_border']};
            border-radius: 18px; padding: {bubble_pad_v}px 22px;
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
            font-size: {text_size}px; color: {p['text']}; line-height: 1.5;
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
        narration=card.get("narration", ""),
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

    role_titles = ["\U0001f4e5 INPUT", "⚙️ PROCESS", "\U0001f4e4 OUTPUT"]
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
        narration=card.get("narration", ""),
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

    # Dynamic font size based on the longest panel speech: short text gets
    # bigger letters to fill the frame; long text shrinks so it still fits
    # without truncation. CSS line-height + word-break handle wrapping.
    longest = max(
        (len((p_.get("speech") or "")) for p_ in panels),
        default=0,
    )
    if longest <= 70:
        bubble_font = 18
    elif longest <= 130:
        bubble_font = 16
    elif longest <= 200:
        bubble_font = 14
    else:
        bubble_font = 13

    extra_css = f"""
        .panels-row {{
            display: flex; align-items: stretch; justify-content: center;
            gap: 0; flex: 1; padding: 0 20px; min-height: 0;
        }}
        .comic-panel {{
            flex: 1; display: flex; flex-direction: column; align-items: center;
            justify-content: flex-end;          /* stick figure + label anchor at bottom */
            min-width: 0; max-width: 380px;
            text-align: center; padding: 0 8px;
            gap: 12px;
        }}
        .comic-bubble {{
            position: relative; background: {p['node_bg']}; border: 2px solid {p['node_border']};
            border-radius: 16px; padding: 14px 20px;
            font-size: {bubble_font}px; color: {p['text']}; line-height: 1.45;
            text-align: center; max-width: 100%;
            word-break: break-word; overflow-wrap: anywhere;
            /* bubble grows upward as text gets longer; flex-end on the
               panel keeps the stick figure pinned to the bottom. */
        }}
        .comic-bubble::after {{
            content: ''; position: absolute; bottom: -10px; left: 50%; transform: translateX(-50%);
            border-left: 9px solid transparent; border-right: 9px solid transparent;
            border-top: 11px solid {p['node_border']};
        }}
        .comic-svg {{ width: 88px; height: 88px; }}
        .comic-svg svg {{ width: 100%; height: 100%; }}
        .comic-label {{
            font-size: 17px; font-weight: 700; color: {p['accent']};
            letter-spacing: 0.5px;
        }}
        .comic-arrow {{
            flex-shrink: 0; width: 50px; padding: 0 2px;
            display: flex; align-items: center; justify-content: center;
            /* Anchor with the stick figure (which sits at the panel
               bottom because .comic-panel uses justify-content: flex-end).
               margin-bottom = label-height + svg-height/2 so the arrow
               points at the middle of the SVG, not floating mid-panel. */
            align-self: flex-end;
            margin-bottom: 56px;
        }}
        .comic-arrow svg {{ width: 100%; height: 24px; }}
    """

    return _wrap_page(
        title=card.get("title", "A typical scenario"),
        subtitle=card.get("subtitle", ""),
        body=(
            '<div class="content">'
            + scene_bar
            + '<div class="panels-row">'
            + ''.join(flow) + '</div>'
            + '</div>'
        ),
        palette=p,
        extra_css=extra_css,
        footer=card.get("footer", ""),
        narration=card.get("narration", ""),
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
        from api.video.onboard_5act.icons import guess_prereq_icon
        chips = "".join(
            f'<span class="prereq-chip">{guess_prereq_icon(pr)} {_esc(pr)}</span>'
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
        narration=card.get("narration", ""),
    )


# ---------------------------------------------------------------------------
# Page wrapper (header + body + footer + base css + extra css)
# ---------------------------------------------------------------------------

SUBTITLE_BAR_HEIGHT_PX = 96  # reserved space at bottom for the narration bar


def _subtitle_bar_css() -> str:
    """CSS injected when a narration string is present (subtitle overlay).

    The .content area gets padding-bottom so flex children don't overlap
    the bar; the bar itself is fixed to the page bottom with a translucent
    black background and crisp white text — same look as TV captions.
    """
    return f"""
    .content {{ padding-bottom: {SUBTITLE_BAR_HEIGHT_PX + 8}px !important; }}
    .footer {{ display: none; }}
    .subtitle-bar {{
        position: absolute; left: 0; right: 0; bottom: 0;
        min-height: {SUBTITLE_BAR_HEIGHT_PX}px;
        padding: 16px 64px 18px;
        background: rgba(0, 0, 0, 0.72);
        color: #ffffff;
        font-size: 22px; font-weight: 500;
        text-align: center; line-height: 1.45;
        word-break: break-word;
        display: flex; align-items: center; justify-content: center;
    }}
    .subtitle-bar .text {{
        max-width: 1100px;
        display: -webkit-box;
        -webkit-line-clamp: 3;
        -webkit-box-orient: vertical;
        overflow: hidden;
    }}
    """


def _wrap_page(
    *, title: str, subtitle: str, body: str, palette: Dict[str, str],
    extra_css: str = "", footer: str = "", narration: str = "",
) -> str:
    """Wrap an act body in the standard page chrome (header + footer + subtitle).

    When ``narration`` is provided, a TV-caption style bar is drawn across
    the bottom of the frame and content is shrunk to leave room for it.
    """
    narration = (narration or "").strip()
    subtitle_css = _subtitle_bar_css() if narration else ""
    subtitle_html = (
        f'<div class="subtitle-bar"><div class="text">{_esc(narration)}</div></div>'
        if narration else ''
    )
    return (
        '<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">'
        + f'<meta name="viewport" content="width={VIDEO_WIDTH}">'
        + '<style>' + _base_css(palette) + subtitle_css + extra_css + '</style></head><body>'
        + '<div class="header">'
        + (f'<div class="subtitle">{_esc(subtitle)}</div>' if subtitle else '')
        + f'<div class="title">{_esc(title)}</div>'
        + '</div>'
        + body
        + (f'<div class="footer">{_esc(footer)}</div>' if footer else '')
        + subtitle_html
        + '</body></html>'
    )
