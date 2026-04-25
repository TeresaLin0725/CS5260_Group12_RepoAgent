"""
Render-act dispatcher: maps an act spec dict to its HTML template, then
runs the result through Playwright to produce a PNG file.

Re-uses the legacy ``api.scene_renderer`` Playwright thread executor so
the whole process ends up with exactly one Chromium instance no matter
which pipeline (legacy vs onboard_5act) is in use. The HTML templates
themselves are local to this subpackage (see ``scene_renderer.py``).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Callable, Dict

from api.video.onboard_5act import scene_renderer as _r

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Section → HTML builder
# ---------------------------------------------------------------------------

# Acts 1 and 2 land in commit 4. For now they fall back to the I/O
# template so the integration test in this commit can still render
# the full 5-frame sequence end-to-end (just with placeholder visuals).

def _placeholder_act1_intro_html(card: dict) -> str:
    return _r.render_act3_io_html({
        "title": card.get("title", ""),
        "subtitle": "Act 1 — Intro (template lands in commit 4)",
        "boxes": [
            {"label": card.get("repo_name", "Repo"), "icon": "📦"},
            {"label": card.get("one_liner", "")[:40] or "What it does", "icon": "💡"},
            {"label": "Coming up next…", "icon": "▶️"},
        ],
        "footer": "",
    })


def _placeholder_act2_metaphor_html(card: dict) -> str:
    segs = card.get("segments") or []
    if segs:
        boxes = [
            {"label": (s.get("brief") or "")[:40] or "…", "icon": "💭"}
            for s in segs[:3]
        ]
    else:
        boxes = [
            {"label": "Imagine…", "icon": "💭"},
            {"label": card.get("fallback_subject", "the project"), "icon": "🎯"},
            {"label": "…makes sense now", "icon": "✨"},
        ]
    while len(boxes) < 3:
        boxes.append({"label": "…", "icon": "💭"})
    return _r.render_act3_io_html({
        "title": "Think of it like this…",
        "subtitle": "Act 2 — Metaphor (template lands in commit 4)",
        "boxes": boxes[:3],
        "footer": "",
    })


_TEMPLATE_DISPATCH: Dict[str, Callable[[dict], str]] = {
    "intro": _placeholder_act1_intro_html,
    "metaphor": _placeholder_act2_metaphor_html,
    "io": _r.render_act3_io_html,
    "usecase": _r.render_act4_usecase_html,
    "setup": _r.render_act5_setup_html,
}


def build_act_html(act: dict) -> str:
    """Build full HTML for one act spec dict (from acts.build_acts())."""
    section = act.get("section", "intro")
    builder = _TEMPLATE_DISPATCH.get(section, _r.render_act3_io_html)
    card = dict(act.get("card") or {})
    # Pass top-level title/subtitle/footer through unless the card overrides.
    card.setdefault("title", act.get("title", ""))
    return builder(card)


# ---------------------------------------------------------------------------
# Playwright entry point
# ---------------------------------------------------------------------------

async def render_act_to_png(act: dict, output_path: str) -> None:
    """Render a single act spec to a PNG file via Playwright.

    Reuses the legacy thread executor so we don't double-launch Chromium.
    """
    html_content = build_act_html(act)

    # Import inside the function so the legacy module's playwright import
    # only happens when we actually try to render (keeps unit tests fast).
    from api.scene_renderer import _executor, _sync_render

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(_executor, _sync_render, html_content, output_path)
    logger.debug("Rendered onboard_5act PNG: %s", output_path)
