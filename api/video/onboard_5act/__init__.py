"""
Onboard 5-Act Video Pipeline.

A new video pipeline aimed at brand-new programmers who need to decide
"should I open this repo?" in 30 seconds. Generates a fixed-structure
5-act narrative:

    Act 1 — Intro: project name + one-liner + commit timeline + ⭐/🍴
    Act 2 — Metaphor: an everyday analogy (kitchen, courier, ...) in
            3-5 comic-style segments
    Act 3 — I/O Diagram: input → process → output, with icons
    Act 4 — Use-case Comic: 3-panel "pain → use → value" stick-figure
    Act 5 — Setup: prereqs + 5-minute getting-started commands

Selected via env var ``VIDEO_PIPELINE=onboard_5act``. The legacy baseline
pipeline at ``api/video/orchestrator.py`` is intentionally untouched —
this subpackage **copies** rendering helpers from
``api/scene_renderer.py`` and ``api/video/card_builder.py`` instead of
importing them, so future iteration here cannot accidentally affect
the baseline.

Public entry point:
    render_onboard_5act_video(analyzed, ...) -> bytes
"""

from api.video.onboard_5act.orchestrator import render_onboard_5act_video

__all__ = ["render_onboard_5act_video"]
