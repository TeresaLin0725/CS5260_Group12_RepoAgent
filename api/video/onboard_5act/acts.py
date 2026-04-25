"""
Build act-by-act scene specs from AnalyzedContent.

Each act is a dict with at least:
    {
        "act_number": 1..5,
        "section": "intro" | "metaphor" | "io" | "usecase" | "setup",
        "title": str,
        "narration": str,            # what TTS will say
        "duration_seconds": float,   # target visual duration (TTS may stretch)
        "card": dict,                # template-specific data the renderer reads
    }

The card payload is intentionally a free-form dict — each act has its own
renderer in templates.py / scene_renderer.py and reads only the fields
it cares about. This mirrors the baseline pipeline's card pattern
(see api/video/card_builder.py for the legacy reference) but is a fresh
copy that only carries what the 5-act renderer needs.

This module never raises: missing data falls back to placeholder text so
the resulting video always has all 5 acts (consistent length / pacing).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Optional

from api.video.onboard_5act import icons

if TYPE_CHECKING:
    from api.content_analyzer import AnalyzedContent

logger = logging.getLogger(__name__)

# Default per-act target durations (seconds). Final video ~ 35-45s total.
DEFAULT_DURATIONS = {
    "intro": 8.0,
    "metaphor": 12.0,   # 3-5 sub-frames
    "io": 6.0,
    "usecase": 9.0,     # 3 sub-frames
    "setup": 7.0,
}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_acts(analyzed: "AnalyzedContent") -> List[dict]:
    """Build all 5 acts from analyzed content.

    Always returns exactly 5 dicts in fixed order (intro → metaphor →
    io → usecase → setup). Empty / fallback acts still render so the
    video pacing stays consistent.
    """
    return [
        _build_act1_intro(analyzed),
        _build_act2_metaphor(analyzed),
        _build_act3_io(analyzed),
        _build_act4_usecase(analyzed),
        _build_act5_setup(analyzed),
    ]


# ---------------------------------------------------------------------------
# Per-act builders. Bodies are stubs in this commit — Commits 3 and 4 fill them.
# Each builder is independent: tests can call any one in isolation.
# ---------------------------------------------------------------------------

def _build_act1_intro(analyzed: "AnalyzedContent") -> dict:
    """Act 1 — project name, one-liner, timeline, badges, contributors."""
    onboard = getattr(analyzed, "onboard", None)
    one_liner = (onboard.one_liner if onboard and onboard.one_liner else
                 (analyzed.project_overview or "")[:100].strip())
    return {
        "act_number": 1,
        "section": "intro",
        "title": analyzed.repo_name or "Repository",
        "narration": one_liner or f"Welcome to {analyzed.repo_name or 'this project'}.",
        "duration_seconds": DEFAULT_DURATIONS["intro"],
        "card": {
            "repo_name": analyzed.repo_name or "Repository",
            "one_liner": one_liner,
            "timeline": _extract_timeline_milestones(analyzed),
            "stats": _extract_stats_dict(analyzed),
            "headline_contributors": _pick_headline_contributors(analyzed),
        },
    }


def _build_act2_metaphor(analyzed: "AnalyzedContent") -> dict:
    """Act 2 — everyday analogy story (kitchen, courier, ...) in 3-5 segments."""
    segments = list(getattr(analyzed, "metaphor_story", []) or [])
    return {
        "act_number": 2,
        "section": "metaphor",
        "title": "Think of it like this…",
        "narration": _metaphor_narration(segments, analyzed),
        "duration_seconds": DEFAULT_DURATIONS["metaphor"],
        "card": {
            # `segments` is a list of MetaphorSegment objects (or empty for fallback).
            "segments": [{"detail": s.detail, "brief": s.brief} for s in segments],
            "fallback_subject": analyzed.repo_name or "this project",
        },
    }


def _build_act3_io(analyzed: "AnalyzedContent") -> dict:
    """Act 3 — input → process → output 3-box diagram with icons."""
    onboard = getattr(analyzed, "onboard", None)
    boxes = list((onboard.mental_model_3_boxes if onboard else []) or [])
    if len(boxes) < 3:
        boxes = (boxes + ["Input", "AI processes", "Output"])[:3]
    boxes = boxes[:3]

    box_with_icons = [
        {"label": boxes[0], "icon": icons.guess_input_icon(boxes[0])},
        {"label": boxes[1], "icon": icons.guess_process_icon(boxes[1])},
        {"label": boxes[2], "icon": icons.guess_output_icon(boxes[2])},
    ]
    return {
        "act_number": 3,
        "section": "io",
        "title": "How it works",
        "narration": (
            f"You give it {boxes[0].lower()}; it {boxes[1].lower()}; "
            f"and you get {boxes[2].lower()}."
        ),
        "duration_seconds": DEFAULT_DURATIONS["io"],
        "card": {
            "boxes": box_with_icons,
        },
    }


def _build_act4_usecase(analyzed: "AnalyzedContent") -> dict:
    """Act 4 — 3-panel comic: problem speaker, user speaker, value speaker."""
    return {
        "act_number": 4,
        "section": "usecase",
        "title": "A typical scenario",
        "narration": _usecase_narration(analyzed),
        "duration_seconds": DEFAULT_DURATIONS["usecase"],
        "card": {
            "scene_context": _usecase_scene_context(analyzed),
            "panels": _usecase_panels(analyzed),
        },
    }


def _build_act5_setup(analyzed: "AnalyzedContent") -> dict:
    """Act 5 — prerequisites + 5-minute setup checklist with step icons."""
    onboard = getattr(analyzed, "onboard", None)
    prereqs = list((onboard.prerequisites if onboard else []) or [])
    raw_steps = (onboard.first_5_minutes if onboard else "") or ""
    steps = _parse_setup_steps(raw_steps)
    return {
        "act_number": 5,
        "section": "setup",
        "title": "Get it running in 5 minutes",
        "narration": (
            f"Make sure you have {', '.join(prereqs[:2])}. " if prereqs
            else ""
        ) + "Then run a few simple commands and open the result in your browser.",
        "duration_seconds": DEFAULT_DURATIONS["setup"],
        "card": {
            "prerequisites": prereqs[:3],
            "steps": [
                {"text": s, "icon": icons.guess_setup_step_icon(s)}
                for s in steps[:5]
            ] or [
                {"text": "git clone <url>", "icon": icons.guess_setup_step_icon("clone")},
                {"text": "Install dependencies", "icon": icons.guess_setup_step_icon("install")},
                {"text": "Run the project", "icon": icons.guess_setup_step_icon("run")},
                {"text": "Open in browser", "icon": icons.guess_setup_step_icon("open")},
            ],
        },
    }


# ---------------------------------------------------------------------------
# Helpers — kept tiny so each is easy to unit-test
# ---------------------------------------------------------------------------

def _extract_timeline_milestones(analyzed: "AnalyzedContent") -> List[dict]:
    """Pick 3-5 key moments for Act 1 timeline ribbon.

    Heuristic: first commit, latest release (if any), latest commit.
    Templates flesh this out further in Commit 4.
    """
    timeline = getattr(analyzed, "commit_timeline", None)
    if not timeline:
        return []
    milestones: list[dict] = []
    if timeline.first_commit_date:
        milestones.append({
            "date": timeline.first_commit_date[:10],
            "label": "First commit",
        })
    for r in (timeline.releases or [])[:2]:
        if r.tag and r.date:
            milestones.append({"date": r.date[:10], "label": f"Release {r.tag}"})
    if timeline.latest_commit_date:
        milestones.append({
            "date": timeline.latest_commit_date[:10],
            "label": "Latest activity",
        })
    return milestones


def _extract_stats_dict(analyzed: "AnalyzedContent") -> Optional[dict]:
    timeline = getattr(analyzed, "commit_timeline", None)
    stats = getattr(timeline, "stats", None) if timeline else None
    if not stats:
        return None
    return {
        "stars": stats.stars,
        "forks": stats.forks,
        "watchers": stats.watchers,
        "license": stats.license,
        "pushed_at": stats.pushed_at[:10] if stats.pushed_at else "",
    }


def _pick_headline_contributors(analyzed: "AnalyzedContent") -> List[dict]:
    """Pick 1-3 'headline' contributors for the Act 1 footer.

    Score = followers * 1 + public_repos * 0.5 + commits-to-this-repo * 0.2.
    Falls back to top-by-commits if reputation data is missing.
    """
    timeline = getattr(analyzed, "commit_timeline", None)
    contribs = list(getattr(timeline, "contributors", None) or [])
    if not contribs:
        return []

    def score(c) -> float:
        return c.followers * 1.0 + c.public_repos * 0.5 + c.commit_count * 0.2

    ranked = sorted(contribs, key=score, reverse=True)[:3]
    out = []
    for i, c in enumerate(ranked, start=1):
        out.append({
            "rank": i,
            "login": c.login,
            "name": c.name or c.login,
            "avatar_url": c.avatar_url or "",
            "commits": c.commit_count,
            "followers": c.followers,
            "medal": icons.rank_medal(i),
        })
    return out


def _metaphor_narration(segments, analyzed) -> str:
    if segments:
        return " ".join(s.brief for s in segments[:4] if s.brief).strip() or \
               f"Here's a story about {analyzed.repo_name or 'this project'}."
    return (
        f"Imagine {analyzed.repo_name or 'this project'} as something familiar "
        "from everyday life — let's walk through how it works."
    )


def _usecase_narration(analyzed) -> str:
    target = (analyzed.target_users or "").strip()
    if target:
        return target[:200]
    return (
        f"Here's how someone might actually use {analyzed.repo_name or 'this project'} "
        "in real life."
    )


def _usecase_scene_context(analyzed) -> str:
    """One-line setting for the comic bar."""
    target = (analyzed.target_users or "").strip()
    if target:
        # Take first sentence
        for sep in [". ", "。", "; "]:
            if sep in target:
                return target.split(sep)[0].strip()[:80]
        return target[:80]
    return f"A typical day with {analyzed.repo_name or 'this project'}"


def _usecase_panels(analyzed) -> List[dict]:
    """Three speech-bubble panels: problem / use / value.

    Speech is derived from analyzed content when possible, falling back
    to project-name templated text. The three roles always exist (one
    per panel) so the comic layout is consistent.
    """
    name = analyzed.repo_name or "the project"
    onboard = getattr(analyzed, "onboard", None)

    # Problem speech: prefer the "audience pain" framing if onboard is
    # available, else generic.
    audience = (onboard.audience if onboard and onboard.audience else "").strip()
    problem_speech = _derive_problem_speech(audience, analyzed)

    # Use speech: short instruction grounded in concrete_io.
    concrete_io = (onboard.concrete_io if onboard and onboard.concrete_io else "").strip()
    use_speech = _derive_use_speech(concrete_io, name)

    # Value speech: outcome from concrete_io's "you get Y" half.
    value_speech = _derive_value_speech(concrete_io, name)

    return [
        {
            "role": "problem",
            "label": "Problem",
            "speech": problem_speech,
            "svg": "person_thinking",
        },
        {
            "role": "use",
            "label": "Use",
            "speech": use_speech,
            "svg": "person_at_desk",
        },
        {
            "role": "value",
            "label": "Value",
            "speech": value_speech,
            "svg": "person_happy",
        },
    ]


def _derive_problem_speech(audience: str, analyzed) -> str:
    """Build a relatable pain-point speech bubble (≤90 chars)."""
    # If audience text mentions a verb phrase like "want to ...", lift it.
    if audience:
        # Pick the first sentence that hints at a need.
        sentences = audience.replace("。", ".").split(".")
        for s in sentences:
            s = s.strip()
            if any(kw in s.lower() for kw in ["want", "need", "trying to", "looking for", "想", "需要"]):
                return _clip(f"I {s.lower().split('want')[-1].split('need')[-1].strip(', ')}…", 110) \
                    if "want" in s.lower() or "need" in s.lower() else _clip(s, 110)
    return "I keep running into the same headache — there has to be a faster way."


def _derive_use_speech(concrete_io: str, name: str) -> str:
    """Build a short 'how I used it' speech."""
    # Look for "you give X" / "paste X" / "input X" patterns.
    if concrete_io:
        low = concrete_io.lower()
        for trigger in ["paste", "give it", "input", "upload", "send"]:
            if trigger in low:
                idx = low.find(trigger)
                snippet = concrete_io[idx:idx + 90].strip()
                return _clip(f"I just {snippet[0].lower()}{snippet[1:]}", 110)
    return f"I just opened {name} and pointed it at the thing I had."


def _derive_value_speech(concrete_io: str, name: str) -> str:
    """Build the 'and now…' outcome speech."""
    if concrete_io:
        low = concrete_io.lower()
        # Find "you get Y" half.
        for trigger in ["you get", "i get", "it gives", "returns", "produces"]:
            if trigger in low:
                idx = low.find(trigger)
                snippet = concrete_io[idx:idx + 100].strip(". ")
                return _clip(f"…and now {snippet[0].lower()}{snippet[1:]}.", 120)
    return "Now I get exactly what I need — in one step."


def _clip(text: str, max_chars: int) -> str:
    text = " ".join(text.split())
    if len(text) > max_chars:
        text = text[:max_chars - 1].rstrip(",.;:") + "…"
    return text


def _parse_setup_steps(raw: str) -> List[str]:
    """Turn a free-form `first_5_minutes` string into bullet-style steps.

    Splits on numbered list markers (1. / 2. / 3.) or newlines. Trims
    whitespace and keeps non-empty entries.
    """
    if not raw:
        return []
    import re as _re
    # Split on "1. ", "2. ", etc., or on newline
    parts = _re.split(r"(?:^|\s)\d+[.)]\s+|\n+", raw)
    return [p.strip() for p in parts if p.strip()]
