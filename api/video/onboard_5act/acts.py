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

    Each act dict gets a ``narration_lines`` field (list of subtitle
    chunks split from ``narration``) so the orchestrator can render one
    PNG per chunk and switch the on-screen subtitle in time with the
    spoken TTS — proper TV-caption behavior.
    """
    acts = [
        _build_act1_intro(analyzed),
        _build_act2_metaphor(analyzed),
        _build_act3_io(analyzed),
        _build_act4_usecase(analyzed),
        _build_act5_setup(analyzed),
    ]
    for act in acts:
        act["narration_lines"] = _split_narration_into_lines(
            act.get("narration", "")
        )
    return acts


_SENTENCE_BOUNDARY = __import__("re").compile(r"(?<=[\.!?。!?])\s+")


def _split_narration_into_lines(narration: str, *, min_chars: int = 24) -> List[str]:
    """Split a TTS narration into subtitle-friendly chunks.

    Splits on sentence terminators ('.', '!', '?', '。', '！', '？') and
    merges any chunk shorter than ``min_chars`` with its neighbour so
    that no caption flashes by too quickly. Returns ``[narration]``
    unchanged if it's already short enough that splitting wouldn't help.
    """
    text = (narration or "").strip()
    if not text:
        return []
    parts = [p.strip() for p in _SENTENCE_BOUNDARY.split(text) if p and p.strip()]
    if len(parts) <= 1:
        return [text]

    merged: List[str] = []
    buffer = ""
    for part in parts:
        if buffer:
            buffer = (buffer + " " + part).strip()
            if len(buffer) >= min_chars:
                merged.append(buffer)
                buffer = ""
        else:
            if len(part) >= min_chars:
                merged.append(part)
            else:
                buffer = part
    if buffer:
        if merged:
            merged[-1] = (merged[-1] + " " + buffer).strip()
        else:
            merged.append(buffer)
    return merged or [text]


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
        "narration": _act1_narration(analyzed, one_liner),
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
            "segments": [
                {
                    "detail": s.detail,
                    "brief": s.brief,
                    "entities": [
                        {"role": e.role, "repo_concept": e.repo_concept}
                        for e in (s.entities or [])
                    ],
                }
                for s in segments
            ],
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
        "narration": _act3_narration(analyzed, boxes),
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
        "narration": _act5_narration(analyzed, prereqs[:2]),
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

    Heuristic: first commit, top releases (with summary), latest commit.
    Each milestone may carry an optional ``summary`` field — for releases,
    this is "what shipped between this and the previous release", filled
    by api.git_metadata.fill_release_summaries (steps C+A) or by the LLM
    fallback in content_analyzer (step B).
    """
    timeline = getattr(analyzed, "commit_timeline", None)
    if not timeline:
        return []
    milestones: list[dict] = []
    if timeline.first_commit_date:
        milestones.append({
            "date": timeline.first_commit_date[:10],
            "label": "First commit",
            "summary": "",
        })
    for r in (timeline.releases or [])[:2]:
        if r.tag and r.date:
            milestones.append({
                "date": r.date[:10],
                "label": f"Release {r.tag}",
                "summary": (r.summary or "").strip(),
            })
    if timeline.latest_commit_date:
        milestones.append({
            "date": timeline.latest_commit_date[:10],
            "label": "Latest activity",
            "summary": "",
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


# ---------------------------------------------------------------------------
# Per-act narration with scene-set openers + transition closers.
# Every helper supports English (default) and Chinese (analyzed.language="zh").
# Goal: each act sounds like part of a continuous video, not isolated cards —
# the closer of one act hints at the next, so the viewer stays engaged.
# ---------------------------------------------------------------------------

def _is_zh(analyzed) -> bool:
    return (getattr(analyzed, "language", "") or "").lower().startswith("zh")


def _join_en(parts: List[str]) -> str:
    return " ".join(p.strip() for p in parts if p and p.strip())


def _join_zh(parts: List[str]) -> str:
    return "".join(p.strip() for p in parts if p and p.strip())


def _connect_oneliner(one_liner: str, *, is_zh: bool) -> str:
    """Wrap the one-liner with a smooth connector so the narration doesn't
    read as two abrupt sentences. "Welcome to typer. Transforms X..." sounds
    robotic — "Welcome to typer. In a nutshell, it transforms X..." flows.
    """
    text = (one_liner or "").strip().rstrip(".。")
    if not text:
        return ""
    # Lowercase first letter so "It X" reads naturally regardless of the
    # one-liner's original capitalization.
    if text[0].isupper():
        text = text[0].lower() + text[1:]
    if is_zh:
        return f"简单来说，它{text}。"
    return f"In a nutshell, it {text}."


def _act1_narration(analyzed, one_liner: str) -> str:
    """Welcome → one-liner → milestones (optional) → stats (optional) → closer."""
    is_zh = _is_zh(analyzed)
    repo = analyzed.repo_name or ("这个项目" if is_zh else "this project")

    # — Milestones sentence (only when we have at least a first commit) —
    timeline = getattr(analyzed, "commit_timeline", None)
    first_date = (timeline.first_commit_date[:10] if timeline and timeline.first_commit_date else "")
    latest_release = None
    if timeline and timeline.releases:
        latest_release = timeline.releases[0]  # releases are sorted newest-first

    milestone_sentence = ""
    if first_date and latest_release and latest_release.tag and latest_release.date:
        rdate = latest_release.date[:10]
        if is_zh:
            milestone_sentence = f"它从 {first_date} 起步，最新版本 {latest_release.tag} 发布于 {rdate}。"
        else:
            milestone_sentence = (
                f"It's been active since {first_date}, "
                f"with {latest_release.tag} the most recent release ({rdate})."
            )
    elif first_date:
        milestone_sentence = (
            f"自 {first_date} 起一直在迭代。" if is_zh
            else f"Active since {first_date}."
        )

    # — Stats sentence (stars + headline contributor) —
    stats_sentence = ""
    stats = (timeline.stats if timeline else None)
    headliners = _pick_headline_contributors(analyzed)
    top = headliners[0] if headliners else None
    if stats and stats.stars and stats.stars >= 50:
        if is_zh:
            stars_part = f"在 GitHub 上有 {stats.stars:,} 颗星"
        else:
            stars_part = f"It has {stats.stars:,} stars on GitHub"
        if top and top.get("followers", 0) >= 500:
            name = top.get("name") or top.get("login") or ""
            if is_zh:
                stats_sentence = f"{stars_part}，由 {name} 主导。"
            else:
                stats_sentence = f"{stars_part}, led by {name}."
        else:
            stats_sentence = stars_part + ("。" if is_zh else ".")

    # — Closer hooks Act 2 —
    closer = "我们用一个比喻来理解它。" if is_zh else "Let's see what makes it tick."

    one_liner_sentence = _connect_oneliner(one_liner, is_zh=is_zh)

    if is_zh:
        opener = f"欢迎了解 {repo}。"
        return _join_zh([opener, one_liner_sentence, milestone_sentence, stats_sentence, closer])
    opener = f"Welcome to {repo}."
    return _join_en([opener, one_liner_sentence, milestone_sentence, stats_sentence, closer])


def _metaphor_narration(segments, analyzed) -> str:
    """Scene-setting opener → speaker-tagged dialogue with repo-concept
    mapping → transition closer.

    The first time a speaker appears we expand them with their entity
    mapping ("Chef, the typer library, says: ...") so the TTS audience
    learns who's who. Subsequent appearances drop the mapping to keep
    the dialogue flowing.
    """
    is_zh = _is_zh(analyzed)
    repo = analyzed.repo_name or ("这个项目" if is_zh else "this project")

    if not segments:
        if is_zh:
            return _join_zh([
                "想象一下这样的场景：",
                f"{repo} 就像是日常生活里某个熟悉的角色。",
                "接下来看看它内部到底是怎么运作的。",
            ])
        return _join_en([
            "Picture this for a moment.",
            f"{repo} works a lot like something you already know from everyday life.",
            "Now here's what's really going on inside.",
        ])

    # Aggregate role → repo_concept across all segments.
    role_to_concept: dict = {}
    for s in segments:
        for ent in (getattr(s, "entities", None) or []):
            role = (getattr(ent, "role", "") or "").strip().lower()
            concept = (getattr(ent, "repo_concept", "") or "").strip()
            if role and concept and role not in role_to_concept:
                role_to_concept[role] = concept

    # Lift speakers out of "Speaker: utterance" briefs so the TTS narrates
    # naturally ("Diner, the developer, says: ...") instead of reading the
    # raw "Diner:".
    import re as _re
    speaker_re = _re.compile(r"^\s*([^:：]{1,30})\s*[:：]\s+(.+)$")

    introduced: set = set()
    lines: List[str] = []
    for seg in segments[:4]:
        text = (getattr(seg, "brief", "") or getattr(seg, "detail", "") or "").strip()
        if not text:
            continue
        m = speaker_re.match(text)
        if not m:
            lines.append(text)
            continue
        speaker, utterance = m.group(1).strip(), m.group(2).strip()
        speaker_key = speaker.lower()
        concept = role_to_concept.get(speaker_key, "")
        # First appearance of this speaker gets the "= concept" expansion;
        # later appearances stay tight so the dialogue doesn't drag.
        first_time = speaker_key not in introduced
        introduced.add(speaker_key)
        if is_zh:
            if concept and first_time:
                lines.append(f"{speaker}（也就是{concept}）说：{utterance}")
            else:
                lines.append(f"{speaker} 说：{utterance}")
        else:
            if concept and first_time:
                lines.append(f"The {speaker.lower()}, which is {concept}, says: {utterance}")
            else:
                lines.append(f"The {speaker.lower()} says: {utterance}")

    body = ("。".join(lines) + "。") if is_zh else (". ".join(lines) + ".")
    body = body.replace("。。", "。").replace("..", ".")

    if is_zh:
        opener = "想象一下这样的场景。"
        closer = "这就是它的精髓。接下来看看它内部到底是怎么运作的。"
        return _join_zh([opener, body, closer])
    opener = "Picture this for a moment."
    closer = "That's the gist of it. Now here's what's really going on inside."
    return _join_en([opener, body, closer])


def _act3_narration(analyzed, boxes: List[str]) -> str:
    """Setup line → 3-box flow → hook into Act 4 use-case."""
    is_zh = _is_zh(analyzed)
    if is_zh:
        return _join_zh([
            "用最简单的话来描述：",
            f"你给它 {boxes[0]}，它会 {boxes[1]}，最后给你 {boxes[2]}。",
            "那真实场景里，谁会用它？我们看一个例子。",
        ])
    return _join_en([
        "Here's the simple version.",
        f"You give it {boxes[0].lower()}; it {boxes[1].lower()}; and you get {boxes[2].lower()}.",
        "But what's it actually for? Let's look at a real scenario.",
    ])


def _usecase_narration(analyzed) -> str:
    """Audience-anchored opener → existing target_users body → hook into Act 5."""
    is_zh = _is_zh(analyzed)
    target = (analyzed.target_users or "").strip()

    body = (target[:240] if target else "")
    if not body:
        repo = analyzed.repo_name or ("这个项目" if is_zh else "this project")
        body = (f"看看普通用户会怎么用 {repo}。" if is_zh
                else f"Here's how someone might actually use {repo} in real life.")

    if is_zh:
        opener = "想象一下你正面临这样的需求。"
        closer = "听起来还不错？只要五分钟就能跑起来。"
        return _join_zh([opener, body, closer])
    opener = "Imagine you've got a problem like this."
    closer = "Sounds useful? You can be running it in five minutes."
    return _join_en([opener, body, closer])


def _act5_narration(analyzed, prereqs: List[str]) -> str:
    """Action opener → prereqs (optional) → quick-start body → wrap-up closer."""
    is_zh = _is_zh(analyzed)
    if is_zh:
        opener = "现在轮到你试一试。"
        prereq_part = (f"先确认你装好了 {'、'.join(prereqs)}。" if prereqs else "")
        body = "然后跑几条简单的命令，在浏览器里打开就好。"
        closer = "搞定了——你已经成功跑起来了！"
        return _join_zh([opener, prereq_part, body, closer])
    opener = "Time to try it yourself."
    prereq_part = (f"Make sure you have {', '.join(prereqs)}." if prereqs else "")
    body = "Then a few quick commands and you can open it in your browser."
    closer = "And that's it — you're up and running. Happy hacking!"
    return _join_en([opener, prereq_part, body, closer])


def _usecase_scene_context(analyzed) -> str:
    """One-line setting for the comic bar.

    Cap is generous (160 chars); the .scene-bar CSS wraps to two lines
    cleanly when needed, which beats a mid-word truncation.
    """
    target = (analyzed.target_users or "").strip()
    if target:
        for sep in [". ", "。", "; "]:
            if sep in target:
                return target.split(sep)[0].strip()[:160]
        return target[:160]
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


_PROBLEM_CAP = 200
_USE_CAP = 220
_VALUE_CAP = 220


def _derive_problem_speech(audience: str, analyzed) -> str:
    """Build a relatable pain-point speech bubble.

    The renderer's Act 4 layout shrinks font dynamically and wraps
    multi-line, so the cap here exists only as a safety net — set well
    above the layout's preferred density.
    """
    if audience:
        sentences = audience.replace("。", ".").split(".")
        for s in sentences:
            s = s.strip()
            if any(kw in s.lower() for kw in ["want", "need", "trying to", "looking for", "想", "需要"]):
                return _clip(f"I {s.lower().split('want')[-1].split('need')[-1].strip(', ')}…", _PROBLEM_CAP) \
                    if "want" in s.lower() or "need" in s.lower() else _clip(s, _PROBLEM_CAP)
    return "I keep running into the same headache — there has to be a faster way."


def _derive_use_speech(concrete_io: str, name: str) -> str:
    """Build a short 'how I used it' speech."""
    if concrete_io:
        low = concrete_io.lower()
        for trigger in ["paste", "give it", "input", "upload", "send"]:
            if trigger in low:
                idx = low.find(trigger)
                snippet = concrete_io[idx:idx + 200].strip()
                return _clip(f"I just {snippet[0].lower()}{snippet[1:]}", _USE_CAP)
    return f"I just opened {name} and pointed it at the thing I had."


def _derive_value_speech(concrete_io: str, name: str) -> str:
    """Build the 'and now…' outcome speech."""
    if concrete_io:
        low = concrete_io.lower()
        for trigger in ["you get", "i get", "it gives", "returns", "produces"]:
            if trigger in low:
                idx = low.find(trigger)
                snippet = concrete_io[idx:idx + 200].strip(". ")
                return _clip(f"…and now {snippet[0].lower()}{snippet[1:]}.", _VALUE_CAP)
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
