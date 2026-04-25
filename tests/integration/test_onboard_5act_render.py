#!/usr/bin/env python3
"""
Integration tests for the 5-act onboard video renderer.

Renders each act to a real PNG via Playwright. Requires a working
Chromium install (pip install playwright + playwright install chromium).

Run with `pytest -s` to see the output paths printed for manual visual
review:
    pytest tests/integration/test_onboard_5act_render.py -s

The PNGs land in /tmp/onboard_5act_preview/ (or whatever
ONBOARD_5ACT_PREVIEW_DIR points to). Each test assertion just checks the
file exists and has plausible size; real visual quality verification is
human-in-the-loop.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def _playwright_chromium_works() -> bool:
    """Return True only if Playwright can actually launch Chromium on this host.

    Chromium needs system libs like libnspr4/libnss3 that may be missing on
    minimal WSL setups. Without this check, the test failure looks like
    'sync API in asyncio loop' (a misleading Playwright error). Skipping
    cleanly is friendlier than a confusing red trace.

    To enable end-to-end rendering on Linux/WSL:
        sudo apt-get install -y libnspr4 libnss3
        # or:
        playwright install-deps chromium
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return False
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            browser.close()
        return True
    except Exception:
        return False


_CHROMIUM_OK = _playwright_chromium_works()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def preview_dir() -> Path:
    """Where to drop the rendered PNGs for manual inspection."""
    custom = os.environ.get("ONBOARD_5ACT_PREVIEW_DIR")
    base = Path(custom) if custom else Path(tempfile.gettempdir()) / "onboard_5act_preview"
    base.mkdir(parents=True, exist_ok=True)
    return base


@pytest.fixture(scope="module")
def sample_analyzed():
    """A reasonably rich AnalyzedContent for visual review."""
    from api.content_analyzer import AnalyzedContent, OnboardSnapshot, MetaphorSegment
    from api.git_metadata import (
        CommitTimeline, CommitTimelineEntry, ContributorInfo, ReleaseInfo, RepoStats,
    )

    a = AnalyzedContent(
        repo_name="seakon/repohelper",
        repo_url="https://github.com/seakon/repohelper",
        language="en",
        repo_type_hint="webapp",
        project_overview=(
            "RepoHelper turns any GitHub repository into an easy-to-read project guide. "
            "It analyzes the code, generates plain-language summaries, and produces "
            "PDF reports, slide decks, explainer videos, and infographic posters."
        ),
        target_users=(
            "Beginner programmers who just discovered an interesting open-source repo "
            "and want to understand it in 5 minutes instead of 2 hours."
        ),
    )
    a.onboard = OnboardSnapshot(
        one_liner="Turns a GitHub link into an easy-to-read project guide.",
        concrete_io=(
            "You paste a GitHub URL; you get a 1-page PDF guide and a 30-second video."
        ),
        audience=(
            "For programmers who can run Python scripts. Not for total newcomers — "
            "start with a Python tutorial first."
        ),
        prerequisites=["Python 3.10+", "Node.js 20+"],
        mental_model_3_boxes=["GitHub URL", "AI reads & explains", "Friendly PDF + video"],
        first_5_minutes=(
            "1. git clone https://github.com/seakon/repohelper "
            "2. pip install -r requirements.txt "
            "3. python -m api.main "
            "4. Open http://localhost:3000 "
            "5. Paste a repo URL and click Generate"
        ),
    )
    a.metaphor_story = [
        MetaphorSegment(
            detail="A diner walks into a busy kitchen with a messy handwritten order.",
            brief="Diner: I'm hungry, what can you make?",
        ),
        MetaphorSegment(
            detail="The chef squints at the ticket, fires up the stove, starts chopping.",
            brief="Chef: One Repo Special, coming up!",
        ),
        MetaphorSegment(
            detail="A beautifully plated dish slides across the pass to the diner.",
            brief="Diner: Wow, this is exactly what I wanted!",
        ),
    ]
    a.commit_timeline = CommitTimeline(
        commits=[
            CommitTimelineEntry(sha="abc12345", author="Alice", date="2026-04-01T00:00:00", message="feat: add video pipeline"),
            CommitTimelineEntry(sha="def67890", author="Bob", date="2025-12-01T00:00:00", message="initial commit"),
        ],
        contributors=[
            ContributorInfo(login="alice", name="Alice Smith", commit_count=120, followers=2500, public_repos=18),
            ContributorInfo(login="bob", name="Bob Wong", commit_count=42, followers=300, public_repos=6),
            ContributorInfo(login="charlie", name="Charlie", commit_count=8, followers=10),
        ],
        releases=[ReleaseInfo(tag="v0.2.0", date="2026-03-20T00:00:00", name="0.2.0")],
        stats=RepoStats(stars=1234, forks=88, watchers=42, license="MIT", pushed_at="2026-04-15T00:00:00Z"),
        first_commit_date="2025-12-01T00:00:00",
        latest_commit_date="2026-04-01T00:00:00",
        total_commits_scanned=2,
    )
    return a


# ---------------------------------------------------------------------------
# HTML-only tests (always run; no Playwright required)
# These verify the templates produce valid-looking HTML for each act.
# ---------------------------------------------------------------------------

def test_html_act3_io_contains_boxes_and_icons(sample_analyzed):
    from api.video.onboard_5act.acts import build_acts
    from api.video.onboard_5act.templates import build_act_html

    acts = build_acts(sample_analyzed)
    html_str = build_act_html(acts[2])
    assert "GitHub URL" in html_str
    assert "AI reads" in html_str or "AI reads & explains" in html_str
    assert "🔗" in html_str    # input icon
    assert "INPUT" in html_str
    assert "OUTPUT" in html_str
    assert "<!DOCTYPE html>" in html_str


def test_html_act4_usecase_has_three_panels_and_scene_bar(sample_analyzed):
    from api.video.onboard_5act.acts import build_acts
    from api.video.onboard_5act.templates import build_act_html

    acts = build_acts(sample_analyzed)
    html_str = build_act_html(acts[3])
    assert html_str.count("comic-panel") >= 3
    assert html_str.count("comic-bubble") >= 3
    assert "📍" in html_str   # scene-context bar marker
    assert "Problem" in html_str
    assert "Use" in html_str
    assert "Value" in html_str


def test_html_act5_setup_has_steps_and_prereqs(sample_analyzed):
    from api.video.onboard_5act.acts import build_acts
    from api.video.onboard_5act.templates import build_act_html

    acts = build_acts(sample_analyzed)
    html_str = build_act_html(acts[4])
    assert "Python 3.10+" in html_str
    assert "step-row" in html_str
    assert html_str.count("step-row") >= 3   # at least 3 steps
    assert "git clone" in html_str.lower()
    assert "📦" in html_str   # install icon


# ---------------------------------------------------------------------------
# Playwright PNG render tests (skipped when Chromium can't launch)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _CHROMIUM_OK,
                    reason="Playwright Chromium missing system libs (e.g. libnspr4); "
                           "run `sudo apt-get install -y libnspr4 libnss3` to enable")
def test_render_act3_io(sample_analyzed, preview_dir):
    """Act 3: input → process → output 3-box diagram with icons."""
    import asyncio
    from api.video.onboard_5act.acts import build_acts
    from api.video.onboard_5act.templates import render_act_to_png

    acts = build_acts(sample_analyzed)
    act3 = acts[2]
    out = preview_dir / "act3_io.png"
    if out.exists():
        out.unlink()

    asyncio.run(render_act_to_png(act3, str(out)))

    assert out.exists(), f"Act 3 PNG not generated at {out}"
    size = out.stat().st_size
    assert size > 5000, f"Act 3 PNG too small ({size} bytes) — likely empty"

    print(f"\n[Act 3] {out}  ({size:,} bytes)")


@pytest.mark.skipif(not _CHROMIUM_OK,
                    reason="Playwright Chromium missing system libs")
def test_render_act4_usecase(sample_analyzed, preview_dir):
    """Act 4: 3-panel use-case comic."""
    import asyncio
    from api.video.onboard_5act.acts import build_acts
    from api.video.onboard_5act.templates import render_act_to_png

    acts = build_acts(sample_analyzed)
    act4 = acts[3]
    out = preview_dir / "act4_usecase.png"
    if out.exists():
        out.unlink()

    asyncio.run(render_act_to_png(act4, str(out)))

    assert out.exists()
    assert out.stat().st_size > 5000

    print(f"\n[Act 4] {out}  ({out.stat().st_size:,} bytes)")


@pytest.mark.skipif(not _CHROMIUM_OK,
                    reason="Playwright Chromium missing system libs")
def test_render_act5_setup(sample_analyzed, preview_dir):
    """Act 5: setup checklist with prereqs + numbered steps."""
    import asyncio
    from api.video.onboard_5act.acts import build_acts
    from api.video.onboard_5act.templates import render_act_to_png

    acts = build_acts(sample_analyzed)
    act5 = acts[4]
    out = preview_dir / "act5_setup.png"
    if out.exists():
        out.unlink()

    asyncio.run(render_act_to_png(act5, str(out)))

    assert out.exists()
    assert out.stat().st_size > 5000

    print(f"\n[Act 5] {out}  ({out.stat().st_size:,} bytes)")


@pytest.mark.skipif(not _CHROMIUM_OK,
                    reason="Playwright Chromium missing system libs")
def test_render_all_5_acts_with_placeholders(sample_analyzed, preview_dir):
    """Render all 5 acts (Acts 1 & 2 are placeholder until commit 4)."""
    import asyncio
    from api.video.onboard_5act.acts import build_acts
    from api.video.onboard_5act.templates import render_act_to_png

    acts = build_acts(sample_analyzed)

    async def _render_all():
        # Use ONE event loop for all 5 renders so the Playwright executor's
        # browser singleton is shared (otherwise we'd init Chromium 5 times).
        for i, act in enumerate(acts, start=1):
            out = preview_dir / f"act{i}_{act['section']}.png"
            if out.exists():
                out.unlink()
            await render_act_to_png(act, str(out))

    asyncio.run(_render_all())

    print()
    for i, act in enumerate(acts, start=1):
        out = preview_dir / f"act{i}_{act['section']}.png"
        assert out.exists()
        assert out.stat().st_size > 5000
        print(f"[Act {i}] {act['section']:8s}  →  {out}  ({out.stat().st_size:,} bytes)")
