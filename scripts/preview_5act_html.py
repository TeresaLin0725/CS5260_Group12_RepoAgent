#!/usr/bin/env python3
"""
Preview the onboard_5act templates by dumping HTML files to disk.

Bypasses Chromium entirely — just writes the raw HTML for each of the
5 acts so you can open them in your normal browser (Edge / Chrome on
Windows) and see the visual layout.

Usage:
    python scripts/preview_5act_html.py
    # → writes ~/.adalflow/preview_5act/act{1..5}_*.html
    # then open them from Windows Explorer or:
    #   wslview ~/.adalflow/preview_5act/act3_io.html
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _build_sample_analyzed():
    """Same rich fixture the integration tests use, copied inline so this
    script has no test-tree dependency."""
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
        concrete_io="You paste a GitHub URL; you get a 1-page PDF guide and a 30-second video.",
        audience="For programmers who can run Python scripts.",
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
            CommitTimelineEntry(sha="abc12345", author="Alice Smith", date="2026-04-01T00:00:00", message="feat: add video pipeline"),
            CommitTimelineEntry(sha="def67890", author="Bob Wong", date="2025-12-01T00:00:00", message="initial commit"),
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


def main() -> int:
    from api.video.onboard_5act.acts import build_acts
    from api.video.onboard_5act.templates import build_act_html

    out_dir = Path(os.path.expanduser("~/.adalflow/preview_5act"))
    out_dir.mkdir(parents=True, exist_ok=True)

    analyzed = _build_sample_analyzed()
    acts = build_acts(analyzed)

    print(f"Writing 5 act HTML previews to {out_dir}\n")
    written: list[Path] = []
    for i, act in enumerate(acts, start=1):
        html_str = build_act_html(act)
        out = out_dir / f"act{i}_{act['section']}.html"
        out.write_text(html_str, encoding="utf-8")
        written.append(out)
        print(f"  Act {i}  ({act['section']:8s})  →  {out}  ({len(html_str):,} chars)")

    # Also create a one-page index that previews all 5 in iframes side-by-side.
    index_path = out_dir / "index.html"
    iframes = "\n".join(
        f'<div class="frame-wrap"><h2>Act {i} — {acts[i-1]["section"]}</h2>'
        f'<iframe src="{p.name}"></iframe></div>'
        for i, p in enumerate(written, start=1)
    )
    index_html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>onboard_5act preview</title>
<style>
  body {{ margin:0; padding:24px; background:#0a0d12; color:#eee;
         font-family: 'Segoe UI', system-ui, sans-serif; }}
  h1 {{ margin: 0 0 24px 0; }}
  .frame-wrap {{ margin-bottom: 32px; }}
  h2 {{ margin: 0 0 8px 0; color: #5d9bff; font-size: 18px; }}
  iframe {{ width: 1280px; height: 720px; border: 2px solid #333;
            border-radius: 12px; display: block; }}
</style></head><body>
<h1>🎬 onboard_5act — HTML preview</h1>
<p style="color:#888;">Each frame below is one act. Final video stacks these in
sequence with TTS narration. Resolution is fixed 1280×720 (the video frame size).</p>
{iframes}
</body></html>"""
    index_path.write_text(index_html, encoding="utf-8")
    print(f"\nIndex page (all 5 acts):  {index_path}")

    # Try to copy index path to a Windows-accessible location for one-click preview.
    win_dir = "/mnt/c/Users/Admin/Desktop/onboard_5act_preview"
    try:
        os.makedirs(win_dir, exist_ok=True)
        for p in written + [index_path]:
            (Path(win_dir) / p.name).write_text(p.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"\nAlso copied to Windows desktop:  {win_dir}\\index.html")
        print("Open that file in Edge/Chrome to see all 5 acts at once.")
    except Exception as e:
        print(f"\n(Could not copy to Windows desktop: {e})")
        print("Open the WSL paths above in your browser instead.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
